from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import os
import shutil
import sys
import time
from functools import wraps
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import models_root, state_root

DEFAULT_MM_SKIN_ROOT = Path(os.environ.get("MM_SKIN_ROOT", Path(__file__).resolve().parents[2] / "MM-Skin"))
MM_SKIN_ROOT = DEFAULT_MM_SKIN_ROOT
if str(MM_SKIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MM_SKIN_ROOT))

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


LOGGER = logging.getLogger("skinvl_openai")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def _resolve_vision_tower_path(raw_path: str | None) -> str | None:
    if not raw_path:
        return None

    explicit = os.environ.get("SKINVL_VISION_TOWER_PATH")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))

    raw = Path(raw_path)
    candidates.append(raw)
    candidates.append(models_root() / raw.name)
    candidates.append(Path("/data/gh/models") / raw.name)

    for candidate in candidates:
        try:
            exists = candidate.exists()
        except PermissionError:
            exists = False
        if exists:
            return str(candidate.resolve())
    return None


def _prepare_model_path(model_path: str) -> str:
    source = Path(model_path).resolve()
    config_path = source / "config.json"
    if not config_path.exists():
        return str(source)

    with config_path.open() as f:
        config = json.load(f)

    current_vision_tower = config.get("mm_vision_tower")
    resolved_vision_tower = _resolve_vision_tower_path(current_vision_tower)
    if not current_vision_tower or current_vision_tower == resolved_vision_tower:
        return str(source)
    if not resolved_vision_tower:
        return str(source)

    runtime_view_id = os.getenv("DERMAGENT_SKINVL_RUNTIME_VIEW_ID") or os.getenv("PORT") or str(os.getpid())
    safe_runtime_view_id = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in runtime_view_id
    )
    runtime_dir = state_root() / "runtime" / f"skinvl_model_view_{safe_runtime_view_id}"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    for child in source.iterdir():
        target = runtime_dir / child.name
        if child.name == "config.json":
            continue
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.symlink_to(child, target_is_directory=child.is_dir())

    config["mm_vision_tower"] = resolved_vision_tower
    with (runtime_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    LOGGER.info(
        "SkinVL model config view prepared: source=%s runtime=%s mm_vision_tower=%s",
        source,
        runtime_dir,
        resolved_vision_tower,
    )
    return str(runtime_dir)


def _patch_skinvl_generation_compat(model: Any) -> None:
    """Drop generation kwargs added by newer Transformers but unsupported by SkinVL."""
    original_forward = model.forward

    @wraps(original_forward)
    def forward_without_cache_position(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("cache_position", None)
        return original_forward(*args, **kwargs)

    model.forward = forward_without_cache_position
    LOGGER.info("Applied SkinVL generation compatibility patch for cache_position")


class ImageURLPayload(BaseModel):
    url: str


class ContentItem(BaseModel):
    type: str
    text: str | None = None
    image_url: ImageURLPayload | None = None


class MessagePayload(BaseModel):
    role: str
    content: str | list[ContentItem]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[MessagePayload]
    temperature: float | None = 0.0
    max_tokens: int | None = 512
    response_format: dict[str, Any] | None = None


class SkinVLServer:
    def __init__(
        self,
        model_path: str,
        model_name: str,
        device: str = "cuda",
        conv_mode: str = "chatml_direct",
        max_new_tokens_default: int = 768,
    ) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.conv_mode = conv_mode
        self.max_new_tokens_default = max_new_tokens_default

        load_model_path = _prepare_model_path(model_path)
        disable_torch_init()
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=load_model_path,
            model_base=None,
            model_name="llava-mistral",
            device=device,
            device_map="auto" if device == "cuda" else device,
        )
        _patch_skinvl_generation_compat(model)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
        LOGGER.info(
            "SkinVL model loaded: model_name=%s device=%s context_len=%s pid=%s",
            self.model_name,
            self.device,
            self.context_len,
            os.getpid(),
        )

    def list_models(self) -> dict[str, Any]:
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_name,
                    "object": "model",
                    "created": now,
                    "owned_by": "local",
                }
            ],
        }

    def chat_completion(self, request: ChatCompletionRequest) -> dict[str, Any]:
        prompt_text, image = self._extract_prompt_and_image(request.messages)
        expects_json = self._expects_json(request)
        effective_prompt = self._enforce_json_prompt(prompt_text) if expects_json else prompt_text
        LOGGER.info(
            "Received chat request: model=%s has_image=%s max_tokens=%s prompt_chars=%s",
            request.model,
            image is not None,
            request.max_tokens or self.max_new_tokens_default,
            len(effective_prompt),
        )
        try:
            raw_output = self._generate(
                prompt_text=effective_prompt,
                image=image,
                temperature=request.temperature or 0.0,
                max_new_tokens=min(request.max_tokens or self.max_new_tokens_default, self.max_new_tokens_default),
            )
            if expects_json and not self._looks_like_json(raw_output):
                LOGGER.info("SkinVL first-pass output was not valid JSON; retrying with JSON repair prompt.")
                raw_output = self._generate(
                    prompt_text=self._build_json_repair_prompt(effective_prompt, raw_output),
                    image=image,
                    temperature=0.0,
                    max_new_tokens=min(request.max_tokens or self.max_new_tokens_default, self.max_new_tokens_default),
                )
        except torch.cuda.OutOfMemoryError as exc:
            LOGGER.exception("SkinVL generation OOM")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise HTTPException(status_code=500, detail=f"SkinVL CUDA OOM: {exc}") from exc
        except Exception as exc:
            LOGGER.exception("SkinVL generation failed")
            raise HTTPException(status_code=500, detail=f"SkinVL generation failed: {exc}") from exc
        content = self._coerce_to_json_text(raw_output) if expects_json else raw_output
        now = int(time.time())
        LOGGER.info("Completed chat request: content_chars=%s", len(content))
        return {
            "id": f"chatcmpl-skinvl-{now}",
            "object": "chat.completion",
            "created": now,
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def _generate(self, *, prompt_text: str, image: Image.Image | None, temperature: float, max_new_tokens: int) -> str:
        conv = conv_templates[self.conv_mode].copy()
        user_prompt = prompt_text
        image_tensor = None
        image_size = None

        if image is not None:
            if getattr(self.model.config, "mm_use_im_start_end", False):
                user_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + user_prompt
            else:
                user_prompt = DEFAULT_IMAGE_TOKEN + "\n" + user_prompt
            image_size = image.size
            image_tensor = process_images([image], self.image_processor, self.model.config)
            if isinstance(image_tensor, list):
                image_tensor = [item.to(self.model.device, dtype=torch.float16) for item in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        input_ids = input_ids.to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size] if image_size is not None else None,
                do_sample=False,
                temperature=None,
                max_new_tokens=max_new_tokens,
                top_p=None,
                num_beams=1,
                use_cache=True,
            )

        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if output_text.startswith(prompt):
            output_text = output_text[len(prompt) :].strip()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        if stop_str and stop_str in output_text:
            output_text = output_text.split(stop_str)[0].strip()
        return output_text

    @staticmethod
    def _expects_json(request: ChatCompletionRequest) -> bool:
        return isinstance(request.response_format, dict) and request.response_format.get("type") == "json_object"

    @staticmethod
    def _enforce_json_prompt(prompt_text: str) -> str:
        return (
            prompt_text.rstrip()
            + "\n\nIMPORTANT:\n"
            + "Return valid JSON only.\n"
            + "Do not output markdown.\n"
            + "Do not output explanatory text before or after the JSON.\n"
            + "If you are unsure, still return the requested JSON schema with best-effort field values.\n"
        )

    @staticmethod
    def _build_json_repair_prompt(prompt_text: str, previous_output: str) -> str:
        return (
            prompt_text.rstrip()
            + "\n\nYour previous answer was not valid JSON.\n"
            + "Rewrite it as valid JSON only.\n"
            + "Do not add any prose.\n"
            + "Previous answer:\n"
            + previous_output.strip()
        )

    @staticmethod
    def _looks_like_json(output_text: str) -> bool:
        candidate = output_text.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 3:
                candidate = "\n".join(lines[1:-1]).strip()
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
        try:
            json.loads(candidate)
            return True
        except Exception:
            return False

    @staticmethod
    def _coerce_to_json_text(output_text: str) -> str:
        candidate = output_text.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 3:
                candidate = "\n".join(lines[1:-1]).strip()
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
        try:
            parsed = json.loads(candidate)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            return json.dumps({"raw_text": output_text}, ensure_ascii=False)

    @staticmethod
    def _extract_prompt_and_image(messages: list[MessagePayload]) -> tuple[str, Image.Image | None]:
        text_parts: list[str] = []
        image: Image.Image | None = None

        for message in messages:
            content = message.content
            if isinstance(content, str):
                text_parts.append(f"{message.role}: {content}")
                continue

            parts: list[str] = []
            for item in content:
                if item.type == "text" and item.text:
                    parts.append(item.text)
                elif item.type == "image_url" and item.image_url and item.image_url.url:
                    image = SkinVLServer._decode_image_url(item.image_url.url)
            if parts:
                text_parts.append(f"{message.role}: " + "\n".join(parts))

        return "\n\n".join(text_parts).strip(), image

    @staticmethod
    def _decode_image_url(url: str) -> Image.Image:
        if not url.startswith("data:"):
            raise HTTPException(status_code=400, detail="Only data:image base64 URLs are supported by the local SkinVL server.")
        try:
            _, encoded = url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image payload: {exc}") from exc
        return Image.open(BytesIO(image_bytes)).convert("RGB")


def create_app(server: SkinVLServer, api_key: str) -> FastAPI:
    app = FastAPI(title="SkinVL OpenAI-Compatible Server")

    def authorize(authorization: str | None) -> None:
        if not api_key:
            return
        expected = f"Bearer {api_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/v1/models")
    def list_models(authorization: str | None = Header(default=None)) -> JSONResponse:
        authorize(authorization)
        return JSONResponse(server.list_models())

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest, authorization: str | None = Header(default=None)) -> JSONResponse:
        authorize(authorization)
        return JSONResponse(server.chat_completion(request))

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve SkinVL-MM through a minimal OpenAI-compatible API.")
    parser.add_argument("--model-path", default=str(Path(__file__).resolve().parents[2] / "models" / "SkinVL-MM"))
    parser.add_argument("--served-model-name", default="SkinVL-MM")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conv-mode", default="chatml_direct")
    parser.add_argument("--max-new-tokens-default", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = SkinVLServer(
        model_path=args.model_path,
        model_name=args.served_model_name,
        device=args.device,
        conv_mode=args.conv_mode,
        max_new_tokens_default=args.max_new_tokens_default,
    )
    app = create_app(server, api_key=args.api_key)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
