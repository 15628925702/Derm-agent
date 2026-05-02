from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoProcessor, AutoModelForCausalLM, MllamaForConditionalGeneration
from peft import PeftModel


LOGGER = logging.getLogger("transformers_vlm_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


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


class TransformersVLMServer:
    def __init__(
        self,
        model_path: str,
        model_name: str,
        model_type: str = "auto",
        lora_adapter_path: str | None = None,
        device: str = "cuda",
        max_new_tokens_default: int = 768,
    ) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.model_type = model_type
        self.lora_adapter_path = lora_adapter_path
        self.device = device
        self.max_new_tokens_default = max_new_tokens_default

        LOGGER.info(
            "Loading model: model_path=%s model_type=%s lora_adapter=%s device=%s",
            model_path,
            model_type,
            lora_adapter_path,
            device,
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Load model based on type
        if model_type == "mllama" or "Llama-3.2" in model_path or "llama-3.2" in model_path.lower():
            LOGGER.info("Loading MllamaForConditionalGeneration model")
            model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            LOGGER.info("Loading AutoModelForCausalLM model")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        # Load LoRA adapter if specified
        if lora_adapter_path:
            LOGGER.info("Loading LoRA adapter from: %s", lora_adapter_path)
            model = PeftModel.from_pretrained(model, lora_adapter_path)

        model.eval()
        self.model = model

        LOGGER.info(
            "Model loaded successfully: model_name=%s device=%s pid=%s",
            self.model_name,
            self.device,
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
                LOGGER.info("First-pass output was not valid JSON; retrying with JSON repair prompt.")
                raw_output = self._generate(
                    prompt_text=self._build_json_repair_prompt(effective_prompt, raw_output),
                    image=image,
                    temperature=0.0,
                    max_new_tokens=min(request.max_tokens or self.max_new_tokens_default, self.max_new_tokens_default),
                )
        except torch.cuda.OutOfMemoryError as exc:
            LOGGER.exception("Generation OOM")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise HTTPException(status_code=500, detail=f"CUDA OOM: {exc}") from exc
        except Exception as exc:
            LOGGER.exception("Generation failed")
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

        content = self._coerce_to_json_text(raw_output) if expects_json else raw_output
        now = int(time.time())
        LOGGER.info("Completed chat request: content_chars=%s", len(content))

        return {
            "id": f"chatcmpl-{self.model_name}-{now}",
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
        # Check if processor supports conversation-based input (like Hulu-Med)
        has_conversation_support = hasattr(self.processor, '__call__') and 'conversation' in str(self.processor.__call__.__code__.co_varnames)
        prompt_for_decode: str | None = None

        # Prepare inputs
        if image is not None:
            if has_conversation_support:
                # Use conversation format for models like Hulu-Med
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ]
                inputs = self.processor(
                    conversation=conversation,
                    add_system_prompt=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                prompt_for_decode = self.processor.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_system_prompt=True,
                    add_generation_prompt=True,
                )
            else:
                # Standard format for other models
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ]
                inputs = self.processor(
                    images=image,
                    text=self.processor.apply_chat_template(messages, add_generation_prompt=True),
                    return_tensors="pt",
                )

            # Move to device and convert dtype
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

            # Convert image tensors to match model dtype (bfloat16)
            if "pixel_values" in inputs and hasattr(inputs["pixel_values"], 'dtype'):
                if inputs["pixel_values"].dtype == torch.float32:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        else:
            messages = [{"role": "user", "content": prompt_text}]
            if has_conversation_support:
                inputs = self.processor(
                    conversation=messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                prompt_for_decode = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                inputs = self.processor(
                    text=self.processor.apply_chat_template(messages, add_generation_prompt=True),
                    return_tensors="pt",
                )
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
            )

        # Decode output. Hulu-Med's remote generate path uses inputs_embeds and may
        # return only generated ids, while standard VLMs return prompt + generated ids.
        if has_conversation_support:
            output_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                use_think=False,
            )[0].strip()
            if prompt_for_decode:
                prompt_text_decoded = self.processor.decode(
                    inputs["input_ids"][0],
                    skip_special_tokens=True,
                    use_think=False,
                ).strip()
                output_text = self._strip_prompt_from_output(output_text, prompt_for_decode, prompt_text_decoded)
        else:
            generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
            output_text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        return output_text

    @staticmethod
    def _strip_prompt_from_output(output_text: str, *prompt_candidates: str | None) -> str:
        stripped = output_text.strip()
        for prompt in prompt_candidates:
            if prompt:
                prompt = prompt.strip()
                if prompt and stripped.startswith(prompt):
                    stripped = stripped[len(prompt):].strip()
        marker = "assistant\n"
        marker_index = stripped.rfind(marker)
        if marker_index >= 0:
            stripped = stripped[marker_index + len(marker):].strip()
        return stripped

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
                    image = TransformersVLMServer._decode_image_url(item.image_url.url)
            if parts:
                text_parts.append(f"{message.role}: " + "\n".join(parts))

        return "\n\n".join(text_parts).strip(), image

    @staticmethod
    def _decode_image_url(url: str) -> Image.Image:
        if not url.startswith("data:"):
            raise HTTPException(status_code=400, detail="Only data:image base64 URLs are supported.")
        try:
            _, encoded = url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image payload: {exc}") from exc
        return Image.open(BytesIO(image_bytes)).convert("RGB")


def create_app(server: TransformersVLMServer, api_key: str) -> FastAPI:
    app = FastAPI(title="Transformers VLM OpenAI-Compatible Server")

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
    parser = argparse.ArgumentParser(description="Serve VLM models through OpenAI-compatible API.")
    parser.add_argument("--model-path", required=True, help="Path to the model")
    parser.add_argument("--served-model-name", required=True, help="Model name to serve")
    parser.add_argument("--model-type", default="auto", help="Model type: auto, mllama, qwen2, etc.")
    parser.add_argument("--lora-adapter-path", default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens-default", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = TransformersVLMServer(
        model_path=args.model_path,
        model_name=args.served_model_name,
        model_type=args.model_type,
        lora_adapter_path=args.lora_adapter_path,
        device=args.device,
        max_new_tokens_default=args.max_new_tokens_default,
    )
    app = create_app(server, api_key=args.api_key)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
