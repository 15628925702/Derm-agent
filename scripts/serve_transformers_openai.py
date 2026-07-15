from __future__ import annotations

import argparse
import base64
import binascii
import gc
import json
import logging
import os
import time
from collections.abc import Mapping
from io import BytesIO
from typing import Any

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel


LOGGER = logging.getLogger("transformers_openai")
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


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _move_to_device(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: _move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [_move_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(_move_to_device(value, device) for value in batch)
    return batch


class TransformersOpenAIServer:
    def __init__(
        self,
        *,
        model_path: str,
        model_name: str,
        backend: str,
        adapter_path: str | None,
        device: str,
        dtype: str,
        max_new_tokens_default: int,
        device_map: str | None,
        gpu_max_memory_gb: float | None,
        cpu_max_memory_gb: float | None,
        load_in_4bit: bool,
        load_in_8bit: bool,
        offload_folder: str | None,
    ) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.backend = backend
        self.adapter_path = adapter_path
        self.device_name = device
        self.dtype = _resolve_dtype(dtype)
        self.max_new_tokens_default = max_new_tokens_default
        self.device_map = str(device_map or "").strip() or None
        self.gpu_max_memory_gb = gpu_max_memory_gb
        self.cpu_max_memory_gb = cpu_max_memory_gb
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.offload_folder = offload_folder
        self.hulumed_image_max_tokens = int(os.environ.get("HULUMED_IMAGE_MAX_TOKENS", "256"))
        self.hulumed_image_merge_size = int(os.environ.get("HULUMED_IMAGE_MERGE_SIZE", "1"))
        self.generation_use_cache = str(os.environ.get("DERMAGENT_HULUMED_USE_CACHE", "0")).strip().lower() in {"1", "true", "yes", "on"}

        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Choose only one of --load-in-4bit or --load-in-8bit.")

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        self.device = torch.device("cuda" if device == "cuda" else device)

        if backend in {"mllama", "mllama_lora"}:
            self._load_mllama()
        elif backend == "hulumed":
            self._load_hulumed()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self.model.eval()
        LOGGER.info(
            "Loaded direct Transformers server: backend=%s model=%s adapter=%s device=%s dtype=%s device_map=%s load_in_4bit=%s load_in_8bit=%s hulumed_image_max_tokens=%s hulumed_image_merge_size=%s pid=%s",
            self.backend,
            self.model_path,
            self.adapter_path or "",
            self.device,
            self.dtype,
            self.device_map or "",
            self.load_in_4bit,
            self.load_in_8bit,
            self.hulumed_image_max_tokens,
            self.hulumed_image_merge_size,
            os.getpid(),
        )

    @staticmethod
    def _memory_limit_mib(memory_gb: float | None) -> str | None:
        if memory_gb is None or memory_gb <= 0:
            return None
        return f"{int(memory_gb * 1024)}MiB"

    def _build_model_load_kwargs(self) -> tuple[dict[str, Any], bool]:
        resolved_device_map: Any = self.device_map
        if resolved_device_map == "single":
            resolved_device_map = {"": 0}

        kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        if self.offload_folder:
            kwargs["offload_folder"] = self.offload_folder

        max_memory: dict[Any, str] = {}
        gpu_limit = self._memory_limit_mib(self.gpu_max_memory_gb)
        cpu_limit = self._memory_limit_mib(self.cpu_max_memory_gb)
        if gpu_limit and torch.cuda.is_available():
            max_memory[0] = gpu_limit
        if cpu_limit:
            max_memory["cpu"] = cpu_limit

        if self.load_in_4bit or self.load_in_8bit:
            from transformers import BitsAndBytesConfig

            quantization_kwargs: dict[str, Any] = {}
            if self.load_in_4bit:
                quantization_kwargs.update(
                    {
                        "load_in_4bit": True,
                        "bnb_4bit_compute_dtype": self.dtype,
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_use_double_quant": True,
                    }
                )
            if self.load_in_8bit:
                quantization_kwargs["load_in_8bit"] = True
                quantization_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
            kwargs["quantization_config"] = BitsAndBytesConfig(**quantization_kwargs)
            kwargs["device_map"] = resolved_device_map or "auto"
        elif resolved_device_map or max_memory:
            kwargs["device_map"] = resolved_device_map or "auto"

        if max_memory and resolved_device_map != {"": 0}:
            kwargs["max_memory"] = max_memory

        use_explicit_to = "device_map" not in kwargs
        return kwargs, use_explicit_to

    def _load_mllama(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        load_kwargs, use_explicit_to = self._build_model_load_kwargs()
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            **load_kwargs,
        )
        if self.backend == "mllama_lora":
            if not self.adapter_path:
                raise ValueError("mllama_lora backend requires --adapter-path.")
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                self.adapter_path,
                is_trainable=False,
                low_cpu_mem_usage=True,
                key_mapping={
                    r"^language_model\.model\.": "model.language_model.",
                },
            )
        self.model = model.to(self.device) if use_explicit_to else model

    def _load_hulumed(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor
        import numpy as np

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # Monkey patch _get_downsampled_grid_sizes to handle numpy arrays
        original_get_downsampled = self.processor._get_downsampled_grid_sizes
        def patched_get_downsampled(image_inputs):
            import numpy as np
            # Deep convert all numpy arrays to torch tensors
            def convert_to_torch(obj):
                if isinstance(obj, np.ndarray):
                    return torch.from_numpy(obj)
                elif isinstance(obj, Mapping):
                    return {k: convert_to_torch(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_torch(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_to_torch(item) for item in obj)
                return obj

            converted_inputs = convert_to_torch(image_inputs)
            return original_get_downsampled(converted_inputs)

        self.processor._get_downsampled_grid_sizes = patched_get_downsampled

        load_kwargs, use_explicit_to = self._build_model_load_kwargs()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            **load_kwargs,
        )
        if use_explicit_to:
            self.model = self.model.to(self.device)

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
        max_new_tokens = min(
            request.max_tokens or self.max_new_tokens_default,
            self.max_new_tokens_default,
        )

        LOGGER.info(
            "Received request: backend=%s model=%s has_image=%s max_tokens=%s prompt_chars=%s",
            self.backend,
            request.model,
            image is not None,
            max_new_tokens,
            len(effective_prompt),
        )
        try:
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            raw_output = self._generate(
                prompt_text=effective_prompt,
                image=image,
                temperature=request.temperature or 0.0,
                max_new_tokens=max_new_tokens,
            )
            if expects_json and not self._looks_like_json(raw_output):
                raw_output = self._generate(
                    prompt_text=self._build_json_repair_prompt(effective_prompt, raw_output),
                    image=image,
                    temperature=0.0,
                    max_new_tokens=max_new_tokens,
                )
        except torch.cuda.OutOfMemoryError as exc:
            LOGGER.exception("Generation OOM")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise HTTPException(status_code=500, detail=f"CUDA OOM: {exc}") from exc
        except Exception as exc:
            LOGGER.exception("Generation failed")
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
        finally:
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

        content = self._coerce_to_json_text(raw_output) if expects_json else raw_output
        now = int(time.time())
        return {
            "id": f"chatcmpl-transformers-{now}",
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

    def _generate(
        self,
        *,
        prompt_text: str,
        image: Image.Image | None,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        if self.backend in {"mllama", "mllama_lora"}:
            return self._generate_mllama(
                prompt_text=prompt_text,
                image=image,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        return self._generate_hulumed(
            prompt_text=prompt_text,
            image=image,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def _generate_mllama(
        self,
        *,
        prompt_text: str,
        image: Image.Image | None,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        content: str | list[dict[str, Any]]
        if image is None:
            content = prompt_text
            images = None
        else:
            content = [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]
            images = image

        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(
            images=images,
            text=prompt,
            return_tensors="pt",
        )
        inputs = _move_to_device(inputs, self.device)
        output_ids = self.model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            max_new_tokens=max_new_tokens,
            use_cache=self.generation_use_cache,
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def _generate_hulumed(
        self,
        *,
        prompt_text: str,
        image: Image.Image | None,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        if image is None:
            conversation: list[dict[str, Any]] = [
                {"role": "user", "content": prompt_text},
            ]
            images = None
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            images = [("image", image)]

        inputs = self._prepare_hulumed_inputs(conversation=conversation, images=images)
        inputs = _move_to_device(inputs, self.device)
        inputs = self._coerce_hulumed_input_dtypes(inputs)
        output_ids = self.model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            max_new_tokens=max_new_tokens,
            use_cache=self.generation_use_cache,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        return self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def _prepare_hulumed_inputs(
        self,
        *,
        conversation: list[dict[str, Any]],
        images: list[tuple[str, Image.Image]] | None,
    ) -> Any:
        import numpy as np

        if images is None:
            image_inputs: dict[str, Any] = {}
        else:
            if hasattr(self.processor, "image_processor") and hasattr(self.processor.image_processor, "max_tokens"):
                self.processor.image_processor.max_tokens = self.hulumed_image_max_tokens
            image_inputs = self.processor.process_images(
                images,
                merge_size=self.hulumed_image_merge_size,
            )
            image_inputs = self._convert_numpy_to_torch(image_inputs)

        prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Convert image_inputs again before passing to process_text
        image_inputs = self._convert_numpy_to_torch(image_inputs)

        result = self.processor.process_text(
            prompt,
            image_inputs,
            return_tensors="pt",
        )

        # Convert result and image_inputs one more time after process_text
        result = self._convert_numpy_to_torch(result)
        image_inputs = self._convert_numpy_to_torch(image_inputs)

        return result | image_inputs

    def _coerce_hulumed_input_dtypes(self, inputs: Any) -> Any:
        if isinstance(inputs, torch.Tensor) and torch.is_floating_point(inputs):
            return inputs.to(dtype=self.dtype)
        if isinstance(inputs, Mapping):
            return {key: self._coerce_hulumed_input_dtypes(value) for key, value in inputs.items()}
        if isinstance(inputs, list):
            return [self._coerce_hulumed_input_dtypes(value) for value in inputs]
        if isinstance(inputs, tuple):
            return tuple(self._coerce_hulumed_input_dtypes(value) for value in inputs)
        return inputs

    @staticmethod
    def _convert_numpy_to_torch(data: Any) -> Any:
        import numpy as np
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        if isinstance(data, Mapping):
            return {key: TransformersOpenAIServer._convert_numpy_to_torch(value) for key, value in data.items()}
        if isinstance(data, list):
            return [TransformersOpenAIServer._convert_numpy_to_torch(value) for value in data]
        if isinstance(data, tuple):
            return tuple(TransformersOpenAIServer._convert_numpy_to_torch(value) for value in data)
        return data

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
                    image = TransformersOpenAIServer._decode_image_url(item.image_url.url)
            if parts:
                text_parts.append(f"{message.role}: " + "\n".join(parts))
        return "\n\n".join(text_parts).strip(), image

    @staticmethod
    def _decode_image_url(url: str) -> Image.Image:
        if not url.startswith("data:"):
            raise HTTPException(status_code=400, detail="Only data:image base64 URLs are supported by this local server.")
        try:
            _, encoded = url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image payload: {exc}") from exc
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        max_dim = 256
        width, height = image.size
        longest = max(width, height)
        if longest > max_dim:
            scale = max_dim / float(longest)
            resized = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            image = image.resize(resized, Image.Resampling.LANCZOS)
        return image


def create_app(server: TransformersOpenAIServer, api_key: str) -> FastAPI:
    app = FastAPI(title="Transformers OpenAI-Compatible Server")

    def authorize(authorization: str | None) -> None:
        if not api_key:
            return
        if authorization != f"Bearer {api_key}":
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
    parser = argparse.ArgumentParser(description="Serve local Transformers models through a minimal OpenAI-compatible API.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--backend", choices=["mllama", "mllama_lora", "hulumed"], required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens-default", type=int, default=256)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--gpu-max-memory-gb", type=float, default=None)
    parser.add_argument("--cpu-max-memory-gb", type=float, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--offload-folder", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = TransformersOpenAIServer(
        model_path=args.model_path,
        model_name=args.served_model_name,
        backend=args.backend,
        adapter_path=args.adapter_path,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens_default=args.max_new_tokens_default,
        device_map=args.device_map,
        gpu_max_memory_gb=args.gpu_max_memory_gb,
        cpu_max_memory_gb=args.cpu_max_memory_gb,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        offload_folder=args.offload_folder,
    )
    app = create_app(server, api_key=args.api_key)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
