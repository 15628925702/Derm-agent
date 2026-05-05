from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

import requests
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Health check for the local Qwen OpenAI-compatible server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL.")
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="API key for the local server. Defaults to OPENAI_API_KEY or EMPTY.",
    )
    parser.add_argument("--model", default=None, help="Optional explicit model name to use for the test request.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout in seconds for HTTP and SDK requests.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def fetch_models(base_url: str, api_key: str, timeout: float) -> dict[str, Any]:
    models_url = f"{base_url.rstrip('/')}/models"
    logging.info("Checking model listing endpoint: %s", models_url)
    response = requests.get(
        models_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError("Unexpected /models response format.")
    if not payload["data"]:
        raise ValueError("No models were returned by /v1/models.")
    return payload


def resolve_model_id(models_payload: dict[str, Any], explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model
    first_entry = models_payload["data"][0]
    model_id = first_entry.get("id")
    if not model_id:
        raise ValueError("First model entry does not contain an id.")
    return model_id


def run_minimal_chat(base_url: str, api_key: str, model: str, timeout: float) -> str:
    logging.info("Sending minimal chat completion request with model: %s", model)
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a health check assistant."},
            {"role": "user", "content": "Reply with OK only."},
        ],
        max_tokens=8,
        temperature=0,
    )
    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise ValueError("The model returned an empty response.")
    return content.strip()


def main() -> int:
    args = parse_args()
    configure_logging()

    try:
        logging.info("Checking service availability at %s", args.base_url)
        models_payload = fetch_models(args.base_url, args.api_key, args.timeout)
        model_id = resolve_model_id(models_payload, args.model)
        response_text = run_minimal_chat(args.base_url, args.api_key, model_id, args.timeout)
    except Exception as exc:
        logging.error("Qwen server health check failed: %s", exc)
        return 1

    logging.info("Service online: OK")
    logging.info("/v1/models reachable: OK")
    logging.info("Minimal request completed: OK")
    logging.info("Model response: %s", response_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
