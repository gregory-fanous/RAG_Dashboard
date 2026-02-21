from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMCallResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class OpenAIProvider:
    def __init__(self, api_key: str, base_url: str, timeout_sec: float) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - exercised in integration only
            raise RuntimeError(
                "The 'openai' package is required for real execution mode. Install dependencies first."
            ) from exc

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec)

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> LLMCallResult:
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        text = ""
        if response.choices and response.choices[0].message is not None:
            text = response.choices[0].message.content or ""

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return LLMCallResult(
            text=text.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def embeddings(self, *, texts: list[str], model: str) -> tuple[list[list[float]], int]:
        if not texts:
            return [], 0

        response = self._client.embeddings.create(model=model, input=texts)
        vectors = [list(item.embedding) for item in response.data]
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        return vectors, prompt_tokens


class JsonParseError(ValueError):
    pass


def extract_json(payload: str) -> Any:
    text = payload.strip()
    if not text:
        raise JsonParseError("Empty model response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Recover from wrapped prose by extracting first object/array span.
    candidates = [
        (text.find("{"), text.rfind("}")),
        (text.find("["), text.rfind("]")),
    ]

    for start, end in candidates:
        if start >= 0 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue

    raise JsonParseError("Could not parse JSON from model response")
