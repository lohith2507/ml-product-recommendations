"""
Vertex AI client for Gemini (chat + tools) and text embeddings.

Uses the google-genai SDK. Gemini chat models for this project are served from
the ``global`` region (e.g. gemini-3.5-flash); embeddings stay on us-central1.
"""

from __future__ import annotations

import base64
import json
import os
from types import SimpleNamespace
from typing import Any, Optional

DEFAULT_PROJECT = "shopping-agent-513607"
DEFAULT_CHAT_LOCATION = "global"
DEFAULT_EMBED_LOCATION = "us-central1"
DEFAULT_CHAT_MODEL = "gemini-3.5-flash"
DEFAULT_FALLBACK_MODEL = "gemini-3.1-flash-lite"
DEFAULT_EMBED_MODEL = "text-embedding-005"


def project_id() -> str:
    return (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT")
        or os.environ.get("VERTEX_PROJECT")
        or DEFAULT_PROJECT
    )


def chat_location() -> str:
    return os.environ.get("VERTEX_CHAT_LOCATION") or os.environ.get("GOOGLE_CLOUD_LOCATION") or DEFAULT_CHAT_LOCATION


def embed_location() -> str:
    return os.environ.get("VERTEX_EMBED_LOCATION") or DEFAULT_EMBED_LOCATION


def location() -> str:
    """Backward-compatible alias used by setup scripts."""
    return embed_location()


def chat_model() -> str:
    return os.environ.get("VERTEX_CHAT_MODEL") or DEFAULT_CHAT_MODEL


def fallback_model() -> str:
    return os.environ.get("VERTEX_FALLBACK_MODEL") or DEFAULT_FALLBACK_MODEL


def embed_model() -> str:
    return os.environ.get("VERTEX_EMBED_MODEL") or DEFAULT_EMBED_MODEL


def vertex_enabled() -> bool:
    if os.environ.get("USE_VERTEX", "").lower() in ("0", "false", "no"):
        return False
    return True


def _genai_client(loc: str):
    from google import genai

    return genai.Client(vertexai=True, project=project_id(), location=loc)


def _openai_tools_to_genai(tools: Optional[list]) -> Optional[list]:
    if not tools:
        return None
    from google.genai import types

    decls = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not fn:
            continue
        decls.append(
            types.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description") or "",
                parameters_json_schema=fn.get("parameters") or {"type": "object", "properties": {}},
            )
        )
    return [types.Tool(function_declarations=decls)] if decls else None


def _messages_to_genai(messages: list[dict]) -> tuple[Optional[str], list]:
    """Split OpenAI-style messages into (system_instruction, genai contents)."""
    from google.genai import types

    system = None
    contents: list[Any] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            system = msg.get("content") or system
            continue

        if role == "tool":
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=msg.get("name") or "tool",
                            response={"result": _parse_tool_payload(msg.get("content"))},
                        )
                    ],
                )
            )
            continue

        if role == "assistant":
            parts = []
            if msg.get("content"):
                parts.append(types.Part.from_text(text=msg["content"]))
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                args = _parse_tool_payload(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    args = {"value": args}
                fc_kwargs = {"name": fn.get("name") or "tool", "args": args}
                if call.get("id"):
                    fc_kwargs["id"] = call["id"]
                part_kwargs = {"function_call": types.FunctionCall(**fc_kwargs)}
                # Gemini 3.x requires the original thought_signature on replayed tool calls.
                sig_b64 = call.get("thought_signature")
                if sig_b64:
                    part_kwargs["thought_signature"] = base64.b64decode(sig_b64)
                parts.append(types.Part(**part_kwargs))
            if parts:
                contents.append(types.Content(role="model", parts=parts))
            continue

        # user
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=str(msg.get("content") or ""))])
        )
    return system, contents


def _parse_tool_payload(raw):
    if isinstance(raw, (dict, list)):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {"raw": str(raw)}


def _genai_response_to_openai(response) -> SimpleNamespace:
    """Map a google-genai response onto the OpenAI message shape ShoppingAgent expects."""
    text_chunks = []
    tool_calls = []
    parts = []
    try:
        parts = response.candidates[0].content.parts or []
    except Exception:
        parts = []

    for i, part in enumerate(parts):
        fc = getattr(part, "function_call", None)
        if fc is not None and getattr(fc, "name", None):
            sig = getattr(part, "thought_signature", None)
            tool_calls.append(
                SimpleNamespace(
                    id=getattr(fc, "id", None) or f"call_{i}_{fc.name}",
                    type="function",
                    thought_signature=base64.b64encode(sig).decode("ascii") if sig else None,
                    function=SimpleNamespace(
                        name=fc.name,
                        arguments=json.dumps(dict(fc.args or {})),
                    ),
                )
            )
        elif getattr(part, "text", None):
            text_chunks.append(part.text)

    message = SimpleNamespace(
        content="".join(text_chunks) if text_chunks else None,
        tool_calls=tool_calls or None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _GenAIChatCompletions:
    def __init__(self, client):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: list,
        tools: Optional[list] = None,
        tool_choice: Any = "auto",
        temperature: float = 0.3,
        max_tokens: int = 1200,
        **_kwargs,
    ):
        from google.genai import types

        system, contents = _messages_to_genai(messages)
        # Attach tool names onto tool-role messages for FunctionResponse (genai needs the name).
        # Our converter already defaults name; enrich from the preceding assistant tool_calls when possible.
        genai_tools = _openai_tools_to_genai(tools)
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system:
            config_kwargs["system_instruction"] = system
        if genai_tools:
            config_kwargs["tools"] = genai_tools
            if tool_choice == "none":
                config_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="NONE")
                )
            elif tool_choice == "auto":
                config_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                )

        # Fix tool response names using the previous assistant tool_calls list.
        contents = _attach_tool_names(messages, contents)

        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return _genai_response_to_openai(response)


def _attach_tool_names(messages: list[dict], contents: list) -> list:
    """google-genai FunctionResponse needs the function name; recover it from prior tool_calls."""
    named: list[dict] = []
    tool_names_queue: list[str] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            tool_names_queue = [
                (c.get("function") or {}).get("name") or "tool"
                for c in (msg.get("tool_calls") or [])
            ]
            named.append(msg)
        elif msg.get("role") == "tool":
            name = msg.get("name")
            if not name and tool_names_queue:
                name = tool_names_queue.pop(0)
            named.append({**msg, "name": name or "tool"})
        else:
            named.append(msg)
    _, rebuilt = _messages_to_genai(named)
    return rebuilt


class GenAIOpenAIAdapter:
    """Exposes ``.chat.completions.create`` like the OpenAI SDK, backed by Vertex Gemini."""

    def __init__(self, loc: str | None = None):
        self._client = _genai_client(loc or chat_location())
        self.chat = SimpleNamespace(completions=_GenAIChatCompletions(self._client))


def get_vertex_openai_client(project: str | None = None, loc: str | None = None):
    """Return an OpenAI-shaped client for Vertex Gemini (google-genai under the hood)."""
    _ = project  # project comes from env / default
    return GenAIOpenAIAdapter(loc=loc or chat_location())


def get_gemini_api_client():
    """Optional Google AI Studio key path (not Vertex)."""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    from openai import OpenAI

    return OpenAI(
        api_key=key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


def agent_clients():
    """Ordered (client, model) pairs for the shopping agent."""
    clients = []
    try:
        client = get_vertex_openai_client()
        clients.append((client, chat_model()))
        if fallback_model() != chat_model():
            clients.append((client, fallback_model()))
    except Exception as e:
        print(f"⚠️ Vertex client unavailable: {e}")

    studio = get_gemini_api_client()
    if studio is not None:
        clients.append((studio, os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")))
    return clients


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    if not texts:
        return []
    from google.genai import types

    client = _genai_client(embed_location())
    result = client.models.embed_content(
        model=embed_model(),
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [list(e.values) for e in result.embeddings]
