import json

from fastapi import Request


def request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def normalize_stream_part(part: object) -> dict[str, object] | None:
    if isinstance(part, dict):
        return part

    if (
        isinstance(part, tuple)
        and len(part) == 2
        and isinstance(part[0], str)
    ):
        stream_type, data = part
        return {"type": stream_type, "data": data}

    return None


def stringify_message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts)
    return str(content)
