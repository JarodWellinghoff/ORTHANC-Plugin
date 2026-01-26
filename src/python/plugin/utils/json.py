import json
from datetime import datetime, date, time

class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (datetime, date, time)):
            return o.isoformat()
        return super().default(o)

def encode_sse_event(event_name: str, payload: dict) -> bytes:
    try:
        json_payload = json.dumps(payload or {}, cls=DateTimeEncoder)
    except Exception as exc:
        json_payload = json.dumps({"eventType": "serialization-error", "message": str(exc)}, cls=DateTimeEncoder)
    lines = json_payload.splitlines() or [""]
    data_section = "\n".join(f"data: {line}" for line in lines)
    return f"event: {event_name}\n{data_section}\n\n".encode("utf-8")
