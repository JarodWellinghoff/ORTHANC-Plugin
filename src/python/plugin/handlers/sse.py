import queue, time as pytime
from datetime import datetime, timezone
import orthanc
from ..utils.http import handle_cors_preflight, set_cors_headers
from ..utils.json import encode_sse_event
from .. import config
from sse_manager import sse_manager
from progress_tracker import progress_tracker

def StreamChoProgress(output: orthanc.RestOutput, url: str, **request) -> None:
    if handle_cors_preflight(output, request):
        return
    if request.get('method') != 'GET':
        output.SendMethodNotAllowed('GET'); return

    set_cors_headers(output)
    output.SetHttpHeader('Cache-Control', 'no-cache')
    output.SetHttpHeader('Connection', 'keep-alive')
    output.SetHttpHeader('X-Accel-Buffering', 'no')
    output.StartStreamAnswer('text/event-stream')
    output.SendStreamChunk(f"retry: {config.SSE_RETRY_MILLISECONDS}\n\n".encode('utf-8'))

    client_id, channel = sse_manager.add_client()
    heartbeat_deadline = pytime.time() + config.SSE_HEARTBEAT_SECONDS
    try:
        snapshot = {
            'eventType': 'snapshot',
            'active': progress_tracker.get_all_active_calculations(),
            'history': progress_tracker.get_calculation_history(),
            'serverTime': datetime.now(timezone.utc).isoformat(),
        }
        output.SendStreamChunk(encode_sse_event('snapshot', snapshot))

        while True:
            now = pytime.time()
            timeout = max(0.1, heartbeat_deadline - now)
            try:
                message = channel.get(timeout=timeout)
            except queue.Empty:
                output.SendStreamChunk(b":heartbeat\n\n")
                heartbeat_deadline = pytime.time() + config.SSE_HEARTBEAT_SECONDS
                continue

            if not isinstance(message, dict): 
                continue
            event = message.get('event', 'cho-calculation')
            payload = message.get('data') or {}
            if isinstance(payload, dict) and 'timestamp' not in payload:
                ts = message.get('timestamp')
                if ts is not None:
                    payload['timestamp'] = datetime.fromtimestamp(ts, timezone.utc).isoformat()

            output.SendStreamChunk(encode_sse_event(event, payload))
            heartbeat_deadline = pytime.time() + config.SSE_HEARTBEAT_SECONDS
    except BrokenPipeError:
        pass
    finally:
        sse_manager.remove_client(client_id)
