import orthanc
from .. import config

def set_cors_headers(output: orthanc.RestOutput) -> None:
    output.SetHttpHeader('Access-Control-Allow-Origin', config.CORS_ALLOWED_ORIGIN)
    output.SetHttpHeader('Access-Control-Allow-Methods', config.CORS_ALLOWED_METHODS)
    output.SetHttpHeader('Access-Control-Allow-Headers', config.CORS_ALLOWED_HEADERS)
    if config.CORS_ALLOW_CREDENTIALS:
        output.SetHttpHeader('Access-Control-Allow-Credentials', 'true')

def handle_cors_preflight(output: orthanc.RestOutput, request: dict) -> bool:
    set_cors_headers(output)
    if request.get('method') == 'OPTIONS':
        output.SendHttpStatusCode(204)
        return True
    return False
