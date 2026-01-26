import orthanc

from .registry import register_routes, register_onchange
from .utils.http import set_cors_headers, handle_cors_preflight  # re-export if useful

def init():
    register_routes()
    register_onchange()

init()
