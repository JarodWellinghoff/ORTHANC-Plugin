import base64
from csv import excel
import io
import json
import os
import queue
import orthanc
import pandas as pd
import numpy as np
from ChangeType import ChangeType
from ResourceType import ResourceType
import psycopg2  # type: ignore
from datetime import datetime, date, time, timedelta, timezone
import xlwt
from results_storage import cho_storage
from progress_tracker import progress_tracker
from calculation_wrapper import run_cho_calculation_with_progress
from dicom_pull_manager import dicom_pull_manager
from sse_manager import sse_manager
import time as pytime
import threading
from patient_specific_calculation import create_lesion_model
from PIL import Image


TOKEN = orthanc.GenerateRestApiAuthorizationToken()
TEST_TYPE = os.getenv("INSTANCE_BEHAVIOR_TEST", "Global Noise")
SAVE_TYPE = os.getenv("INSTANCE_BEHAVIOR_SAVE", "true") == "true"

print(f"=== CHO Analysis Plugin Initialized ===")
print(f" - Test Type: {TEST_TYPE}")
print(f" - Save Type: {SAVE_TYPE}")

SSE_HEARTBEAT_SECONDS = 15
SSE_RETRY_MILLISECONDS = 4000


print("=== CHO Analysis Plugin Initialized ===")
print("Available endpoints:")
print("  - /cho-dashboard - View CHO results dashboard")
print("  - /cho-calculation-status - Get the status of a calculation")
print("  - /cho-active-calculations - Get all active calculations")
print("  - /cho-results - Get all CHO results (with pagination)")
print("  - /cho-results/{series_id} - Get CHO results for a specific series")
print("  - /static/{filename} - Serve static files (JS, CSS)")
print("  - Auto Analysis: Automatically triggered on stable series (if enabled)")
print("  - /dicom-modalities - List configured DICOM modalities")
print("  - /dicom-store/query - Query remote DICOM modalities")
print("  - /dicom-pull/batches - Manage scheduled DICOM pulls")
print("  - /dicom-pull/batches/{id} - Inspect scheduled pull details")
print("  - /dicom-pull/recover - Attempt to recover DICOM data immediately")

CORS_ALLOWED_ORIGIN = os.getenv("CORS_ALLOWED_ORIGIN", "*")
CORS_ALLOWED_METHODS = os.getenv(
    "CORS_ALLOWED_METHODS", "GET, POST, PUT, DELETE, OPTIONS"
)
CORS_ALLOWED_HEADERS = os.getenv("CORS_ALLOWED_HEADERS", "Content-Type, Authorization")
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"


def send_json(output: orthanc.RestOutput, obj: dict, status: int = 200) -> None:
    if status != 200:
        output.SendHttpStatusCode(status)
    output.AnswerBuffer(
        json.dumps(obj, indent=2, cls=DateTimeEncoder).encode("utf-8"),
        "application/json",
    )


def set_cors_headers(output: orthanc.RestOutput) -> None:
    output.SetHttpHeader("Access-Control-Allow-Origin", CORS_ALLOWED_ORIGIN)
    output.SetHttpHeader("Access-Control-Allow-Methods", CORS_ALLOWED_METHODS)
    output.SetHttpHeader("Access-Control-Allow-Headers", CORS_ALLOWED_HEADERS)
    if CORS_ALLOW_CREDENTIALS:
        output.SetHttpHeader("Access-Control-Allow-Credentials", "true")


def handle_cors_preflight(output: orthanc.RestOutput, request: dict) -> bool:
    """Apply CORS headers and finish OPTIONS preflight when needed."""
    set_cors_headers(output)
    if request.get("method") == "OPTIONS":
        output.SendHttpStatusCode(204)
        return True
    return False


def parse_bool(value, default=False):
    """Best-effort boolean parsing for JSON or query parameters."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off"):
            return False
        return default
    return bool(value)


def _encode_sse_event(event_name: str, payload: dict) -> bytes:
    """Serialize an SSE event payload to bytes."""
    try:
        json_payload = json.dumps(payload or {}, cls=DateTimeEncoder)
    except Exception as exc:
        json_payload = json.dumps(
            {
                "eventType": "serialization-error",
                "message": str(exc),
            },
            cls=DateTimeEncoder,
        )

    lines = json_payload.splitlines() or [""]
    data_section = "\n".join(f"data: {line}" for line in lines)
    return f"event: {event_name}\n{data_section}\n\n".encode("utf-8")


def ListDicomModalities(output: orthanc.RestOutput, url: str, **request) -> None:
    """Return configured DICOM modalities from Orthanc."""
    if handle_cors_preflight(output, request):
        return

    method = request.get("method")
    if method != "GET":
        output.SendMethodNotAllowed("GET")
        return

    try:
        raw = orthanc.RestApiGet("/modalities")
        modality_ids = json.loads(raw.decode("utf-8"))
        items = []
        for key in modality_ids:
            details_raw = orthanc.RestApiGet(f"/modalities/{key}/configuration")
            details = json.loads(details_raw.decode("utf-8")) if details_raw else {}
            items.append(
                {
                    "id": key,
                    "aet": details.get("AET") or details.get("Aet") or key,
                    "title": details.get("Title") or key,
                    "host": details.get("Host"),
                    "port": details.get("Port"),
                    "manufacturer": details.get("Manufacturer"),
                    "description": details.get("Description"),
                }
            )
        response = {"modalities": items}
        send_json(output, response)
    except Exception as exc:
        print(f"Error listing DICOM modalities: {exc}")
        error_response = {"error": f"Failed to list DICOM modalities: {str(exc)}"}
        send_json(output, error_response, status=500)


def QueryDicomStore(output: orthanc.RestOutput, url: str, **request) -> None:
    """Run a remote DICOM query via Orthanc."""
    if handle_cors_preflight(output, request):
        return

    method = request.get("method")
    if method != "POST":
        output.SendMethodNotAllowed("POST")
        return

    body = request.get("body", b"{}")
    if isinstance(body, bytes):
        body = body.decode("utf-8") or "{}"

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        error = {"error": "Invalid JSON payload."}
        send_json(output, error, status=400)
        return

    modality = (payload.get("modality") or "").strip()
    level = payload.get("level", "Series") or "Series"
    query = payload.get("query") or {}
    limit = payload.get("limit", 50)

    if not modality:
        error = {"error": "modality is required."}
        send_json(output, error, status=400)
        return

    try:
        limit_value = int(limit)
    except Exception:
        limit_value = 50

    try:
        results = dicom_pull_manager.query_remote(modality, level, query, limit_value)
        response = {"results": results}
        send_json(output, response)
    except Exception as exc:
        print(f"Error querying modality {modality}: {exc}")
        error = {"error": f"Query failed: {str(exc)}"}
        send_json(output, error, status=500)


def HandleDicomPullBatches(output: orthanc.RestOutput, url: str, **request) -> None:
    """Manage batch schedules for DICOM pulls."""
    if handle_cors_preflight(output, request):
        return

    method = request.get("method")
    if method == "GET":
        params = request.get("get", {})
        limit = params.get("limit", 50)
        try:
            limit_value = int(limit)
        except Exception:
            limit_value = 50
        try:
            batches = dicom_pull_manager.list_batches(limit_value)
            response = {"batches": batches}
            send_json(output, response)
        except Exception as exc:
            print(f"Error listing pull batches: {exc}")
            error = {"error": f"Failed to list batches: {str(exc)}"}
            send_json(output, error, status=500)
        return

    if method == "POST":
        body = request.get("body", b"{}")
        if isinstance(body, bytes):
            body = body.decode("utf-8") or "{}"

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            error = {"error": "Invalid JSON payload."}
            send_json(output, error, status=400)
            return

        try:
            batch = dicom_pull_manager.create_batch(payload)
            send_json(output, batch)
        except ValueError as exc:
            error = {"error": str(exc)}
            send_json(output, error, status=400)
        except Exception as exc:
            print(f"Error creating pull batch: {exc}")
            error = {"error": f"Failed to create batch: {str(exc)}"}
            send_json(output, error, status=500)
        return

    output.SendMethodNotAllowed("GET, POST")


def HandleDicomPullBatchDetail(output: orthanc.RestOutput, url: str, **request) -> None:
    """Get or mutate a specific batch."""
    if handle_cors_preflight(output, request):
        return

    method = request.get("method")
    groups = request.get("groups", [])
    tail = groups[0] if groups else ""
    if not tail:
        error = {"error": "Batch not found."}
        send_json(output, error, status=404)
        return

    parts = tail.split("/")
    try:
        batch_id = int(parts[0])
    except ValueError:
        error = {"error": "Invalid batch identifier."}
        send_json(output, error, status=400)
        return

    action = parts[1] if len(parts) > 1 else None

    if action == "cancel":
        if method != "POST":
            output.SendMethodNotAllowed("POST")
            return
        body = request.get("body", b"{}")
        if isinstance(body, bytes):
            body = body.decode("utf-8") or "{}"
        reason = None
        try:
            data = json.loads(body)
            reason = data.get("reason")
        except json.JSONDecodeError:
            reason = None

        success = dicom_pull_manager.cancel_batch(batch_id, reason)
        if not success:
            error = {"error": "Batch is not cancellable or not found."}
            send_json(output, error, status=409)
            return

        response = {"status": "cancelled", "batch_id": batch_id}
        send_json(output, response)
        return

    if method != "GET":
        output.SendMethodNotAllowed("GET")
        return

    batch = dicom_pull_manager.get_batch(batch_id)
    if not batch:
        error = {"error": "Batch not found."}
        send_json(output, error, status=404)
        return

    send_json(output, batch)


def HandleDicomPullRecover(output: orthanc.RestOutput, url: str, **request) -> None:
    """Attempt to recover DICOM data immediately by scheduling an urgent pull."""
    if handle_cors_preflight(output, request):
        return

    method = request.get("method")
    if method != "POST":
        output.SendMethodNotAllowed("POST")
        return

    body = request.get("body", b"{}")
    if isinstance(body, bytes):
        body = body.decode("utf-8") or "{}"

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        error = {"error": "Invalid JSON payload."}
        send_json(output, error, status=400)
        return

    series_uid = (
        payload.get("seriesInstanceUID") or payload.get("series_instance_uid") or ""
    ).strip()
    if not series_uid:
        error = {"error": "seriesInstanceUID is required."}
        send_json(output, error, status=400)
        return

    study_uid = (
        payload.get("studyInstanceUID") or payload.get("study_instance_uid") or ""
    ).strip()
    preferred_modalities = payload.get("modalities") or payload.get("modality")

    if isinstance(preferred_modalities, str):
        modality_ids = [preferred_modalities]
    elif isinstance(preferred_modalities, list):
        modality_ids = [str(mod) for mod in preferred_modalities if mod]
    else:
        modality_ids = None

    if not modality_ids:
        try:
            raw = orthanc.RestApiGet("/modalities")
            parsed = json.loads(raw.decode("utf-8"))
            if isinstance(parsed, list):
                modality_ids = parsed
            elif isinstance(parsed, dict):
                modality_ids = list(parsed.keys())
            else:
                modality_ids = []
        except Exception as exc:
            print(f"Error retrieving modalities for recovery: {exc}")
            modality_ids = []

    if not modality_ids:
        error = {"error": "No modalities available for recovery."}
        send_json(output, error, status=404)
        return

    query = {"SeriesInstanceUID": series_uid}
    if study_uid:
        query["StudyInstanceUID"] = study_uid

    last_error = None
    for modality in modality_ids:
        try:
            matches = dicom_pull_manager.query_remote(
                modality, "Series", query, limit=5
            )
        except Exception as exc:
            print(f"Error querying modality {modality} for {series_uid}: {exc}")
            last_error = exc
            continue

        if not matches:
            continue

        match = None
        for candidate in matches:
            candidate_uid = candidate.get("seriesInstanceUID") or candidate.get(
                "series_instance_uid"
            )
            if not candidate_uid or candidate_uid == series_uid:
                match = candidate
                break
        if match is None:
            match = matches[0]

        item_payload = dict(match)
        if not item_payload.get("seriesInstanceUID"):
            item_payload["seriesInstanceUID"] = series_uid
        if study_uid and not item_payload.get("studyInstanceUID"):
            item_payload["studyInstanceUID"] = study_uid
        if payload.get("patientId") and not item_payload.get("patientId"):
            item_payload["patientId"] = payload["patientId"]

        estimated_seconds = item_payload.get("estimatedSeconds")
        try:
            estimated_seconds = int(estimated_seconds)
        except Exception:
            estimated_seconds = dicom_pull_manager.min_item_seconds
        if estimated_seconds <= 0:
            estimated_seconds = dicom_pull_manager.min_item_seconds
        item_payload["estimatedSeconds"] = int(estimated_seconds)

        start_time = datetime.now(timezone.utc)
        estimated_total = (
            int(estimated_seconds) + dicom_pull_manager.batch_overhead_seconds
        )
        window_seconds = max(estimated_total + 120, int(estimated_seconds) + 300)
        end_time = start_time + timedelta(seconds=window_seconds)

        batch_payload = {
            "modality": modality,
            "displayName": payload.get("displayName") or f"Recovery {series_uid}",
            "notes": payload.get("notes"),
            "requestedBy": payload.get("requestedBy"),
            "startTime": start_time.isoformat(),
            "endTime": end_time.isoformat(),
            "timezone": payload.get("timezone") or "UTC",
            "items": [item_payload],
        }

        try:
            batch = dicom_pull_manager.create_batch(batch_payload)
            response = {
                "status": "scheduled",
                "modality": modality,
                "batch": batch,
            }
            send_json(output, response)
            return
        except ValueError as exc:
            print(f"Invalid recovery batch for modality {modality}: {exc}")
            last_error = exc
        except Exception as exc:
            print(f"Failed to create recovery batch for modality {modality}: {exc}")
            last_error = exc

    if last_error is not None:
        error = {"error": f"Failed to schedule recovery: {str(last_error)}"}
        send_json(output, error, status=500)
    else:
        error = {"error": "Series not found on remote modalities."}
        send_json(output, error, status=404)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        elif isinstance(o, date):
            return o.isoformat()
        elif isinstance(o, time):
            return o.isoformat()
        return super().default(o)


def get_database_connection():
    """Get database connection using Orthanc's PostgreSQL settings"""

    try:
        conn = psycopg2.connect(
            host="postgres",
            port=5432,
            database="orthanc",
            user="postgres",
            password="pgpassword",
            connect_timeout=5,  # 5 second timeout
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"Failed to connect to database: {e}")


# def ServeStaticFile(output: orthanc.RestOutput, url: str, **request) -> None:
#     """Serve static files"""

#     if handle_cors_preflight(output, request):
#         return

#     method = request.get('method')
#     groups = request.get('groups')

#     if method != 'GET':
#         output.SendMethodNotAllowed('GET')
#         return

#     # Extract filename from URL
#     filename = groups[0] if groups else None
#     filepath = os.path.join("/src", "static", filename) if filename else None

#     print(f"Requested static file: {filename}")

#     if not filepath or not os.path.exists(filepath):
#         output.SendHttpStatusCode(404)
#         return

#     _, ext = os.path.splitext(filepath)


#     try:
#         with open(filepath, 'rb') as f:
#             content = f.read()
#         output.SetHttpHeader('Cache-Control', 'public, max-age=3600')  # Cache for 1 hour
#         if ext == '.js':
#             output.SetHttpHeader('Content-Type', 'application/javascript; charset=utf-8')
#         elif ext == '.css':
#             output.SetHttpHeader('Content-Type', 'text/css; charset=utf-8')
#         output.AnswerBuffer(content, 'application/octet-stream')
#     except FileNotFoundError:
#         print(f"Static file not found: {filepath}")
#         output.SendHttpStatusCode(404)
def DeleteDicomSeries(output: orthanc.RestOutput, url: str, **request) -> None:
    """Delete a DICOM series from Orthanc storage by its Orthanc UUID.

    DELETE /cho-dicom/<series_uuid>

    This is a separate action from deleting analysis results in the database;
    both can co-exist independently.
    """
    if handle_cors_preflight(output, request):
        return

    method = request.get("method")
    groups = request.get("groups", [])
    series_uuid = groups[0] if groups else None

    if method != "DELETE":
        output.SendMethodNotAllowed("DELETE")
        return

    if not series_uuid:
        error = {"error": "series_uuid is required"}
        send_json(output, error, status=400)
        return

    try:
        orthanc.RestApiDelete(f"/series/{series_uuid}")
        response = {
            "message": f"Successfully deleted DICOM series {series_uuid} from Orthanc"
        }
        send_json(output, response)
        print(f"Deleted DICOM series from Orthanc: {series_uuid}")
    except Exception as e:
        print(f"Error deleting DICOM series {series_uuid}: {e}")
        error_response = {"error": f"Failed to delete DICOM series: {str(e)}"}
        send_json(output, error_response, status=500)


def HandleCHOResult(output: orthanc.RestOutput, url: str, **request) -> None:
    """Handle CHO results for a specific series - both GET and DELETE"""

    if handle_cors_preflight(output, request):
        return

    method = request.get("method")
    groups = request.get("groups")

    series_instance_uid = groups[0] if groups else None

    if not series_instance_uid:
        error = {"error": "Series instance UID is required."}
        send_json(output, error, status=400)
        return

    if method == "GET":
        try:
            result, description = cho_storage.get_result(series_instance_uid)

            if result and description is not None:
                # Convert to dictionary
                columns = [desc[0] for desc in description]
                result_dict = {col: row for col, row in zip(columns, result)}
                # Convert datetime to string for JSON serialization
                for key, value in result_dict.items():
                    if isinstance(value, datetime):
                        result_dict[key] = value.isoformat()

                send_json(output, result_dict)
            else:
                error = {"error": "CHO results not found."}
                send_json(output, error, status=404)

        except Exception as e:
            error_response = {"error": f"Failed to retrieve CHO results: {str(e)}"}
            print(f"Error retrieving CHO results: {e}")
            send_json(output, error_response, status=500)

    elif method == "DELETE":
        try:
            deleted = cho_storage.delete_results(series_instance_uid)
            if deleted:
                # Send success response
                response = {
                    "message": f"Successfully deleted CHO results for series {series_instance_uid}"
                }
                send_json(output, response)
            else:
                error = {"error": "CHO results not found."}
                send_json(output, error, status=404)
        except Exception as e:
            print(f"Error deleting CHO result: {e}")
            error_response = {"error": f"Failed to delete CHO result: {str(e)}"}
            send_json(output, error_response, status=500)

    else:
        output.SendMethodNotAllowed("GET, DELETE")


def StartCHOAnalysis(output: orthanc.RestOutput, url: str, **request) -> None:
    """New endpoint specifically for modal requests with custom parameters"""
    if handle_cors_preflight(output, request):
        return

    method = request.get("method")
    get = request.get("get", {})

    if method not in ["POST", "GET"]:
        output.SendMethodNotAllowed("GET, POST")
        return

    try:
        # Get request body with parameters
        body = request.get("body", b"{}")
        if isinstance(body, bytes):
            body = body.decode("utf-8")

        params = json.loads(body)
        # update body with get parameters
        params.update(get)
        series_uuid = params.get("series_uuid")

        if not series_uuid:
            error = {"error": "series_uuid is required."}
            send_json(output, error, status=400)
            return

        # Extract parameters
        test_type = params.get("testType", "global")
        full_test = test_type == "full"

        # Custom parameters that could be used to modify the analysis
        custom_params = {
            "resamples": params.get("resamples", 500),
            "internalNoise": params.get("internalNoise", 2.25),
            "resamplingMethod": params.get("resamplingMethod", "Bootstrap"),
            "roiSize": params.get("roiSize", 6),
            "thresholdLow": params.get("thresholdLow", 0),
            "thresholdHigh": params.get("thresholdHigh", 150),
            "windowLength": params.get("windowLength", 15.0),
            "stepSize": params.get("stepSize", 5.0),
            "channelType": params.get("channelType", "Gabor"),
            "lesionSet": params.get("lesionSet", "standard"),
            "saveResults": parse_bool(params.get("saveResults"), True),
            "deleteAfterCompletion": params.get("deleteAfterCompletion", False),
            "report_progress": params.get("report_progress", True),
        }

        print(f"Starting CHO analysis with custom parameters: {custom_params}")

        # Start calculation with custom parameters
        calculation_id = run_cho_calculation_with_progress(
            series_uuid, full_test, custom_params  # Pass custom parameters
        )

        response = {
            "status": "started",
            "calculation_id": calculation_id,
            "message": "Analysis started with custom parameters",
            "parameters": custom_params,
        }

        send_json(output, response)

    except Exception as e:
        print(f"Error starting CHO analysis from modal: {str(e)}")
        error_response = {"status": "error", "message": str(e)}
        send_json(output, error_response, status=500)


def GetAllCHOResults(output: orthanc.RestOutput, url: str, **request) -> None:
    """Get all CHO results with enhanced filtering options and pagination"""

    if handle_cors_preflight(output, request):
        return

    method = request.get("method")
    get = request.get("get", {})

    if method != "GET":
        output.SendMethodNotAllowed("GET")
        return

    try:
        response = cho_storage.get_results(get)
        send_json(output, response)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error retrieving CHO results: {e}")
        error_response = {"error": f"Failed to retrieve CHO results: {str(e)}"}
        send_json(output, error_response, status=500)


def GetFilterOptions(output: orthanc.RestOutput, url: str, **request) -> None:
    """Get available filter options for dropdowns"""

    if handle_cors_preflight(output, request):
        return

    method = request.get("method")

    if method != "GET":
        output.SendMethodNotAllowed("GET")
        return

    try:
        # Get unique values for filter dropdowns - updated for new schema
        filter_options = cho_storage.get_filter_options()
        send_json(output, filter_options)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error retrieving filter options: {e}")
        error_response = {"error": f"Failed to retrieve filter options: {str(e)}"}
        send_json(output, error_response, status=500)


def ExportCHOResultsCSV(output: orthanc.RestOutput, url: str, **request) -> None:
    """Export CHO results to CSV format"""

    if handle_cors_preflight(output, request):
        return

    method = request.get("method")

    if method != "GET":
        output.SendMethodNotAllowed("GET")
        return

    try:
        csv_content = cho_storage.export_results()
        output.SetHttpHeader("Content-Type", "text/csv")
        output.SetHttpHeader(
            "Content-Disposition", 'attachment; filename="cho_results.csv"'
        )
        output.AnswerBuffer(csv_content.encode("utf-8"), "text/csv")

    except Exception as e:
        print(f"Error exporting CSV: {e}")
        error = {"error": "Failed to export CSV."}
        send_json(output, error, status=500)


# def ServeCHODashboard(output: orthanc.RestOutput, url: str, **request) -> None:
#     """Serve the CHO dashboard HTML page"""

#     if handle_cors_preflight(output, request):
#         return

#     method = request.get('method')

#     if method != 'GET':
#         output.SendMethodNotAllowed('GET')
#         return

#     # Read the HTML file with UTF-8 encoding
#     try:
#         with open('/src/templates/dashboard.html', 'r', encoding='utf-8') as f:
#             html_content = f.read()
#         output.AnswerBuffer(html_content.encode('utf-8'), 'text/html')
#     except Exception as e:
#         print(f"Error serving dashboard: {e}")
#         error = {"error": "Failed to serve dashboard."}
#         send_json(output, error, status=500)


def ServeExportResults(output: orthanc.RestOutput, url: str, **request) -> None:
    """Serve the export results endpoint

    Args:
        output (orthanc.RestOutput): The output object for sending responses
        url (str): The request URL
    """
    print(request)
    if handle_cors_preflight(output, request):
        return

    method = request.get("method")

    if method != "POST":
        output.SendMethodNotAllowed("POST")
        return

    try:
        # Get request body with parameters
        body = request.get("body", b"{}")
        if isinstance(body, bytes):
            body = body.decode("utf-8")

        params = json.loads(body)
        series_ids = params.get("series_ids")
        if not series_ids:
            error = {"error": "Series IDs are required."}
            send_json(output, error, status=400)
            return
        elif len(series_ids) == 1:
            # Handle single series export
            series_id = series_ids[0]
            r = orthanc.RestApiGetAfterPlugins(f"/cho-results/{series_id}")
            r = json.loads(r)
            del r["id"], r["series_id_fk"]

            # Create a new Excel workbook and add a worksheet
            excel = xlwt.Workbook()
            sheet = excel.add_sheet("Results")

            # Write the header row
            headers = r.keys()
            for col_num, header in enumerate(headers):
                sheet.write(0, col_num, header)

            my_dict_2 = {}

            for key, value in r.items():
                if isinstance(value, list):
                    my_dict_2[key] = len(value)
                else:
                    my_dict_2[key] = 0

            longest_list_count = max(my_dict_2.values())

            # Write the data rows
            for row_num in range(1, longest_list_count + 1):
                for col_num, header in enumerate(headers):
                    value = ""
                    if row_num == 1:
                        if isinstance(r[header], list):
                            value = r[header][0]
                        else:
                            value = r[header] if r[header] is not None else "N/A"
                    else:
                        if isinstance(r[header], list):
                            value = (
                                r[header][row_num - 1]
                                if row_num - 1 < len(r[header])
                                else ""
                            )

                    sheet.write(row_num, col_num, value)

            # Save the workbook to a BytesIO buffer
            b = io.BytesIO()
            excel.save(b)
            output.AnswerBuffer(b.getvalue(), "application/vnd.ms-excel")
        # else:
        #     # Handle multiple series export
        #     ServeExportMultipleSeries(output, series_ids)
    except Exception as e:
        print(f"Error decoding request body: {e}")
        error = {"error": "Invalid request body."}
        send_json(output, error, status=400)


def ServeSaveResults(output: orthanc.RestOutput, url: str, **request) -> None:
    """Serve the save results endpoint"""

    if handle_cors_preflight(output, request):
        return

    method = request.get("method")

    if method != "POST":
        output.SendMethodNotAllowed("POST")
        return

    try:
        # Get request body with parameters
        body = request.get("body", b"{}")
        if isinstance(body, bytes):
            body = body.decode("utf-8")

        params = json.loads(body)
        print(params)
        patient = params.get("patient")
        study = params.get("study")
        scanner = params.get("scanner")
        series = params.get("series")
        ct = params.get("ct")
        converted_results = params.get("converted_results")

        success = cho_storage.save_results(
            patient, study, scanner, series, ct, converted_results
        )

        progress_tracker.cleanup_calculation(series["uuid"])
        progress_tracker.cleanup_history(series["uuid"])
        if success:
            print(
                f"Full analysis results saved to database for series {series['series_instance_uid']}"
            )
        else:
            print(
                f"Failed to save full analysis results to database for series {series['series_instance_uid']}"
            )
            raise Exception("Database save failed")

    except Exception as e:
        print(f"Error saving to database: {str(e)}")
        error = {"error": "Failed to save results to database."}
        send_json(output, error, status=500)
        return

    success_response = {"message": "Results saved successfully"}
    send_json(output, success_response)


def ServeMinIOImage(output: orthanc.RestOutput, url: str, **request) -> None:
    """Serve images from MinIO storage"""

    if handle_cors_preflight(output, request):
        return

    method = request.get("method")
    groups = request.get("groups")

    if method != "GET":
        output.SendMethodNotAllowed("GET")
        return

    if not groups or len(groups) < 1:
        error = {"error": "Invalid request."}
        send_json(output, error, status=400)
        return

    path_tail = groups[0]
    parts = path_tail.split("/", 1)
    series_instance_uid = parts[0]
    object_suffix = parts[1] if len(parts) > 1 else None
    object_name = object_suffix or f"{series_instance_uid}_coronal_view.png"

    response = None
    try:
        try:
            response = cho_storage.minio_client.get_object(
                bucket_name=cho_storage.bucket_name, object_name=object_name
            )
            image_data = response.read()
        except Exception as e:
            print(f"Error retrieving image from MinIO: {str(e)}")
            error = {"error": "Image not found."}
            send_json(output, error, status=404)
            return
        finally:
            if response is not None:
                try:
                    response.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    response.release_conn()  # type: ignore[attr-defined]
                except Exception:
                    pass

        output.SetHttpHeader(
            "Cache-Control", "public, max-age=3600"
        )  # Cache for 1 hour
        output.AnswerBuffer(image_data, "image/png")
    except Exception as e:
        print(f"Error accessing MinIO: {str(e)}")
        error = {"error": "Failed to retrieve image from MinIO."}
        send_json(output, error, status=500)


def GetImageMetadata(output: orthanc.RestOutput, url: str, **request) -> None:
    """Get metadata about available images for a series"""

    if handle_cors_preflight(output, request):
        return

    method = request.get("method")
    groups = request.get("groups")

    if method != "GET":
        output.SendMethodNotAllowed("GET")
        return

    if not groups or len(groups) < 1:
        error = {"error": "Invalid request."}
        send_json(output, error, status=400)
        return

    series_instance_uid = groups[0]

    try:
        # List objects with the series prefix
        objects = cho_storage.minio_client.list_objects(
            bucket_name=cho_storage.bucket_name, prefix=f"{series_instance_uid}_"
        )

        images = []
        for obj in objects:
            # Get object stats for metadata
            object_name = obj.object_name if obj.object_name else ""
            stat = cho_storage.minio_client.stat_object(
                bucket_name=cho_storage.bucket_name, object_name=object_name
            )

            images.append(
                {
                    "name": obj.object_name,
                    "size": stat.size,
                    "last_modified": (
                        stat.last_modified.isoformat() if stat.last_modified else None
                    ),
                    "content_type": stat.content_type,
                    "url": f"/minio-images/{series_instance_uid}/{object_name.split('_', 1)[1]}",
                }
            )

        response = {
            "series_instance_uid": series_instance_uid,
            "images": images,
            "count": len(images),
        }

        send_json(output, response)

    except Exception as e:
        print(f"Error listing images: {str(e)}")
        error_response = {"error": f"Failed to list images: {str(e)}"}
        send_json(output, error_response, status=500)


def ServeResultsStatistics(output: orthanc.RestOutput, url: str, **request) -> None:
    """Serve the results statistics page"""

    if handle_cors_preflight(output, request):
        return

    method = request.get("method")

    if method != "GET":
        output.SendMethodNotAllowed("GET")
        return

    # Get statistics data
    try:

        result = cho_storage.get_results_statistics()

        if result:
            (
                total_results_count,
                global_noise_count,
                detectability_count,
                error_count,
            ) = result
            statistics = {
                "total_results_count": total_results_count,
                "global_noise_count": global_noise_count,
                "detectability_count": detectability_count,
                "error_count": error_count,
            }
            send_json(output, statistics)
        else:
            error_response = {"error": "No statistics available"}
            send_json(output, error_response, status=404)

    except Exception as e:
        print(f"Error retrieving results statistics: {str(e)}")
        error_response = {"error": f"Failed to retrieve results statistics: {str(e)}"}
        send_json(output, error_response, status=500)


def StreamChoProgress(output: orthanc.RestOutput, url: str, **request) -> None:
    """Stream CHO calculation progress updates using Server-Sent Events."""

    if handle_cors_preflight(output, request):
        return

    method = request.get("method")
    if method != "GET":
        output.SendMethodNotAllowed("GET")
        return

    set_cors_headers(output)
    output.SetHttpHeader("Cache-Control", "no-cache")
    output.SetHttpHeader("Connection", "keep-alive")
    output.SetHttpHeader("X-Accel-Buffering", "no")

    output.StartStreamAnswer("text/event-stream")
    output.SendStreamChunk(f"retry: {SSE_RETRY_MILLISECONDS}\n\n".encode("utf-8"))

    client_id, channel = sse_manager.add_client()
    heartbeat_deadline = pytime.time() + SSE_HEARTBEAT_SECONDS

    try:
        snapshot_payload = {
            "eventType": "snapshot",
            "active": progress_tracker.get_all_active_calculations(),
            "history": progress_tracker.get_calculation_history(),
            "serverTime": datetime.now(timezone.utc).isoformat(),
        }
        output.SendStreamChunk(_encode_sse_event("snapshot", snapshot_payload))

        while True:
            now = pytime.time()
            timeout = max(0.1, heartbeat_deadline - now)
            try:
                message = channel.get(timeout=timeout)
            except queue.Empty:
                output.SendStreamChunk(b":heartbeat\n\n")
                heartbeat_deadline = pytime.time() + SSE_HEARTBEAT_SECONDS
                continue

            if not isinstance(message, dict):
                continue

            event_name = message.get("event", "cho-calculation")
            payload = message.get("data") or {}
            if not isinstance(payload, dict):
                payload = {"value": payload}

            timestamp = message.get("timestamp")
            if timestamp is not None and "timestamp" not in payload:
                payload["timestamp"] = datetime.fromtimestamp(
                    timestamp, timezone.utc
                ).isoformat()

            output.SendStreamChunk(_encode_sse_event(event_name, payload))
            heartbeat_deadline = pytime.time() + SSE_HEARTBEAT_SECONDS
    except BrokenPipeError:
        pass
    except Exception as exc:
        msg = str(exc)
        # Orthanc closes with “Error in the network protocol” when the client goes away
        if "Error in the network protocol" in msg or "broken pipe" in msg.lower():
            pass
        else:
            print(f"StreamChoProgress terminated for client {client_id}: {exc}")
    finally:
        sse_manager.remove_client(client_id)


# def CreateLesion(output: orthanc.RestOutput, url: str, **request) -> None:
#     """Create lesion PNG via Celery; revoke superseded jobs to support abort."""
#     if handle_cors_preflight(output, request):
#         return
#     if request.get('method') != 'GET':
#         output.SendMethodNotAllowed('GET')
#         return


#     q = request.get('get', {}) or {}
#     if 'set' not in q or 'index' not in q:
#         output.SendHttpStatusCode(400)
#         output.AnswerBuffer(json.dumps({"error": "Missing set or index parameter"}, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
#         return

#     lesion_set = q['set']
#     try:
#         lesion_index = int(q['index'])
#     except Exception:
#         output.SendHttpStatusCode(400)
#         output.AnswerBuffer(json.dumps({"error": "index must be an integer"}, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
#         return

#     mtf50 = q.get('mtf50')
#     mtf10 = q.get('mtf10')
#     try:
#         mtf50 = float(mtf50) if mtf50 is not None else None
#         mtf10 = float(mtf10) if mtf10 is not None else None
#     except Exception:
#         output.SendHttpStatusCode(400)
#         output.AnswerBuffer(json.dumps({"error": "mtf50/mtf10 must be numeric"}, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
#         return

#     # Group equivalent work; include resolution params
#     token = q.get('_') or f"{lesion_set}:{lesion_index}:{mtf50}:{mtf10}"
#     req_id = int(pytime.time() * 1e9)

#     def is_stale() -> bool:
#         with ACTIVE_LOCK:
#             return ACTIVE_JOBS.get(token, {}).get('req_id') != req_id

#     def abort_job(job: dict) -> None:
#         if not job or not job.get('task_id'):
#             return
#         try:
#             AbortableAsyncResult(job['task_id'], app=celery_app).abort()
#         except Exception as exc:
#             print(f"Failed to abort lesion task {job.get('task_id')}: {exc}")

#     # Cancel any older job for this token, then register ours
#     with ACTIVE_LOCK:
#         prev = ACTIVE_JOBS.get(token)
#         abort_job(prev)
#         async_handle = generate_lesion_png.apply_async(
#             args=(lesion_set, lesion_index, mtf50, mtf10),
#             expires=30,  # bail quickly if not claimed
#         )
#         ACTIVE_JOBS[token] = {'req_id': req_id, 'task_id': async_handle.id}

#     async_result = AbortableAsyncResult(ACTIVE_JOBS[token]['task_id'], app=celery_app)

#     payload = None
#     try:
#         # Poll the task; bail if a newer request has superseded us
#         while True:
#             if is_stale():
#                 try:
#                     async_result.abort()
#                 except Exception:
#                     pass
#                 output.SendHttpStatusCode(499)
#                 return

#             try:
#                 payload = async_result.get(timeout=0.1)
#                 break
#             except CeleryTimeoutError:
#                 continue
#             except TaskRevokedError:
#                 output.SendHttpStatusCode(499)
#                 return

#         if not isinstance(payload, dict) or 'png_base64' not in payload:
#             raise ValueError("Invalid lesion task response")

#         if is_stale():
#             try:
#                 async_result.abort()
#             except Exception:
#                 pass
#             output.SendHttpStatusCode(499)
#             return

#         png_bytes = base64.b64decode(payload['png_base64'])
#         output.AnswerBuffer(png_bytes, 'image/png')
#         try:
#             output.SetHttpHeader('Content-Disposition', 'inline; filename="lesion.png"')
#             output.SetHttpHeader('Cache-Control', 'no-store')
#         except Exception:
#             pass

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         output.SendHttpStatusCode(500)
#         output.AnswerBuffer(json.dumps({"status": "error", "message": str(e)}, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
#     finally:
#         try:
#             async_result.forget()
#         except Exception:
#             pass
#         # Clean our slot only if we're still the latest for this token
#         with ACTIVE_LOCK:
#             job = ACTIVE_JOBS.get(token)
#             if job and job.get('req_id') == req_id:
#                 ACTIVE_JOBS.pop(token, None)


def CreateLesion(output: orthanc.RestOutput, url: str, **request) -> None:
    """Create a new lesion entry in the database"""

    if handle_cors_preflight(output, request):
        return

    method = request.get("method")
    if method != "GET":
        output.SendMethodNotAllowed("GET")
        return

    q = request.get("get", {}) or {}
    if "set" not in q or "index" not in q:
        error_response = {"error": "Missing set or index parameter"}
        send_json(output, error_response, status=400)
        return

    try:
        lesion_set = q["set"]
        lesion_index = int(q["index"])
        mtf50 = q.get("mtf50", None)
        mtf10 = q.get("mtf10", None)
        recon_diameter_mm = q.get("recon_diameter_mm", None)
        rows = q.get("rows", None)
        cache_bust = q.get("_")

        if mtf50 is not None:
            mtf50 = float(mtf50)
        if mtf10 is not None:
            mtf10 = float(mtf10)
        if recon_diameter_mm is not None:
            recon_diameter_mm = float(recon_diameter_mm)
        if rows is not None:
            rows = float(rows)

        print(
            f"Creating lesion from set '{lesion_set}', index {lesion_index}, mtf50={mtf50}, mtf10={mtf10}, recon_diameter_mm={recon_diameter_mm}, rows={rows}"
        )

        lesion = create_lesion_model(
            lesion_set, lesion_index, mtf50, mtf10, recon_diameter_mm, rows
        )
        print(f"lesion.shape: {lesion.shape}")
        print(f"lesion.max: {lesion.max()}")
        print(f"lesion.min: {lesion.min()}")
        print(f"lesion.mean: {lesion.mean()}")
        print(f"lesion[0,0]: {lesion[0,0]}")

        window_width = 200
        window_level = 0

        old_min = window_level - window_width // 2
        old_max = window_level + window_width // 2
        new_min = 0
        new_max = 255
        normalized = (lesion - old_min) / (old_max - old_min) * (
            new_max - new_min
        ) + new_min
        background = normalized[0, 0]
        new_lesion = normalized - background + 127
        # print(f"Old min: {old_min}, New min: {np.min(new_lesion)}")
        # Normalize to uint8 PNG-safe image
        a = np.asarray(new_lesion)
        if a.dtype != np.uint8:
            # robust clamp -> uint8
            a = np.clip(a, 0, 255).astype(np.uint8)

        # Choose mode based on shape
        if a.ndim == 2:
            img = Image.fromarray(a, mode="L")
        elif a.ndim == 3 and a.shape[2] in (3, 4):
            img = Image.fromarray(a)  # RGB or RGBA
        else:
            # If it's a volume or unexpected shape, take the first slice
            if a.ndim >= 3:
                a2 = a[..., 0] if a.shape[-1] not in (3, 4) else a
                if a2.ndim == 2:
                    img = Image.fromarray(a2.astype(np.uint8), mode="L")
                else:
                    # last fallback: squeeze to 2D
                    img = Image.fromarray(np.squeeze(a2).astype(np.uint8), mode="L")
            else:
                img = Image.fromarray(a.astype(np.uint8))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        # Return raw PNG bytes (NOT base64)
        output.SetHttpHeader("Content-Disposition", 'inline; filename="lesion.png"')
        output.SetHttpHeader("Cache-Control", "no-store")
        output.AnswerBuffer(png_bytes, "image/png")

    except Exception as e:
        import traceback

        traceback.print_exc()
        error_response = {"status": "error", "message": str(e)}
        send_json(output, error_response, status=500)


def GetCalculationStatus(output: orthanc.RestOutput, url: str, **request) -> None:
    if handle_cors_preflight(output, request):
        return
    if request.get("method") != "GET":
        output.SendMethodNotAllowed("GET")
        return

    get = request.get("get", {})
    series_id = get.get("series_id")
    action = get.get("action")

    if not series_id:
        send_json(output, {"error": "Missing series_id parameter"}, 400)
        return
    if action not in ("check", "discard"):
        send_json(output, {"error": "Invalid action parameter"}, 400)
        return

    if action == "check":
        status = progress_tracker.get_calculation_status(series_id)
        if status:
            send_json(output, status)
        else:
            output.SendHttpStatusCode(204)
    else:
        progress_tracker.cleanup_calculation(series_id)
        progress_tracker.cleanup_history(series_id)
        output.SendHttpStatusCode(204)


def DelCalculationStatus(output: orthanc.RestOutput, url: str, **request) -> None:
    """Delete the status of a calculation"""

    if handle_cors_preflight(output, request):
        return

    method = request.get("method")
    get = request.get("get", {})

    if method != "GET":
        output.SendMethodNotAllowed("GET")
        return

    series_id = get.get("series_id")

    if not series_id:
        error_response = {"error": "Missing series_id parameter"}
        send_json(output, error_response, status=400)
        return

    status = progress_tracker.get_calculation_status(series_id)

    if status:
        send_json(output, status)
    else:
        error_response = {"error": f"No calculation found for series {series_id}"}
        send_json(output, error_response, status=404)


def GetActiveCalculations(output: orthanc.RestOutput, url: str, **request) -> None:
    """Get all active calculations"""
    if handle_cors_preflight(output, request):
        return
    method = request.get("method")
    if method != "GET":
        output.SendMethodNotAllowed("GET")
        return

    active = progress_tracker.get_all_active_calculations()
    send_json(output, active)


def test_database_connection():
    """Test if PostgreSQL database is connected and working"""
    try:
        print("=== Testing Database Connection ===")

        system_info = orthanc.RestApiGet("/system")
        system_data = json.loads(system_info)

        print(f"Orthanc Name: {system_data.get('Name', 'Unknown')}")
        print(f"Orthanc Version: {system_data.get('Version', 'Unknown')}")

        try:
            patients = orthanc.RestApiGet("/patients")
            patient_count = len(json.loads(patients))
            print(f"Database accessible - Found {patient_count} patients")
        except Exception as e:
            print(f"Database access test failed: {str(e)}")

        try:
            stats = orthanc.RestApiGet("/statistics")
            stats_data = json.loads(stats)
            print(f"Statistics: {stats_data}")
        except Exception as e:
            print(f"Statistics test failed: {str(e)}")

        try:
            db_id = orthanc.GetDatabaseServerIdentifier()
            print(f"Database Server ID: {db_id}")
        except Exception as e:
            print(f"Database server ID test failed: {str(e)}")

        print("=== Database Connection Test Complete ===")

    except Exception as e:
        print(f"Database connection test failed: {str(e)}")


def save_dicom_headers_only(series_uuid):
    """Save DICOM headers to database without running analysis"""
    from dicom_parser import DicomParser
    import pydicom

    try:
        print(f"Saving DICOM headers for series: {series_uuid}")

        # Get series instances
        instances_res = orthanc.RestApiGet(f"/series/{series_uuid}/instances")
        instances_json = json.loads(instances_res)
        print(f"loading: {series_uuid}")
        files = []
        for i in range(len(instances_json)):
            instance_id = instances_json[i]["ID"]
            f = orthanc.GetDicomForInstance(instance_id)
            files.append(pydicom.dcmread(io.BytesIO(f)))

        slices = [f for f in files if hasattr(f, "SliceLocation")]
        slices = sorted(slices, key=lambda s: s.SliceLocation)

        # Create parser and extract metadata
        dcm_parser = DicomParser(files)
        patient, study, scanner, series, ct = dcm_parser.extract_core()

        # Add series-specific metadata
        series["uuid"] = series_uuid
        series["image_count"] = len(files)

        # Save to database using results_storage
        success = cho_storage.save_dicom_headers_only(
            patient, study, scanner, series, ct
        )

        if success:
            print(f"DICOM headers saved successfully for series {series_uuid}")
        else:
            print(f"Failed to save DICOM headers for series {series_uuid}")

        return success

    except Exception as e:
        print(f"Error saving DICOM headers for series {series_uuid}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


# Register callbacks and endpoints
def OnChange(change_type, resource_type, resource_id):
    """Handle automatic analysis on stable series"""
    if change_type == orthanc.ChangeType.JOB_SUCCESS:
        print(f"Job succeeded for resource: {resource_id}")
        print(f"Resource type: {resource_type}")
    if (
        change_type
        in (orthanc.ChangeType.STABLE_SERIES, orthanc.ChangeType.JOB_SUCCESS)
        and resource_type == orthanc.ResourceType.SERIES
    ):
        print(f"Stable series: {resource_id}")
        series_uuid = resource_id
        try:
            # Check if this is a headers-only instance
            if TEST_TYPE == "None":
                # Save DICOM headers without running analysis
                save_dicom_headers_only(series_uuid)

                # Delete series if not saving
                if not SAVE_TYPE:
                    orthanc.RestApiDelete(f"/series/{series_uuid}")
                    print(f"Deleted series after saving headers: {series_uuid}")
                return

            test = "global" if TEST_TYPE == "Global Noise" else "full"

            body = {
                "series_uuid": series_uuid,
                "testType": test,
                "saveResults": True,
                "deleteAfterCompletion": not SAVE_TYPE,
                "report_progress": False,
            }
            orthanc.RestApiPostAfterPlugins(
                "/cho-analysis-modal", json.dumps(body).encode("utf-8")
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error in automatic CHO analysis for series {series_uuid}: {str(e)}")


orthanc.RegisterOnChangeCallback(OnChange)

orthanc.RegisterRestCallback("/cho-results/(.*)", HandleCHOResult)  # type: ignore
orthanc.RegisterRestCallback("/cho-results", GetAllCHOResults)  # type: ignore
orthanc.RegisterRestCallback("/cho-results-export", ExportCHOResultsCSV)  # type: ignore
orthanc.RegisterRestCallback("/cho-filter-options", GetFilterOptions)  # type: ignore
# orthanc.RegisterRestCallback('/cho-dashboard', ServeCHODashboard) # type: ignore
orthanc.RegisterRestCallback("/cho-calculation-status", GetCalculationStatus)  # type: ignore
orthanc.RegisterRestCallback("/cho-active-calculations", GetActiveCalculations)  # type: ignore
# orthanc.RegisterRestCallback('/static/(.*)', ServeStaticFile) # type: ignore
orthanc.RegisterRestCallback("/cho-analysis-modal", StartCHOAnalysis)  # type: ignore
orthanc.RegisterRestCallback("/minio-images/(.*)", ServeMinIOImage)  # type: ignore
orthanc.RegisterRestCallback("/image-metadata/(.*)", GetImageMetadata)  # type: ignore
orthanc.RegisterRestCallback("/results-statistics", ServeResultsStatistics)  # type: ignore
orthanc.RegisterRestCallback("/cho-save-results", ServeSaveResults)  # type: ignore
orthanc.RegisterRestCallback("/cho-export-results", ServeExportResults)  # type: ignore
orthanc.RegisterRestCallback("/dicom-modalities", ListDicomModalities)  # type: ignore
orthanc.RegisterRestCallback("/dicom-store/query", QueryDicomStore)  # type: ignore
orthanc.RegisterRestCallback("/dicom-pull/batches/(.*)", HandleDicomPullBatchDetail)  # type: ignore
orthanc.RegisterRestCallback("/dicom-pull/batches", HandleDicomPullBatches)  # type: ignore
orthanc.RegisterRestCallback("/dicom-pull/recover", HandleDicomPullRecover)  # type: ignore
orthanc.RegisterRestCallback("/cho-progress/stream", StreamChoProgress)  # type: ignore
orthanc.RegisterRestCallback("/create-lesion", CreateLesion)  # type: ignore
orthanc.RegisterRestCallback("/cho-dicom/(.*)", DeleteDicomSeries)  # type: ignore
