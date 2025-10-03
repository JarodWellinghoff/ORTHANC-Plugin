import io
import json
import os
import orthanc
import pydicom
import pandas as pd
from ChangeType import ChangeType
from ResourceType import ResourceType
import psycopg2  # type: ignore
from datetime import datetime, date, time
import xlwt
from results_storage import cho_storage
from progress_tracker import progress_tracker
from calculation_wrapper import run_cho_calculation_with_progress

TOKEN = orthanc.GenerateRestApiAuthorizationToken()
TEST_TYPE = os.getenv("INSTANCE_BEHAVIOR_TEST", "Global Noise")
SAVE_TYPE = (os.getenv("INSTANCE_BEHAVIOR_SAVE", "true") == "true")

print(f'=== CHO Analysis Plugin Initialized ===')
print(f' - Test Type: {TEST_TYPE}')
print(f' - Save Type: {SAVE_TYPE}')

print("=== CHO Analysis Plugin Initialized ===")
print("Available endpoints:")
print("  - /cho-dashboard - View CHO results dashboard")
print("  - /cho-calculation-status - Get the status of a calculation")
print("  - /cho-active-calculations - Get all active calculations")
print("  - /cho-results - Get all CHO results (with pagination)")
print("  - /cho-results/{series_id} - Get CHO results for a specific series")
print("  - /static/{filename} - Serve static files (JS, CSS)")
print("  - Auto Analysis: Automatically triggered on stable series (if enabled)")

CORS_ALLOWED_ORIGIN = os.getenv('CORS_ALLOWED_ORIGIN', '*')
CORS_ALLOWED_METHODS = os.getenv('CORS_ALLOWED_METHODS', 'GET, POST, PUT, DELETE, OPTIONS')
CORS_ALLOWED_HEADERS = os.getenv('CORS_ALLOWED_HEADERS', 'Content-Type, Authorization')
CORS_ALLOW_CREDENTIALS = os.getenv('CORS_ALLOW_CREDENTIALS', 'false').lower() == 'true'

def set_cors_headers(output: orthanc.RestOutput) -> None:
    output.SetHttpHeader('Access-Control-Allow-Origin', CORS_ALLOWED_ORIGIN)
    output.SetHttpHeader('Access-Control-Allow-Methods', CORS_ALLOWED_METHODS)
    output.SetHttpHeader('Access-Control-Allow-Headers', CORS_ALLOWED_HEADERS)
    if CORS_ALLOW_CREDENTIALS:
        output.SetHttpHeader('Access-Control-Allow-Credentials', 'true')

def handle_cors_preflight(output: orthanc.RestOutput, request: dict) -> bool:
    """Apply CORS headers and finish OPTIONS preflight when needed."""
    set_cors_headers(output)
    if request.get('method') == 'OPTIONS':
        output.SendHttpStatusCode(204)
        return True
    return False

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
            host='postgres',
            port=5432,
            database="orthanc",
            user="postgres",
            password="pgpassword",
            connect_timeout=5  # 5 second timeout
        )
        return conn
    except (psycopg2.OperationalError) as e:
        print(f"Failed to connect to database: {e}")

def ServeStaticFile(output: orthanc.RestOutput, url: str, **request) -> None:
    """Serve static files"""
    
    if handle_cors_preflight(output, request):
        return

    method = request.get('method')
    groups = request.get('groups')

    if method != 'GET':
        output.SendMethodNotAllowed('GET')
        return
    
    # Extract filename from URL
    filename = groups[0] if groups else None
    filepath = os.path.join("/src", "static", filename) if filename else None

    print(f"Requested static file: {filename}")
    
    if not filepath or not os.path.exists(filepath):
        output.SendHttpStatusCode(404)
        return
    
    _, ext = os.path.splitext(filepath)

    try:
        with open(filepath, 'rb') as f:
            content = f.read()
        output.SetHttpHeader('Cache-Control', 'public, max-age=3600')  # Cache for 1 hour
        if ext == '.js':
            output.SetHttpHeader('Content-Type', 'application/javascript; charset=utf-8')
        elif ext == '.css':
            output.SetHttpHeader('Content-Type', 'text/css; charset=utf-8')
        output.AnswerBuffer(content, 'application/octet-stream')
    except FileNotFoundError:
        print(f"Static file not found: {filepath}")
        output.SendHttpStatusCode(404)

def HandleCHOResult(output: orthanc.RestOutput, url: str, **request) -> None:
    """Handle CHO results for a specific series - both GET and DELETE"""
    
    if handle_cors_preflight(output, request):
        return

    method = request.get('method')
    groups = request.get('groups')

    series_instance_uid = groups[0] if groups else None

    if not series_instance_uid:
        output.SendHttpStatusCode(400)
        return

    if method == 'GET':
        try:
            result, description = cho_storage.get_result(series_instance_uid)

            if result and description is not None:
                # Convert to dictionary
                columns = [desc[0] for desc in description]
                result_dict = {
                    col: row for col, row in zip(columns, result)
                }
                # Convert datetime to string for JSON serialization
                for key, value in result_dict.items():
                    if isinstance(value, datetime):
                        result_dict[key] = value.isoformat()
                
                output.AnswerBuffer(json.dumps(result_dict, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
            else:
                output.SendHttpStatusCode(404)
            
        except Exception as e:
            print(f"Error retrieving CHO results: {e}")
            output.SendHttpStatusCode(500)
    
    elif method == 'DELETE':
        try:
            deleted = cho_storage.delete_results(series_instance_uid)
            if deleted:
                # Send success response
                response = {"message": f"Successfully deleted CHO results for series {series_instance_uid}"}
                output.AnswerBuffer(json.dumps(response, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
            else:
                output.SendHttpStatusCode(404)
        except Exception as e:
            print(f"Error deleting CHO result: {e}")
            error_response = {"error": f"Failed to delete CHO result: {str(e)}"}
            output.AnswerBuffer(json.dumps(error_response, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
            output.SendHttpStatusCode(500)
    
    else:
        output.SendMethodNotAllowed('GET, DELETE')

def StartCHOAnalysis(output: orthanc.RestOutput, url: str, **request) -> None:
    """New endpoint specifically for modal requests with custom parameters"""
    if handle_cors_preflight(output, request):
        return

    method = request.get('method')
    get = request.get('get', {})

    if method not in ['POST', 'GET'] :
        output.SendMethodNotAllowed('GET, POST')
        return
    
    try:
        # Get request body with parameters
        body = request.get('body', b'{}')
        if isinstance(body, bytes):
            body = body.decode('utf-8')
        
        params = json.loads(body)
        # update body with get parameters
        params.update(get)
        series_uuid = params.get('series_uuid')
        
        if not series_uuid:
            output.SendHttpStatusCode(400)
            return
        
        # Extract parameters
        test_type = params.get('testType', 'global')
        full_test = (test_type == 'full')
        
        # Custom parameters that could be used to modify the analysis
        custom_params = {
            'resamples': params.get('resamples', 500),
            'internalNoise': params.get('internalNoise', 2.25),
            'resamplingMethod': params.get('resamplingMethod', 'Bootstrap'),
            'roiSize': params.get('roiSize', 6),
            'thresholdLow': params.get('thresholdLow', 0),
            'thresholdHigh': params.get('thresholdHigh', 150),
            'windowLength': params.get('windowLength', 15.0),
            'stepSize': params.get('stepSize', 5.0),
            'channelType': params.get('channelType', 'Gabor'),
            'lesionSet': params.get('lesionSet', 'standard'),
            'saveResults': params.get('saveResults', False),
            'deleteAfterCompletion': params.get('deleteAfterCompletion', False),
            'report_progress': params.get('report_progress', True)
        }
        
        print(f"Starting CHO analysis with custom parameters: {custom_params}")
        
        # Start calculation with custom parameters
        calculation_id = run_cho_calculation_with_progress(
            series_uuid, 
            full_test,
            custom_params  # Pass custom parameters
        )
        
        response = {
            'status': 'started',
            'calculation_id': calculation_id,
            'message': 'Analysis started with custom parameters',
            'parameters': custom_params
        }
        
        output.AnswerBuffer(json.dumps(response, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
        
    except Exception as e:
        print(f"Error starting CHO analysis from modal: {str(e)}")
        error_response = {
            'status': 'error',
            'message': str(e)
        }
        output.AnswerBuffer(json.dumps(error_response, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
        output.SendHttpStatusCode(500)

def GetAllCHOResults(output: orthanc.RestOutput, url: str, **request) -> None:
    """Get all CHO results with enhanced filtering options and pagination"""
    
    if handle_cors_preflight(output, request):
        return

    method = request.get('method')
    get = request.get('get', {}) 

    if method != 'GET':
        output.SendMethodNotAllowed('GET')
        return
    
    try:
        response = cho_storage.get_results(get)
        output.AnswerBuffer(json.dumps(response, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error retrieving CHO results: {e}")
        output.SendHttpStatusCode(500)


def GetFilterOptions(output: orthanc.RestOutput, url: str, **request) -> None:
    """Get available filter options for dropdowns"""

    if handle_cors_preflight(output, request):
        return

    method = request.get('method')
    
    if method != 'GET':
        output.SendMethodNotAllowed('GET')
        return
    
    try:
        # Get unique values for filter dropdowns - updated for new schema
        filter_options = cho_storage.get_filter_options()
        output.AnswerBuffer(json.dumps(filter_options, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error retrieving filter options: {e}")
        output.SendHttpStatusCode(500)


def ExportCHOResultsCSV(output: orthanc.RestOutput, url: str, **request) -> None:
    """Export CHO results to CSV format"""

    if handle_cors_preflight(output, request):
        return

    method = request.get('method')

    if method != 'GET':
        output.SendMethodNotAllowed('GET')
        return
    
    try:
        csv_content = cho_storage.export_results()
        output.SetHttpHeader("Content-Type", "text/csv")
        output.SetHttpHeader("Content-Disposition", 'attachment; filename="cho_results.csv"')
        output.AnswerBuffer(csv_content.encode('utf-8'), 'text/csv')
        
    except Exception as e:
        print(f"Error exporting CSV: {e}")
        output.SendHttpStatusCode(500)

def ServeCHODashboard(output: orthanc.RestOutput, url: str, **request) -> None:
    """Serve the CHO dashboard HTML page"""

    if handle_cors_preflight(output, request):
        return

    method = request.get('method')

    if method != 'GET':
        output.SendMethodNotAllowed('GET')
        return
    
    # Read the HTML file with UTF-8 encoding
    try:
        with open('/src/templates/dashboard.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        output.AnswerBuffer(html_content.encode('utf-8'), 'text/html')
    except Exception as e:
        print(f"Error serving dashboard: {e}")
        output.SendHttpStatusCode(500)

def ServeExportResults(output: orthanc.RestOutput, url: str, **request) -> None:
    """Serve the export results endpoint

    Args:
        output (orthanc.RestOutput): The output object for sending responses
        url (str): The request URL
    """
    print(request)
    if handle_cors_preflight(output, request):
        return

    method = request.get('method')

    if method != 'POST':
        output.SendMethodNotAllowed('POST')
        return
    
    try:
        # Get request body with parameters
        body = request.get('body', b'{}')
        if isinstance(body, bytes):
            body = body.decode('utf-8')

        params = json.loads(body)
        series_ids = params.get('series_ids')
        if not series_ids:
            output.SendHttpStatusCode(400)
            return
        elif len(series_ids) == 1:
            # Handle single series export
            series_id = series_ids[0]
            r = orthanc.RestApiGetAfterPlugins(f'/cho-results/{series_id}')
            r = json.loads(r)
            del r['id'], r['series_id_fk']

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
                            value = r[header][row_num - 1] if row_num - 1 < len(r[header]) else ""

                    sheet.write(row_num, col_num, value)

            # Save the workbook to a BytesIO buffer
            b = io.BytesIO()
            excel.save(b)
            output.AnswerBuffer(b.getvalue(), 'application/vnd.ms-excel')
        # else:
        #     # Handle multiple series export
        #     ServeExportMultipleSeries(output, series_ids)
    except Exception as e:
        print(f"Error decoding request body: {e}")
        output.SendHttpStatusCode(400)
        return

def ServeSaveResults(output: orthanc.RestOutput, url: str, **request) -> None:
    """Serve the save results endpoint"""

    if handle_cors_preflight(output, request):
        return

    method = request.get('method')

    if method != 'POST':
        output.SendMethodNotAllowed('POST')
        return
    
    try:
        # Get request body with parameters
        body = request.get('body', b'{}')
        if isinstance(body, bytes):
            body = body.decode('utf-8')

        
        params = json.loads(body)
        print(params)
        patient = params.get('patient')
        study = params.get('study')
        scanner = params.get('scanner')
        series = params.get('series')
        ct = params.get('ct')
        converted_results = params.get('converted_results')

        success = cho_storage.save_results(patient, study, scanner, series, ct, converted_results)

        progress_tracker.cleanup_calculation(series['uuid'])
        progress_tracker.cleanup_history(series['uuid'])
        if success:
            print(f"Full analysis results saved to database for series {series['series_instance_uid']}")
        else:
            print(f"Failed to save full analysis results to database for series {series['series_instance_uid']}")
            raise Exception("Database save failed")

    except Exception as e:
        print(f"Error saving to database: {str(e)}")
        output.SendHttpStatusCode(500)
        return

    output.AnswerBuffer(b"Results saved successfully", 'text/plain')

def ServeMinIOImage(output: orthanc.RestOutput, url: str, **request) -> None:
    """Serve images from MinIO storage"""
    
    if handle_cors_preflight(output, request):
        return

    method = request.get('method')
    groups = request.get('groups')
    
    if method != 'GET':
        output.SendMethodNotAllowed('GET')
        return
    
    if not groups or len(groups) < 1:
        output.SendHttpStatusCode(400)
        return
    
    # Extract series_instance_uid from URL
    series_instance_uid = groups[0]
    object_name = f"{series_instance_uid}_coronal_view.png"
    
    try:
        # Check if object exists
        try:
            response = cho_storage.minio_client.get_object(
                bucket_name=cho_storage.bucket_name,
                object_name=object_name
            )
            
            # Read image data
            image_data = response.read()
            
            # Set appropriate headers
            output.SetHttpHeader('Content-Type', 'image/png')
            output.SetHttpHeader('Content-Length', str(len(image_data)))
            output.SetHttpHeader('Cache-Control', 'public, max-age=3600')  # Cache for 1 hour
            
            # Send image data
            output.AnswerBuffer(image_data, 'image/png')
            
        except Exception as e:
            print(f"Error retrieving image from MinIO: {str(e)}")
            output.SendHttpStatusCode(404)
            
    except Exception as e:
        print(f"Error accessing MinIO: {str(e)}")
        output.SendHttpStatusCode(500)


def GetImageMetadata(output: orthanc.RestOutput, url: str, **request) -> None:
    """Get metadata about available images for a series"""
    
    if handle_cors_preflight(output, request):
        return

    method = request.get('method')
    groups = request.get('groups')
    
    if method != 'GET':
        output.SendMethodNotAllowed('GET')
        return
    
    if not groups or len(groups) < 1:
        output.SendHttpStatusCode(400)
        return
    
    series_instance_uid = groups[0]
    
    try:
        # List objects with the series prefix
        objects = cho_storage.minio_client.list_objects(
            bucket_name=cho_storage.bucket_name,
            prefix=f"{series_instance_uid}_"
        )
        
        images = []
        for obj in objects:
            # Get object stats for metadata
            object_name = obj.object_name if obj.object_name else ""
            stat = cho_storage.minio_client.stat_object(
                bucket_name=cho_storage.bucket_name,
                object_name=object_name
            )
            
            images.append({
                'name': obj.object_name,
                'size': stat.size,
                'last_modified': stat.last_modified.isoformat() if stat.last_modified else None,
                'content_type': stat.content_type,
                'url': f"/minio-images/{series_instance_uid}/{object_name.split('_', 1)[1]}"
            })
        
        response = {
            'series_instance_uid': series_instance_uid,
            'images': images,
            'count': len(images)
        }
        
        output.AnswerBuffer(json.dumps(response, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
        
    except Exception as e:
        print(f"Error listing images: {str(e)}")
        error_response = {"error": f"Failed to list images: {str(e)}"}
        output.AnswerBuffer(json.dumps(error_response, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
        output.SendHttpStatusCode(500)


def ServeResultsStatistics(output: orthanc.RestOutput, url: str, **request) -> None:
    """Serve the results statistics page"""

    if handle_cors_preflight(output, request):
        return

    method = request.get('method')

    if method != 'GET':
        output.SendMethodNotAllowed('GET')
        return

    # Get statistics data
    try:

            result = cho_storage.get_results_statistics()

            if result:
                total_results_count, global_noise_count, detectability_count, error_count = result
                statistics = {
                    "total_results_count": total_results_count,
                    "global_noise_count": global_noise_count,
                    "detectability_count": detectability_count,
                    "error_count": error_count
                }
                output.AnswerBuffer(json.dumps(statistics, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
            else:
                output.SendHttpStatusCode(404)

    except Exception as e:
        print(f"Error retrieving results statistics: {str(e)}")
        output.SendHttpStatusCode(500)

def GetCalculationStatus(output: orthanc.RestOutput, url: str, **request) -> None:
    """Get the status of a calculation"""

    if handle_cors_preflight(output, request):
        return

    method = request.get('method')
    get = request.get('get', {})

    if method != 'GET':
        output.SendMethodNotAllowed('GET')
        return

    series_id = get.get('series_id')
    action = get.get('action')

    if not series_id:
        output.SendHttpStatusCode(400)
        output.AnswerBuffer(json.dumps({"error": "Missing series_id parameter"}, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
        return
    
    if action == "check":
        status = progress_tracker.get_calculation_status(series_id)
    elif action == "discard":
        status = progress_tracker.cleanup_calculation(series_id)
        progress_tracker.cleanup_history(series_id)
    else:
        output.SendHttpStatusCode(400)
        output.AnswerBuffer(json.dumps({"error": "Invalid action parameter"}, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
        return
    
    if status:
        output.AnswerBuffer(json.dumps(status, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
    else:
        output.SendHttpStatusCode(404)
        output.AnswerBuffer(json.dumps({"error": f"No calculation found for series {series_id}"}, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')


def DelCalculationStatus(output: orthanc.RestOutput, url: str, **request) -> None:
    """Delete the status of a calculation"""

    if handle_cors_preflight(output, request):
        return

    method = request.get('method')
    get = request.get('get', {})

    if method != 'GET':
        output.SendMethodNotAllowed('GET')
        return

    series_id = get.get('series_id')
    
    if not series_id:
        output.SendHttpStatusCode(400)
        output.AnswerBuffer(json.dumps({"error": "Missing series_id parameter"}, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
        return
    
    status = progress_tracker.get_calculation_status(series_id)
    
    if status:
        output.AnswerBuffer(json.dumps(status, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')
    else:
        output.SendHttpStatusCode(404)
        output.AnswerBuffer(json.dumps({"error": f"No calculation found for series {series_id}"}, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')

def GetActiveCalculations(output: orthanc.RestOutput, url: str, **request) -> None:
    """Get all active calculations"""
    if handle_cors_preflight(output, request):
        return
    method = request.get('method')
    if method != 'GET':
        output.SendMethodNotAllowed('GET')
        return
    
    active = progress_tracker.get_all_active_calculations()
    output.AnswerBuffer(json.dumps(active, indent=2, cls=DateTimeEncoder).encode('utf-8'), 'application/json')

def test_database_connection():
    """Test if PostgreSQL database is connected and working"""
    try:
        print("=== Testing Database Connection ===")
        
        system_info = orthanc.RestApiGet('/system')
        system_data = json.loads(system_info)
        
        print(f"Orthanc Name: {system_data.get('Name', 'Unknown')}")
        print(f"Orthanc Version: {system_data.get('Version', 'Unknown')}")
        
        try:
            patients = orthanc.RestApiGet('/patients')
            patient_count = len(json.loads(patients))
            print(f"Database accessible - Found {patient_count} patients")
        except Exception as e:
            print(f"Database access test failed: {str(e)}")
        
        try:
            stats = orthanc.RestApiGet('/statistics')
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
        instances_res = orthanc.RestApiGet(f'/series/{series_uuid}/instances')
        instances_json = json.loads(instances_res)
        print(f"loading: {series_uuid}")
        files = []
        for i in range(len(instances_json)):
            instance_id = instances_json[i]['ID']
            f = orthanc.GetDicomForInstance(instance_id)
            files.append(pydicom.dcmread(io.BytesIO(f)))
        
        slices = [f for f in files if hasattr(f, "SliceLocation")]
        slices = sorted(slices, key=lambda s: s.SliceLocation)

        # Create parser and extract metadata
        dcm_parser = DicomParser(files)
        patient, study, scanner, series, ct = dcm_parser.extract_core()

        # Add series-specific metadata
        series['uuid'] = series_uuid
        series['image_count'] = len(files)
        
        # Save to database using results_storage
        success = cho_storage.save_dicom_headers_only(patient, study, scanner, series, ct)
        
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
    if change_type == orthanc.ChangeType.STABLE_SERIES and resource_type == orthanc.ResourceType.SERIES:
        print(f'Stable series: {resource_id}')
        series_uuid = resource_id
        try:            
            # Check if this is a headers-only instance
            if TEST_TYPE == "None":
                # Save DICOM headers without running analysis
                save_dicom_headers_only(series_uuid)
                
                # Delete series if not saving
                if not SAVE_TYPE:
                    orthanc.RestApiDelete(f'/series/{series_uuid}')
                    print(f"Deleted series after saving headers: {series_uuid}")
                return
            
            test = 'global' if TEST_TYPE == "Global Noise" else 'full'
            
            body = {
                    'series_uuid': series_uuid, 
                    'testType': test,
                    'saveResults': True,
                    'deleteAfterCompletion': not SAVE_TYPE,
                    'report_progress': False
            }
            orthanc.RestApiPostAfterPlugins('/cho-analysis-modal', json.dumps(body).encode('utf-8'))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in automatic CHO analysis for series {series_uuid}: {str(e)}")

orthanc.RegisterOnChangeCallback(OnChange)

orthanc.RegisterRestCallback('/cho-results/(.*)', HandleCHOResult) # type: ignore
orthanc.RegisterRestCallback('/cho-results', GetAllCHOResults) # type: ignore
orthanc.RegisterRestCallback('/cho-results-export', ExportCHOResultsCSV) # type: ignore
orthanc.RegisterRestCallback('/cho-filter-options', GetFilterOptions) # type: ignore
orthanc.RegisterRestCallback('/cho-dashboard', ServeCHODashboard) # type: ignore
orthanc.RegisterRestCallback('/cho-calculation-status', GetCalculationStatus) # type: ignore
orthanc.RegisterRestCallback('/cho-active-calculations', GetActiveCalculations) # type: ignore
orthanc.RegisterRestCallback('/static/(.*)', ServeStaticFile) # type: ignore
orthanc.RegisterRestCallback('/cho-analysis-modal', StartCHOAnalysis) # type: ignore
orthanc.RegisterRestCallback('/minio-images/(.*)', ServeMinIOImage) # type: ignore
orthanc.RegisterRestCallback('/image-metadata/(.*)', GetImageMetadata) # type: ignore
orthanc.RegisterRestCallback('/results-statistics', ServeResultsStatistics) # type: ignore
orthanc.RegisterRestCallback('/cho-save-results', ServeSaveResults) # type: ignore
orthanc.RegisterRestCallback('/cho-export-results', ServeExportResults) # type: ignore


# Update the Orthanc Explorer to add buttons to the toolbar
try:
    with open("/src/static/js/extend-explorer.js", "r", encoding='utf-8') as f:
        orthanc.ExtendOrthancExplorer(f.read())
except FileNotFoundError:
    print("No frontend to update")
