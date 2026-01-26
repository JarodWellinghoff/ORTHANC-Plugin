from ..utils.http import handle_cors_preflight, set_cors_headers
import orthanc
from results_storage import cho_storage
from datetime import datetime, timezone
import json
from ..utils.json import DateTimeEncoder

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
