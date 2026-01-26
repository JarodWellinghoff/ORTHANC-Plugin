import json, io
from datetime import datetime, timezone, timedelta
import orthanc
from ..config import TEST_TYPE, SAVE_TYPE
from results_storage import cho_storage

def save_dicom_headers_only(series_uuid):  # (body same as yours)
    ...

def OnChange(change_type, resource_type, resource_id):
    if change_type == orthanc.ChangeType.JOB_SUCCESS:
        print(f"Job succeeded for resource: {resource_id}")
    if change_type == (orthanc.ChangeType.STABLE_SERIES or change_type == orthanc.ChangeType.JOB_SUCCESS) and resource_type == orthanc.ResourceType.SERIES:
        series_uuid = resource_id
        try:
            if TEST_TYPE == "None":
                save_dicom_headers_only(series_uuid)
                if not SAVE_TYPE:
                    orthanc.RestApiDelete(f'/series/{series_uuid}')
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
            import traceback; traceback.print_exc()
            print(f"Error in automatic CHO analysis for series {series_uuid}: {str(e)}")
