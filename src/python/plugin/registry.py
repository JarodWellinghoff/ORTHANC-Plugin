import orthanc
from .automation.on_change import OnChange

# Import handlers
from .handlers.cho_results import HandleCHOResult, GetAllCHOResults, ExportCHOResultsCSV, GetFilterOptions
from .handlers.cho_analysis import StartCHOAnalysis
from .handlers.status import GetCalculationStatus, DelCalculationStatus, GetActiveCalculations
from .handlers.minio_images import ServeMinIOImage, GetImageMetadata
from .handlers.stats import ServeResultsStatistics
from .handlers.sse import StreamChoProgress
from .handlers.dicom_modalities import ListDicomModalities
from .handlers.dicom_store import QueryDicomStore
from .handlers.dicom_pull import HandleDicomPullBatches, HandleDicomPullBatchDetail, HandleDicomPullRecover

def register_routes():
    orthanc.RegisterRestCallback('/cho-results/(.*)', HandleCHOResult)
    orthanc.RegisterRestCallback('/cho-results', GetAllCHOResults)
    orthanc.RegisterRestCallback('/cho-results-export', ExportCHOResultsCSV)
    orthanc.RegisterRestCallback('/cho-filter-options', GetFilterOptions)
    orthanc.RegisterRestCallback('/cho-calculation-status', GetCalculationStatus)
    orthanc.RegisterRestCallback('/cho-active-calculations', GetActiveCalculations)
    orthanc.RegisterRestCallback('/cho-analysis-modal', StartCHOAnalysis)
    orthanc.RegisterRestCallback('/minio-images/(.*)', ServeMinIOImage)
    orthanc.RegisterRestCallback('/image-metadata/(.*)', GetImageMetadata)
    orthanc.RegisterRestCallback('/results-statistics', ServeResultsStatistics)
    orthanc.RegisterRestCallback('/cho-save-results', ... )          # keep if you want, or merge into cho_results
    orthanc.RegisterRestCallback('/cho-export-results', ... )        # ditto
    orthanc.RegisterRestCallback('/dicom-modalities', ListDicomModalities)
    orthanc.RegisterRestCallback('/dicom-store/query', QueryDicomStore)
    orthanc.RegisterRestCallback('/dicom-pull/batches/(.*)', HandleDicomPullBatchDetail)
    orthanc.RegisterRestCallback('/dicom-pull/batches', HandleDicomPullBatches)
    orthanc.RegisterRestCallback('/dicom-pull/recover', HandleDicomPullRecover)
    orthanc.RegisterRestCallback('/cho-progress/stream', StreamChoProgress)

def register_onchange():
    orthanc.RegisterOnChangeCallback(OnChange)
