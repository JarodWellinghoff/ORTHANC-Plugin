from enum import Enum

class ChangeType(Enum):
    """
    The supported types of changes that can be signaled to the change callback. Note: this enum is not used to store changes in the DB !
    """

    """
    Series is now complete
    """
    COMPLETED_SERIES = 0

    """
    Deleted resource
    """
    DELETED = 1

    """
    A new instance was added to this resource
    """
    NEW_CHILD_INSTANCE = 2

    """
    New instance received
    """
    NEW_INSTANCE = 3

    """
    New patient created
    """
    NEW_PATIENT = 4

    """
    New series created
    """
    NEW_SERIES = 5

    """
    New study created
    """
    NEW_STUDY = 6

    """
    Timeout: No new instance in this patient
    """
    STABLE_PATIENT = 7

    """
    Timeout: No new instance in this series
    """
    STABLE_SERIES = 8

    """
    Timeout: No new instance in this study
    """
    STABLE_STUDY = 9

    """
    Orthanc has started
    """
    ORTHANC_STARTED = 10

    """
    Orthanc is stopping
    """
    ORTHANC_STOPPED = 11

    """
    Some user-defined attachment has changed for this resource
    """
    UPDATED_ATTACHMENT = 12

    """
    Some user-defined metadata has changed for this resource
    """
    UPDATED_METADATA = 13

    """
    The list of Orthanc peers has changed
    """
    UPDATED_PEERS = 14

    """
    The list of DICOM modalities has changed
    """
    UPDATED_MODALITIES = 15

    """
    New Job submitted
    """
    JOB_SUBMITTED = 16

    """
    A Job has completed successfully
    """
    JOB_SUCCESS = 17

    """
    A Job has failed
    """
    JOB_FAILURE = 18
