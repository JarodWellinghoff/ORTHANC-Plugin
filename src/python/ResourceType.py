from enum import Enum

class ResourceType(Enum):
    """
    The supported types of DICOM resources.
    """

    """
    Patient
    """
    PATIENT = 0

    """
    Study
    """
    STUDY = 1

    """
    Series
    """
    SERIES = 2

    """
    Instance
    """
    INSTANCE = 3

    """
    Unavailable resource type
    """
    NONE = 4