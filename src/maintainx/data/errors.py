class CMAPSSDataError(Exception):
    """Base exception for all C-MAPSS data pipeline errors."""


class ArchiveNotFoundError(CMAPSSDataError):
    """Raised when the C-MAPSS archive cannot be located."""


class InvalidArchiveError(CMAPSSDataError):
    """Raised when an archive is not a valid C-MAPSS ZIP file."""


class MissingFilesError(CMAPSSDataError):
    """Raised when required C-MAPSS files are missing."""


class DataValidationError(CMAPSSDataError):
    """Raised when loaded data fails validation."""
