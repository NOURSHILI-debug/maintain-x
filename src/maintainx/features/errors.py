class FeatureEngineeringError(ValueError):
    """Base error for feature-engineering failures."""


class LeakageError(FeatureEngineeringError):
    """Raised when a transformation lacks identifiers needed for grouping."""


class InvalidWindowError(FeatureEngineeringError):
    """Raised when a rolling window is invalid."""


class FeatureSelectionError(FeatureEngineeringError):
    """Raised when requested feature columns are unavailable."""