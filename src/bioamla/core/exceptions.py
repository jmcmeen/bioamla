class UnsupportAudioFormatError(Exception):
    """Exception raised for unsupported audio formats."""

    def __init__(self, message="Audio format is not supported."):
        super().__init__(message)
        
class NoModelLoadedError(Exception):
    """Exception raised when no model is loaded."""

    def __init__(self, message="No model is loaded."):
        super().__init__(message)