class BullseyeError(Exception):
    """Base exception for bullseye package."""


class ModelLoadError(BullseyeError):
    pass


class OnnxExportError(BullseyeError):
    pass


class OnnxLoadError(BullseyeError):
    pass


class InferenceError(BullseyeError):
    pass

