"""
Cascade — a layer-wise failure diagnostic for CNNs under distribution shift.

Measures how internal representations (via GradCAM attribution) drift between
clean and shifted inputs, layer by layer, and surfaces where a prediction's
"point of no return" occurs.
"""

from .core import Cascade, find_conv_layers
from .diagnose import DiagnosisResult, diagnose
from .pnr import PNRThresholds, calibrate_pnr_thresholds
from .report import FragilityProfile, fragility_profile

__all__ = [
    "Cascade",
    "find_conv_layers",
    "DiagnosisResult",
    "diagnose",
    "PNRThresholds",
    "calibrate_pnr_thresholds",
    "FragilityProfile",
    "fragility_profile",
]

__version__ = "0.1.0"
