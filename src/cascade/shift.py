"""
Optional shift-synthesis presets.

Cascade's core engine works on any pre-built (clean, shifted) image pair —
it does not require this module. These presets exist purely as a
convenience for users who don't already have a shifted dataset, mirroring
the corruptions used in earlier experiments (rotation + Gaussian blur).

IMPORTANT: use a *deterministic* rotation angle, not a random range. A
random range resampled on every access will silently corrupt D(k)
measurements, since "clean" and "shifted" versions of the same index must
correspond to the same underlying image.
"""

from __future__ import annotations

from typing import Callable

import torchvision.transforms as T

PRESETS = {
    "mild": T.Compose(
        [
            T.RandomRotation(degrees=(30, 30)),
            T.GaussianBlur(kernel_size=3),
        ]
    ),
    "aggressive": T.Compose(
        [
            T.RandomRotation(degrees=(75, 75)),
            T.GaussianBlur(kernel_size=7),
        ]
    ),
}


def get_preset(name: str) -> Callable:
    """Return a torchvision transform for a named shift preset."""
    if name not in PRESETS:
        raise KeyError(
            f"Unknown shift preset '{name}'. Available presets: "
            f"{list(PRESETS.keys())}"
        )
    return PRESETS[name]


def custom_rotation_blur(degrees: float, blur_kernel: int) -> Callable:
    """Build a deterministic rotation+blur shift with custom parameters."""
    if blur_kernel % 2 == 0:
        raise ValueError("blur_kernel must be odd.")
    return T.Compose(
        [
            T.RandomRotation(degrees=(degrees, degrees)),
            T.GaussianBlur(kernel_size=blur_kernel),
        ]
    )
