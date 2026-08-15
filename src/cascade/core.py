"""
Core Cascade engine.

This is a generalized version of the GradCAM drift metric D(k) used in the
ResNet18/CIFAR-10 experiment: for a given layer k,

    D(k) = || GradCAM_shifted(k) - GradCAM_clean(k) ||

Unlike the original notebook (which hardcoded eight specific ResNet18
sublayers), this module auto-discovers Conv2d layers from an arbitrary
PyTorch model so the same engine works on any CNN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from captum.attr import LayerGradCam
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "cascade requires captum. Install it with `pip install captum`."
    ) from exc


def find_conv_layers(
    model: nn.Module, max_layers: Optional[int] = None
) -> List[Tuple[str, nn.Module]]:
    """
    Auto-discover Conv2d layers in a model, in module-registration order.

    If max_layers is given and fewer than the total number of conv layers,
    layers are sampled evenly across depth (not just the first N) so the
    resulting set still spans early/mid/late representations.
    """
    layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    if not layers:
        raise ValueError(
            "No nn.Conv2d layers found in this model. "
            "Cascade currently only supports CNNs; pass `layers` explicitly "
            "to Cascade(...) if your architecture uses a different conv type."
        )
    if max_layers is not None and max_layers < len(layers):
        idx = torch.linspace(0, len(layers) - 1, max_layers).round().long().tolist()
        idx = sorted(set(idx))
        layers = [layers[i] for i in idx]
    return layers


@dataclass
class LayerDrift:
    """D(k) values for a single sample, across all instrumented layers."""

    layer_names: List[str]
    dk_true: List[float]
    dk_pred: List[float]


class Cascade:
    """
    Wraps a trained PyTorch model with GradCAM hooks on its conv layers and
    computes attribution drift D(k) between a clean/shifted image pair.

    Example:
        model = load_my_trained_model()
        cascade = Cascade(model, device="cuda")
        drift = cascade.layer_drift(clean_img, shifted_img, true_label, pred_label)
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[Sequence[Tuple[str, nn.Module]]] = None,
        max_layers: Optional[int] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.model = model.to(device).eval()
        self.layers = list(layers) if layers is not None else find_conv_layers(
            self.model, max_layers=max_layers
        )
        self.layer_names = [name for name, _ in self.layers]
        self.gradcam_layers = [
            LayerGradCam(self.model, module) for _, module in self.layers
        ]

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def _attribute(
        self, image: torch.Tensor, target_class: int
    ) -> List[torch.Tensor]:
        """Compute GradCAM attribution at every instrumented layer for one image."""
        img = image
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(self.device).requires_grad_(True)
        return [
            gc.attribute(img, target=target_class) for gc in self.gradcam_layers
        ]

    def dk(
        self,
        clean_image: torch.Tensor,
        shifted_image: torch.Tensor,
        target_class: int,
    ) -> List[float]:
        """
        D(k) = || GradCAM_shifted(k) - GradCAM_clean(k) || for every layer k.

        Accepts unbatched (C, H, W) or batched (1, C, H, W) tensors.
        """
        attrs_clean = self._attribute(clean_image, target_class)
        attrs_shifted = self._attribute(shifted_image, target_class)
        return [
            (a_shift - a_clean).norm().item()
            for a_clean, a_shift in zip(attrs_clean, attrs_shifted)
        ]

    def layer_drift(
        self,
        clean_image: torch.Tensor,
        shifted_image: torch.Tensor,
        true_label: int,
        pred_label: int,
    ) -> LayerDrift:
        """
        Compute D(k) against both the true label (correct semantic path) and
        the predicted label (the model's own, possibly wrong, reasoning path).
        This is the dual-label split from the original experiment.
        """
        dk_true = self.dk(clean_image, shifted_image, true_label)
        dk_pred = self.dk(clean_image, shifted_image, pred_label)
        return LayerDrift(
            layer_names=self.layer_names, dk_true=dk_true, dk_pred=dk_pred
        )
