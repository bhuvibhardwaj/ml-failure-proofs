"""
Aggregate "fragility profile" reporting.

Generalizes the stats block from the ResNet18/CIFAR-10 experiment: given
D(k) scores collected across many misclassified samples, compute per-layer
means/CIs, locality growth ratios L(k) = D(k)/D(k-1), and layer-to-layer
t-tests testing whether drift significantly increases with depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy import stats

from .core import Cascade


@dataclass
class LayerStats:
    mean: np.ndarray
    std: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray


@dataclass
class FragilityProfile:
    layer_names: List[str]
    n_samples: int
    true_stats: LayerStats
    pred_stats: LayerStats
    growth_true: List[float]
    growth_pred: List[float]
    ttests_true: List[Tuple[float, float]]  # (t, p) per adjacent layer pair
    ttests_pred: List[Tuple[float, float]]

    def summary(self) -> str:
        lines = [
            f"Fragility profile ({self.n_samples} samples, "
            f"{len(self.layer_names)} layers)",
            "",
            "D(k) — drift relative to TRUE label:",
        ]
        for i, name in enumerate(self.layer_names):
            lines.append(
                f"  {name}: {self.true_stats.mean[i]:.4f} "
                f"± {self.true_stats.std[i]:.4f}  "
                f"CI=[{self.true_stats.ci_lo[i]:.4f}, "
                f"{self.true_stats.ci_hi[i]:.4f}]"
            )
        lines.append("")
        lines.append("D(k) — drift relative to PREDICTED label:")
        for i, name in enumerate(self.layer_names):
            lines.append(
                f"  {name}: {self.pred_stats.mean[i]:.4f} "
                f"± {self.pred_stats.std[i]:.4f}  "
                f"CI=[{self.pred_stats.ci_lo[i]:.4f}, "
                f"{self.pred_stats.ci_hi[i]:.4f}]"
            )
        lines.append("")
        lines.append("Layer-to-layer t-tests (TRUE label drift):")
        for i, (t, p) in enumerate(self.ttests_true):
            sig = "significant" if p < 0.05 else "not significant"
            lines.append(
                f"  {self.layer_names[i + 1]} vs {self.layer_names[i]}: "
                f"t={t:.3f} p={p:.4f} ({sig})"
            )
        return "\n".join(lines)


def _summarise(scores: List[List[float]]) -> LayerStats:
    means, stds, ci_lo, ci_hi = [], [], [], []
    for layer_scores in scores:
        arr = np.array(layer_scores)
        m = np.mean(arr)
        s = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
        n = len(arr)
        if n > 1:
            se = s / np.sqrt(n)
            t_crit = stats.t.ppf(0.975, df=n - 1)
            lo, hi = m - t_crit * se, m + t_crit * se
        else:
            lo, hi = m, m
        means.append(m)
        stds.append(s)
        ci_lo.append(lo)
        ci_hi.append(hi)
    return LayerStats(
        mean=np.array(means),
        std=np.array(stds),
        ci_lo=np.array(ci_lo),
        ci_hi=np.array(ci_hi),
    )


def _growth_ratios(means: np.ndarray) -> List[float]:
    return [means[i] / means[i - 1] if means[i - 1] != 0 else float("nan")
            for i in range(1, len(means))]


def _layer_ttests(scores: List[List[float]]) -> List[Tuple[float, float]]:
    results = []
    for k in range(1, len(scores)):
        t, p = stats.ttest_ind(scores[k], scores[k - 1])
        results.append((float(t), float(p)))
    return results


def fragility_profile(
    cascade: Cascade,
    samples: Iterable[Tuple],
    n_samples: Optional[int] = None,
) -> FragilityProfile:
    """
    Build an aggregate fragility profile over a set of misclassified samples.

    `samples` should yield (clean_image, shifted_image, true_label, pred_label)
    tuples — e.g. pre-filtered misclassified examples from a shifted test set.
    """
    all_true: List[List[float]] = [[] for _ in range(cascade.n_layers)]
    all_pred: List[List[float]] = [[] for _ in range(cascade.n_layers)]

    count = 0
    for clean_img, shifted_img, true_label, pred_label in samples:
        drift = cascade.layer_drift(clean_img, shifted_img, true_label, pred_label)
        for k in range(cascade.n_layers):
            all_true[k].append(drift.dk_true[k])
            all_pred[k].append(drift.dk_pred[k])
        count += 1
        if n_samples is not None and count >= n_samples:
            break

    if count == 0:
        raise ValueError("No samples provided to fragility_profile().")

    true_stats = _summarise(all_true)
    pred_stats = _summarise(all_pred)

    return FragilityProfile(
        layer_names=cascade.layer_names,
        n_samples=count,
        true_stats=true_stats,
        pred_stats=pred_stats,
        growth_true=_growth_ratios(true_stats.mean),
        growth_pred=_growth_ratios(pred_stats.mean),
        ttests_true=_layer_ttests(all_true),
        ttests_pred=_layer_ttests(all_pred),
    )
