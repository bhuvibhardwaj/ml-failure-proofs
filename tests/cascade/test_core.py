import torch
import torch.nn as nn

from cascade.core import Cascade, find_conv_layers
from cascade.diagnose import diagnose
from cascade.pnr import calibrate_pnr_thresholds
from cascade.report import fragility_profile


class TinyCNN(nn.Module):
    """Minimal 2-conv-layer CNN, just enough to exercise the engine."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _fake_pair():
    clean = torch.rand(1, 8, 8)
    shifted = torch.rand(1, 8, 8)
    return clean, shifted


def test_find_conv_layers():
    model = TinyCNN()
    layers = find_conv_layers(model)
    assert len(layers) == 2

    layers_capped = find_conv_layers(model, max_layers=1)
    assert len(layers_capped) == 1


def test_dk_returns_one_value_per_layer():
    model = TinyCNN()
    cascade = Cascade(model)
    clean, shifted = _fake_pair()
    dk = cascade.dk(clean, shifted, target_class=0)
    assert len(dk) == cascade.n_layers == 2
    assert all(isinstance(v, float) for v in dk)
    assert all(v >= 0 for v in dk)  # norms are non-negative


def test_layer_drift_dual_label():
    model = TinyCNN()
    cascade = Cascade(model)
    clean, shifted = _fake_pair()
    drift = cascade.layer_drift(clean, shifted, true_label=0, pred_label=1)
    assert len(drift.dk_true) == len(drift.dk_pred) == cascade.n_layers


def test_diagnose_produces_verdict():
    model = TinyCNN()
    cascade = Cascade(model)
    clean, shifted = _fake_pair()
    result = diagnose(cascade, clean, shifted, true_label=0, pred_label=1)
    assert result.verdict in {"stable", "unstable"}
    assert len(result.dk_true) == cascade.n_layers


def test_fragility_profile_over_multiple_samples():
    model = TinyCNN()
    cascade = Cascade(model)
    samples = []
    for _ in range(5):
        clean, shifted = _fake_pair()
        samples.append((clean, shifted, 0, 1))
    profile = fragility_profile(cascade, samples)
    assert profile.n_samples == 5
    assert len(profile.true_stats.mean) == cascade.n_layers
    assert len(profile.growth_true) == cascade.n_layers - 1
    assert len(profile.ttests_true) == cascade.n_layers - 1

def test_diagnose_with_no_threshold_reports_not_computed():
    """
    Regression test: diagnose() with threshold=None must NOT silently fall
    back to a self-referential auto-threshold. If this ever returns a real
    point_of_no_return instead of None, something is reintroducing the old
    _auto_threshold() behavior (or a stale build is being imported).
    """
    model = TinyCNN()
    cascade = Cascade(model)
    clean, shifted = _fake_pair()
    result = diagnose(cascade, clean, shifted, true_label=0, pred_label=1)
    assert result.pnr_thresholds is None
    assert result.point_of_no_return is None
    assert "not computed" in result.summary()


def test_calibrated_pnr_thresholds_are_falsifiable():
    model = TinyCNN()
    cascade = Cascade(model)

    clean = torch.rand(1, 8, 8)
    baseline_pairs = [(clean, clean, 0) for _ in range(3)]
    thresholds = calibrate_pnr_thresholds(cascade, baseline_pairs, quantile=0.95)

    same = diagnose(cascade, clean, clean, true_label=0, pred_label=1, threshold=thresholds)
    assert same.point_of_no_return is None

    shifted = torch.rand(1, 8, 8)
    different = diagnose(
        cascade, clean, shifted, true_label=0, pred_label=1, threshold=thresholds
    )
    assert different.pnr_thresholds is not None
