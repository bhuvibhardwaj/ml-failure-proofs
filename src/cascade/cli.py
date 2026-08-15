"""
Command-line interface.

Usage:
    cascade analyze \
        --model path/to/model.pt \
        --dataset cifar10 \
        --data-root ./data \
        --shift aggressive \
        --n-samples 500 \
        --device cuda

`--model` must point to a file loadable with `torch.load(...)` that yields
either a full nn.Module or a state_dict-compatible checkpoint alongside
`--model-class` (a dotted import path to the class, e.g.
`torchvision.models.resnet18`) and `--num-classes`.

This CLI intentionally supports only torchvision's built-in datasets
(cifar10, mnist) out of the box, since arbitrary dataset loading is
inherently project-specific. For anything else, use the Python API directly
(see cascade.core.Cascade and cascade.report.fragility_profile) and pass
your own DataLoader.
"""

from __future__ import annotations

import argparse
import importlib
import sys

import torch
import torchvision
import torchvision.transforms as T

from .core import Cascade
from .report import fragility_profile
from .shift import get_preset

DATASETS = {
    "cifar10": torchvision.datasets.CIFAR10,
    "mnist": torchvision.datasets.MNIST,
}


def _load_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _load_model(args) -> torch.nn.Module:
    obj = torch.load(args.model, map_location=args.device)
    if isinstance(obj, torch.nn.Module):
        return obj
    if not args.model_class:
        raise ValueError(
            "Checkpoint appears to be a state_dict, not a full model. "
            "Pass --model-class (e.g. torchvision.models.resnet18) to "
            "reconstruct the architecture before loading weights."
        )
    model_cls = _load_class(args.model_class)
    model = model_cls(num_classes=args.num_classes) if args.num_classes else model_cls()
    model.load_state_dict(obj)
    return model


def _collect_misclassified(model, dataset_clean, dataset_shift, device, limit):
    """Find (clean_img, shifted_img, true_label, pred_label) for misclassified shifted inputs."""
    model.eval()
    samples = []
    with torch.no_grad():
        for i in range(len(dataset_shift)):
            shifted_img, true_label = dataset_shift[i]
            pred = model(shifted_img.unsqueeze(0).to(device)).argmax(dim=1).item()
            if pred != true_label:
                clean_img, _ = dataset_clean[i]
                samples.append((clean_img, shifted_img, true_label, pred))
            if len(samples) >= limit:
                break
    return samples


def cmd_analyze(args):
    device = args.device
    model = _load_model(args).to(device)

    if args.dataset not in DATASETS:
        print(f"Unknown dataset '{args.dataset}'. Choices: {list(DATASETS.keys())}",
              file=sys.stderr)
        sys.exit(1)

    dataset_cls = DATASETS[args.dataset]
    clean_transform = T.Compose([T.ToTensor()])
    shift_transform = T.Compose([get_preset(args.shift), T.ToTensor()])

    dataset_clean = dataset_cls(
        root=args.data_root, train=False, download=True, transform=clean_transform
    )
    dataset_shift = dataset_cls(
        root=args.data_root, train=False, download=True, transform=shift_transform
    )

    print("Finding misclassified samples under shift...")
    samples = _collect_misclassified(
        model, dataset_clean, dataset_shift, device, limit=args.n_samples
    )
    print(f"Found {len(samples)} misclassified samples "
          f"(requested up to {args.n_samples}).")

    if not samples:
        print("No misclassified samples found under this shift; nothing to analyze.")
        return

    cascade = Cascade(model, max_layers=args.max_layers, device=device)
    profile = fragility_profile(cascade, samples)
    print()
    print(profile.summary())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cascade")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser(
        "analyze", help="Run a fragility profile over misclassified shifted samples"
    )
    analyze.add_argument("--model", required=True, help="Path to a torch.load-able checkpoint")
    analyze.add_argument(
        "--model-class", default=None,
        help="Dotted path to model class, required if --model is a state_dict"
    )
    analyze.add_argument("--num-classes", type=int, default=None)
    analyze.add_argument(
        "--dataset", required=True, choices=list(DATASETS.keys())
    )
    analyze.add_argument("--data-root", default="./data")
    analyze.add_argument(
        "--shift", default="aggressive", choices=["mild", "aggressive"]
    )
    analyze.add_argument("--n-samples", type=int, default=500)
    analyze.add_argument(
        "--max-layers", type=int, default=None,
        help="Cap the number of conv layers instrumented (sampled evenly across depth)"
    )
    analyze.add_argument("--device", default="cpu")
    analyze.set_defaults(func=cmd_analyze)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
