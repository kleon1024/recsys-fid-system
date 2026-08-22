"""Generate and optionally train the heavy Feed model on a scale scenario."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as functional

from ..surfaces.contracts import SURFACE_SPECS
from ..surfaces.model import build_surface_model
from .contracts import FEED_TASKS, ScaleConfig
from .synthetic import build_scale_dataset, summarize_distribution


def train(dataset, device: str, epochs: int) -> dict[str, object]:
    torch.manual_seed(dataset.config.seed)
    model = build_surface_model(SURFACE_SPECS["feed_poi_video"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    split = int(dataset.examples * 0.8)
    positive = torch.from_numpy(dataset.labels[:split].sum(axis=0)).clamp_min(1.0)
    pos_weight = ((split - positive) / positive).clamp(max=200.0).to(device)
    rng = np.random.default_rng(dataset.config.seed + 1)
    model.train()
    for _ in range(epochs):
        for start in range(0, split, 1_024):
            index = rng.integers(0, split, size=min(1_024, split))
            features = torch.from_numpy(dataset.dense_features[index]).to(device)
            sequence = torch.from_numpy(dataset.sequences[index]).to(device)
            labels = torch.from_numpy(dataset.labels[index]).to(device)
            outputs = model(features, sequence)
            losses = [
                functional.binary_cross_entropy_with_logits(
                    outputs[task], labels[:, task_index], pos_weight=pos_weight[task_index]
                )
                for task_index, task in enumerate(FEED_TASKS)
            ]
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return {
        "device": device,
        "epochs": epochs,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "final_train_loss": float(loss.detach().cpu()),
        "train_examples": split,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-impressions", type=int, default=1_000_000)
    parser.add_argument("--anchor-rate", type=float, default=0.02)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    dataset = build_scale_dataset(
        ScaleConfig(main_impressions=args.main_impressions, anchor_rate=args.anchor_rate)
    )
    report = {"distribution": summarize_distribution(dataset)}
    if not args.no_train:
        report["training"] = train(dataset, args.device, args.epochs)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
