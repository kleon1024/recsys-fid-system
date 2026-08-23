"""Bootstrap-ensemble training and content-bound artifact publication."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter

import torch

from .contracts import WORLD_MODEL_VERSION, WorldModelConfig
from .data import WorldModelSplit
from .ensemble import WorldModelEnsemble
from .ensemble import StructuralNoise
from .loss import world_model_loss


def _validation_loss(member, split, config, device, limit=50_000):
    member.eval()
    losses = []
    with torch.inference_mode():
        for start in range(0, min(len(split), limit), config.batch_size):
            index = torch.arange(start, min(start + config.batch_size, len(split)))
            batch = split.batch(index, device)
            output = member(
                batch,
                latent_noise=torch.zeros(len(index), config.latent_dim, device=device),
            )
            loss, _ = world_model_loss(output, batch, config)
            losses.append(float(loss))
    return sum(losses) / max(len(losses), 1)


def _train_member(member, train, validation, config, device, member_index):
    generator = torch.Generator().manual_seed(config.seed + 1009 * member_index)
    optimizer = torch.optim.AdamW(
        member.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history = []
    best_validation = float("inf")
    best_state = None
    for epoch in range(config.epochs):
        member.train()
        sampled = torch.randint(len(train), (len(train),), generator=generator)
        training_loss = _train_epoch(
            member, train, sampled, config, device, optimizer
        )
        validation_loss = _validation_loss(
            member, validation, config, device
        )
        history.append({
            "epoch": epoch + 1,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
        })
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in member.state_dict().items()
            }
    if best_state is not None:
        member.load_state_dict(best_state)
    return history


def _train_epoch(member, train, sampled, config, device, optimizer):
    total = 0.0
    batches = 0
    for start in range(0, len(sampled), config.batch_size):
        index = sampled[start:start + config.batch_size]
        batch = train.batch(index, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = member(batch)
            loss, _ = world_model_loss(output, batch, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(member.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach())
        batches += 1
    return total / max(batches, 1)


def train_world_ensemble(train: WorldModelSplit, validation: WorldModelSplit,
                         config: WorldModelConfig, device_name: str,
                         calibration_split: WorldModelSplit | None = None):
    device = torch.device(device_name)
    ensemble = WorldModelEnsemble(config).to(device)
    histories = []
    started = perf_counter()
    for member_index, member in enumerate(ensemble.members):
        histories.append(_train_member(
            member, train, validation, config, device, member_index
        ))
    calibration = _calibrate_stay(
        ensemble, calibration_split or validation, device
    )
    return ensemble, histories, calibration, perf_counter() - started


def _calibrate_stay(ensemble, validation, device, limit=100_000):
    observable = validation.label_masks[:limit, 2] > 0.5
    observed = validation.labels[:limit, 2][observable]
    if not len(observed):
        raise ValueError("stay calibration requires observable stay labels")
    generated = _sample_stay(ensemble, validation, device, limit)
    observed_normalized = torch.log1p(observed) / math.log(181.0)
    generated_normalized = torch.log1p(generated) / math.log(181.0)
    observed_quantiles = torch.quantile(
        observed_normalized, torch.tensor((0.5, 0.9))
    )
    generated_quantiles = torch.quantile(
        generated_normalized, torch.tensor((0.5, 0.9))
    )
    scale = float((
        (observed_quantiles[1] - observed_quantiles[0])
        / (generated_quantiles[1] - generated_quantiles[0]).clamp_min(1e-4)
    ).clamp(0.1, 4.0))
    shift = float(observed_quantiles[0] - scale * generated_quantiles[0])
    for member in ensemble.members:
        member.stay_calibration_scale.fill_(scale)
        member.stay_calibration_shift.fill_(shift)
    calibrated = _sample_stay(ensemble, validation, device, limit)
    return {
        "method": "heldout_log_stay_p50_p90_affine",
        "rows": len(observed),
        "observed_seconds": _stay_quantiles(observed),
        "uncalibrated_seconds": _stay_quantiles(generated),
        "calibrated_seconds": _stay_quantiles(calibrated),
        "normalized_scale": scale,
        "normalized_shift": shift,
    }


def _sample_stay(ensemble, split, device, limit):
    generated = []
    rows = min(len(split), limit)
    for start in range(0, rows, ensemble.config.batch_size):
        index = torch.arange(start, min(start + ensemble.config.batch_size, rows))
        batch = split.batch(index, device)
        noise = StructuralNoise.generate(
            len(index), ensemble.config, device, ensemble.config.seed + 700_000 + start
        )
        generated.extend(
            sample["stay_seconds"].detach().cpu()
            for sample in ensemble.sample_members(batch, noise)
        )
    return torch.cat(generated)


def _stay_quantiles(values):
    quantiles = torch.quantile(values, torch.tensor((0.5, 0.9)))
    return {"p50": float(quantiles[0]), "p90": float(quantiles[1])}


def save_world_ensemble(
    ensemble, histories, artifact_dir: Path, dataset_manifest, calibration=None
):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "world_model.pt"
    torch.save({
        "version": WORLD_MODEL_VERSION,
        "config": asdict(ensemble.config),
        "members": [member.state_dict() for member in ensemble.members],
    }, weights_path)
    manifest = {
        "schema": "neural-scm-world-model-artifact-v1",
        "world_model_version": WORLD_MODEL_VERSION,
        "config": asdict(ensemble.config),
        "dataset_manifest_sha256": dataset_manifest.get("manifest_sha256"),
        "dataset_authority_bundle_id": dataset_manifest.get("authority_bundle_id"),
        "weights_sha256": sha256(weights_path.read_bytes()).hexdigest(),
        "training_history": histories,
        "calibration": calibration,
        "authority_status": "research_challenger",
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    return manifest


def load_world_ensemble(artifact_dir: Path, device_name: str):
    payload = torch.load(
        artifact_dir / "world_model.pt", map_location=device_name, weights_only=False
    )
    config = WorldModelConfig(**payload["config"])
    ensemble = WorldModelEnsemble(config).to(device_name)
    for member, state in zip(ensemble.members, payload["members"], strict=True):
        member.load_state_dict(state, strict=False)
    return ensemble
