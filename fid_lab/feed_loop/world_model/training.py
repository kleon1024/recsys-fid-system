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
                         config: WorldModelConfig, device_name: str):
    device = torch.device(device_name)
    ensemble = WorldModelEnsemble(config).to(device)
    histories = []
    started = perf_counter()
    for member_index, member in enumerate(ensemble.members):
        histories.append(_train_member(
            member, train, validation, config, device, member_index
        ))
    calibration = _calibrate_stay(ensemble, validation, device)
    return ensemble, histories, calibration, perf_counter() - started


def _calibrate_stay(ensemble, validation, device, limit=100_000):
    observed = validation.labels[:limit, 2]
    generated = []
    for start in range(0, min(len(validation), limit), ensemble.config.batch_size):
        index = torch.arange(start, min(start + ensemble.config.batch_size, limit))
        batch = validation.batch(index, device)
        noise = StructuralNoise.generate(
            len(index), ensemble.config, device, ensemble.config.seed + 700_000 + start
        )
        generated.extend(
            sample["stay_seconds"].detach().cpu()
            for sample in ensemble.sample_members(batch, noise)
        )
    observed_median = float(observed.median())
    simulated_median = float(torch.cat(generated).median())
    shift = (
        math.log1p(observed_median) - math.log1p(simulated_median)
    ) / math.log(181.0)
    for member in ensemble.members:
        member.stay_calibration_shift.fill_(shift)
    return {
        "method": "validation_log_stay_median_shift",
        "rows": min(len(validation), limit),
        "observed_median_seconds": observed_median,
        "uncalibrated_median_seconds": simulated_median,
        "normalized_log_shift": shift,
    }


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
        member.load_state_dict(state)
    return ensemble
