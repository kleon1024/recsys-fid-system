"""Bootstrap-ensemble training and content-bound artifact publication."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter

import torch

from .contracts import STOCHASTIC_ACTIONS, WORLD_MODEL_VERSION, WorldModelConfig
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
            member, train, sampled, config, device, optimizer, generator,
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


def _train_epoch(member, train, sampled, config, device, optimizer, generator):
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
            latent_noise = torch.randn(
                len(index), config.latent_dim, generator=generator,
            ).to(device)
            output = member(batch, latent_noise=latent_noise)
            loss, _ = world_model_loss(output, batch, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(member.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach())
        batches += 1
    return total / max(batches, 1)


def train_world_ensemble(train: WorldModelSplit, validation: WorldModelSplit,
                         config: WorldModelConfig, device_name: str,
                         calibration_split: WorldModelSplit | None = None,
                         adaptation_split: WorldModelSplit | None = None,
                         structural_adaptation_split: WorldModelSplit | None = None,
                         structural_validation_split: WorldModelSplit | None = None):
    device = torch.device(device_name)
    ensemble = WorldModelEnsemble(config).to(device)
    histories = []
    started = perf_counter()
    for member_index, member in enumerate(ensemble.members):
        histories.append(_train_member(
            member, train, validation, config, device, member_index
        ))
    adaptation = adapt_world_ensemble(
        ensemble, adaptation_split, device,
    ) if adaptation_split is not None else None
    calibration = calibrate_world_ensemble(
        ensemble, calibration_split or validation, device,
    )
    structural_adaptation = adapt_structural_ensemble(
        ensemble, structural_adaptation_split, device,
        structural_validation_split,
    ) if structural_adaptation_split is not None else None
    if adaptation is not None:
        calibration["randomized_adaptation"] = adaptation
    if structural_adaptation is not None:
        calibration["structural_adaptation"] = structural_adaptation
        calibration["post_structural_utility_recenter"] = (
            recenter_utility_ensemble(
                ensemble, calibration_split or validation, device,
            )
        )
    return ensemble, histories, calibration, perf_counter() - started


def _require_structural_pairs(split):
    values = (
        split.structural_intervention_features,
        split.structural_intervention_slates,
        split.structural_intervention_sequences,
        split.structural_intervention_effects,
    )
    if any(value is None for value in values):
        raise ValueError("structural adaptation requires paired interventions")


def _structural_treated_batch(split, control, index, intervention, device):
    return {
        **control,
        "selected_features": split.structural_intervention_features[
            index, intervention
        ].to(device),
        "slate_features": split.structural_intervention_slates[
            index, intervention
        ].to(device),
        "sequence": split.structural_intervention_sequences[
            index, intervention
        ].to(device),
    }


def _structural_adaptation_epoch(member, split, device, optimizer, generator):
    member.train()
    order = torch.randperm(len(split), generator=generator)
    losses = []
    for start in range(0, len(order), member.config.batch_size):
        index = order[start:start + member.config.batch_size]
        control = split.batch(index, device)
        zero_noise = torch.zeros(
            len(index), member.config.latent_dim, device=device,
        )
        control_value = member.utility_value(
            member(control, latent_noise=zero_noise).utility_logit
        )
        effects = []
        for intervention in range(
            split.structural_intervention_effects.shape[1]
        ):
            treated = _structural_treated_batch(
                split, control, index, intervention, device,
            )
            effects.append(
                member.utility_value(
                    member(treated, latent_noise=zero_noise).utility_logit
                ) - control_value
            )
        prediction = torch.stack(effects, dim=1)
        target = split.structural_intervention_effects[index].to(device)
        row_loss = torch.nn.functional.mse_loss(prediction, target)
        mean_loss = torch.nn.functional.mse_loss(
            prediction.mean(dim=0), target.mean(dim=0),
        )
        loss = row_loss + member.config.structural_mean_loss_weight * mean_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(member.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return sum(losses) / max(len(losses), 1)


def _structural_effect_error(member, split, device):
    member.eval()
    predicted_sum = torch.zeros(
        split.structural_intervention_effects.shape[1], device=device,
    )
    target_sum = torch.zeros_like(predicted_sum)
    rows = 0
    with torch.inference_mode():
        for start in range(0, len(split), member.config.batch_size):
            index = torch.arange(
                start, min(start + member.config.batch_size, len(split)),
            )
            control = split.batch(index, device)
            zero_noise = torch.zeros(
                len(index), member.config.latent_dim, device=device,
            )
            baseline = member.utility_value(
                member(control, latent_noise=zero_noise).utility_logit
            )
            for intervention in range(len(predicted_sum)):
                treated = _structural_treated_batch(
                    split, control, index, intervention, device,
                )
                predicted_sum[intervention] += (
                    member.utility_value(
                        member(treated, latent_noise=zero_noise).utility_logit
                    ) - baseline
                ).sum()
            target_sum += split.structural_intervention_effects[index].to(
                device
            ).sum(dim=0)
            rows += len(index)
    predicted = predicted_sum / rows
    target = target_sum / rows
    return float(
        (predicted - target).abs().mean()
        / target.abs().mean().clamp_min(1e-8)
    )


def adapt_structural_ensemble(ensemble, split, device, validation=None):
    """Fit response slopes on train-family pairs, never on held-out worlds."""
    _require_structural_pairs(split)
    validation = split if validation is None else validation
    _require_structural_pairs(validation)
    config = ensemble.config
    histories = []
    for member_index, member in enumerate(ensemble.members):
        optimizer = torch.optim.AdamW(
            tuple(member.utility_head.parameters())
            + tuple(member.utility_feature_adapter.parameters()),
            lr=config.structural_adaptation_learning_rate,
            weight_decay=config.weight_decay,
        )
        generator = torch.Generator().manual_seed(
            config.seed + 300_007 + member_index * 1_009,
        )
        history = []
        best_error = float("inf")
        best_state = None
        for epoch in range(config.structural_adaptation_epochs):
            training_loss = _structural_adaptation_epoch(
                member, split, device, optimizer, generator,
            )
            validation_error = _structural_effect_error(
                member, validation, device,
            )
            history.append({
                "epoch": epoch + 1,
                "loss": training_loss,
                "validation_normalized_mae": validation_error,
            })
            if validation_error < best_error:
                best_error = validation_error
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in member.state_dict().items()
                }
        if best_state is not None:
            member.load_state_dict(best_state)
        histories.append({"member": member_index, "history": history})
    return {
        "method": "train_family_paired_effect_finetune",
        "rows": len(split),
        "epochs": config.structural_adaptation_epochs,
        "learning_rate": config.structural_adaptation_learning_rate,
        "mean_loss_weight": config.structural_mean_loss_weight,
        "selection": "minimum_validation_family_normalized_mae",
        "members": histories,
    }


def adapt_world_ensemble(ensemble, split, device):
    """Learn response slopes on randomized users before held-out calibration."""
    histories = []
    config = ensemble.config
    pairs = _randomized_pairs(split)
    for member_index, member in enumerate(ensemble.members):
        optimizer = torch.optim.AdamW(
            member.parameters(),
            lr=config.randomized_adaptation_learning_rate,
            weight_decay=config.weight_decay,
        )
        generator = torch.Generator().manual_seed(
            config.seed + 200_003 + member_index * 1_009
        )
        member_history = []
        for epoch in range(config.randomized_adaptation_epochs):
            member.train()
            sampled = torch.randperm(len(split), generator=generator)
            loss = _train_epoch(
                member, split, sampled, config, device, optimizer, generator,
            )
            member_history.append({"epoch": epoch + 1, "training_loss": loss})
        pairwise_history = []
        for epoch in range(config.randomized_pairwise_epochs):
            loss = _pairwise_adaptation_epoch(
                member, split, pairs, device, optimizer, generator,
                config.batch_size,
            )
            pairwise_history.append({"epoch": epoch + 1, "loss": loss})
        histories.append({
            "member": member_index,
            "pointwise_history": member_history,
            "pairwise_history": pairwise_history,
        })
    return {
        "method": "user_disjoint_randomized_finetune",
        "rows": len(split),
        "same_user_day_pairs": len(pairs),
        "epochs": config.randomized_adaptation_epochs,
        "pairwise_epochs": config.randomized_pairwise_epochs,
        "learning_rate": config.randomized_adaptation_learning_rate,
        "members": histories,
    }


def _observed_adaptation_utility(split):
    return (
        0.55 * torch.log1p(split.labels[:, 2]) / math.log(181.0)
        + 0.30 * split.labels[:, 5]
        + 0.10 * split.labels[:, 7]
        - 0.05 * split.labels[:, 8]
    )


def _randomized_pairs(split, minimum_delta=0.05):
    if split.event_days is None:
        raise ValueError("randomized pairwise adaptation requires event days")
    utility = _observed_adaptation_utility(split)
    groups = {}
    for index, key in enumerate(zip(
        split.user_ids.tolist(), split.event_days.tolist(), strict=True,
    )):
        groups.setdefault(key, []).append(index)
    pairs = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        values = utility[indices]
        high = indices[int(values.argmax())]
        low = indices[int(values.argmin())]
        if utility[high] - utility[low] >= minimum_delta:
            pairs.append((high, low))
    if not pairs:
        raise ValueError("randomized adaptation produced no informative pairs")
    return torch.tensor(pairs, dtype=torch.long)


def _expected_adaptation_utility(output, batch):
    del batch
    return torch.sigmoid(output.utility_logit)


def _pairwise_adaptation_epoch(member, split, pairs, device, optimizer,
                               generator, batch_size):
    member.train()
    order = pairs[torch.randperm(len(pairs), generator=generator)]
    losses = []
    for start in range(0, len(order), batch_size):
        pair = order[start:start + batch_size]
        positive = split.batch(pair[:, 0], device)
        negative = split.batch(pair[:, 1], device)
        latent_noise = torch.zeros(
            len(pair), member.config.latent_dim, device=device
        )
        positive_score = _expected_adaptation_utility(
            member(positive, latent_noise=latent_noise), positive
        )
        negative_score = _expected_adaptation_utility(
            member(negative, latent_noise=latent_noise), negative
        )
        loss = torch.nn.functional.softplus(
            -(positive_score - negative_score) / 0.05
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(member.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return sum(losses) / len(losses)


def calibrate_world_ensemble(ensemble, calibration_split, device):
    for member in ensemble.members:
        member.action_calibration_scale.fill_(1.0)
        member.action_calibration_bias.zero_()
        member.stay_calibration_scale.fill_(1.0)
        member.stay_calibration_shift.zero_()
        member.utility_calibration_scale.fill_(1.0)
        member.utility_calibration_shift.zero_()
    action_calibration = _calibrate_actions(
        ensemble, calibration_split, device,
    )
    calibration = _calibrate_stay(
        ensemble, calibration_split, device,
    )
    calibration["binary_platt"] = action_calibration
    calibration["utility_affine"] = _calibrate_utility(
        ensemble, calibration_split, device,
    )
    return calibration


def _fit_platt(logits, labels):
    initial_scale = torch.logit(torch.tensor((1.0 - 0.05) / 9.95))
    raw_scale = initial_scale.clone().requires_grad_()
    raw_bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        (raw_scale, raw_bias), max_iter=40, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        scale = 0.05 + 9.95 * torch.sigmoid(raw_scale)
        bias = 10.0 * torch.tanh(raw_bias)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits * scale + bias, labels,
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    scale = 0.05 + 9.95 * torch.sigmoid(raw_scale.detach())
    bias = 10.0 * torch.tanh(raw_bias.detach())
    return float(scale), float(bias)


def _fit_linear_utility(prediction, target):
    centered = prediction - prediction.mean()
    variance = centered.square().mean().clamp_min(1e-8)
    scale = ((centered * (target - target.mean())).mean() / variance).clamp(
        0.1, 4.0
    )
    shift = (target.mean() - scale * prediction.mean()).clamp(-1.0, 1.0)
    return float(scale), float(shift)


def _utility_predictions(member, validation, device, rows):
    predictions = []
    for start in range(0, rows, member.config.batch_size):
        index = torch.arange(start, min(start + member.config.batch_size, rows))
        batch = validation.batch(index, device)
        with torch.inference_mode():
            output = member(
                batch, latent_noise=torch.zeros(
                    len(index), member.config.latent_dim, device=device,
                ),
            )
            predictions.append(member.utility_value(output.utility_logit).cpu())
    return torch.cat(predictions)


def recenter_utility_ensemble(ensemble, validation, device, limit=100_000):
    rows = min(len(validation), limit)
    mask = validation.label_masks[:rows, (2, 5, 7, 8)].amin(dim=1) > 0.5
    target = _observed_adaptation_utility(validation)[:rows][mask]
    target_mean = target.mean()
    report = []
    for member_index, member in enumerate(ensemble.members):
        prediction = _utility_predictions(member, validation, device, rows)[mask]
        before = prediction.mean()
        adjustment = (target_mean - before).clamp(-1.0, 1.0)
        member.utility_calibration_shift.add_(adjustment).clamp_(-1.0, 1.0)
        after = _utility_predictions(member, validation, device, rows)[mask]
        report.append({
            "member": member_index,
            "rows": len(target),
            "target_mean": float(target_mean),
            "before_mean": float(before),
            "shift_adjustment": float(adjustment),
            "after_mean": float(after.mean()),
            "after_mean_error": float((after.mean() - target_mean).abs()),
        })
    return {
        "method": "heldout_post_structural_intercept_only",
        "members": report,
    }


def _calibrate_utility(ensemble, validation, device, limit=100_000):
    rows = min(len(validation), limit)
    mask = validation.label_masks[:rows, (2, 5, 7, 8)].amin(dim=1) > 0.5
    target = _observed_adaptation_utility(validation)[:rows][mask]
    report = []
    for member_index, member in enumerate(ensemble.members):
        prediction = _utility_predictions(
            member, validation, device, rows,
        )[mask]
        scale, shift = _fit_linear_utility(prediction, target)
        member.utility_calibration_scale.fill_(scale)
        member.utility_calibration_shift.fill_(shift)
        calibrated = (prediction * scale + shift).clamp(0.0, 1.0)
        report.append({
            "member": member_index, "rows": len(target),
            "scale": scale, "shift": shift,
            "mae": float((calibrated - target).abs().mean()),
        })
    return {"method": "heldout_per_member_affine", "members": report}


def _calibrate_actions(ensemble, validation, device, limit=100_000):
    rows = min(len(validation), limit)
    report = []
    for member_index, member in enumerate(ensemble.members):
        logits = {action.name: [] for action in STOCHASTIC_ACTIONS}
        for start in range(0, rows, ensemble.config.batch_size):
            index = torch.arange(start, min(start + ensemble.config.batch_size, rows))
            batch = validation.batch(index, device)
            with torch.inference_mode():
                output = member(
                    batch,
                    latent_noise=torch.zeros(
                        len(index), ensemble.config.latent_dim, device=device,
                    ),
                )
            for action in STOCHASTIC_ACTIONS:
                logits[action.name].append(output.logits[action.name].cpu())
        member_report = {}
        for action_index, action in enumerate(STOCHASTIC_ACTIONS):
            mask = validation.label_masks[:rows, action.label_index] > 0.5
            labels = validation.labels[:rows, action.label_index][mask].float()
            values = torch.cat(logits[action.name])[mask].float()
            positives = int(labels.sum())
            negatives = len(labels) - positives
            if positives < 10 or negatives < 10:
                member_report[action.name] = {
                    "status": "insufficient_support",
                    "rows": len(labels),
                    "positives": positives,
                }
                continue
            scale, bias = _fit_platt(values, labels)
            member.action_calibration_scale[action_index] = scale
            member.action_calibration_bias[action_index] = bias
            member_report[action.name] = {
                "status": "calibrated",
                "rows": len(labels),
                "positives": positives,
                "scale": scale,
                "bias": bias,
            }
        report.append({
            "member": member_index,
            "actions": member_report,
        })
    return {
        "method": "heldout_per_member_platt",
        "rows": rows,
        "members": report,
    }


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
    ensemble, histories, artifact_dir: Path, dataset_manifest, calibration=None,
    support_profile=None,
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
        "dataset_source_manifest_sha256s": dataset_manifest.get(
            "source_manifest_sha256s", [dataset_manifest.get("manifest_sha256")]
        ),
        "feature_contract_sha256": dataset_manifest.get(
            "feature_contract_sha256"
        ),
        "feature_coverage": dataset_manifest.get("feature_coverage"),
        "weights_sha256": sha256(weights_path.read_bytes()).hexdigest(),
        "training_history": histories,
        "calibration": calibration,
        "support_profile": support_profile,
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
