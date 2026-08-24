"""Request-aware search ranker training, calibration, and artifact replay."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as functional
from xgboost import XGBRanker

from ...training.common.tensor_ops import gather_candidates
from ..contracts import LOCAL_SEARCH_TASKS
from .architectures import MODEL_FACTORIES


def _xgb_array(tensor):
    if tensor.is_cuda:
        import cupy

        return cupy.from_dlpack(tensor.detach().contiguous())
    return tensor.detach().cpu().numpy()


def _numpy(values):
    if hasattr(values, "get"):
        return values.get()
    return np.asarray(values)


@dataclass(frozen=True)
class TorchRankerBundle:
    name: str
    model: nn.Module
    mean: torch.Tensor
    scale: torch.Tensor
    logit_offsets: dict[str, torch.Tensor]
    offline: dict[str, object]

    def score(self, features, semantic, history, chunk=20_000):
        values = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(features), chunk):
                current = features[start : start + chunk]
                outputs = self.model(
                    (current - self.mean) / self.scale,
                    semantic[start : start + chunk],
                    history[start : start + chunk],
                )
                task_value = (
                    0.15 * torch.sigmoid(
                        outputs["click"] - self.logit_offsets["click"]
                    )
                    + 0.25 * torch.sigmoid(
                        outputs["detail"] - self.logit_offsets["detail"]
                    )
                    + 0.15 * torch.sigmoid(
                        outputs["save"] - self.logit_offsets["save"]
                    )
                    + 0.45 * torch.sigmoid(
                        outputs["order"] - self.logit_offsets["order"]
                    )
                )
                guardrail = (
                    0.10 * current[:, :, 6] + 0.10 * current[:, :, 7]
                    - 0.15 * current[:, :, 9] - 0.10 * current[:, :, 4]
                )
                values.append(task_value + guardrail)
        return torch.cat(values)


@dataclass(frozen=True)
class XGBoostRankerBundle:
    name: str
    model: XGBRanker
    offline: dict[str, object]

    def score(self, features, semantic, history, chunk=200_000):
        del semantic, history
        shape = features.shape[:2]
        flat = features.flatten(0, 1)
        values = []
        for start in range(0, len(flat), chunk):
            batch = _xgb_array(flat[start : start + chunk])
            prediction = self.model.predict(batch)
            values.append(torch.from_dlpack(prediction) if features.is_cuda else (
                torch.from_numpy(prediction)
            ))
        score = torch.cat(values).to(features.device).reshape(shape)
        return score + (
            0.10 * features[:, :, 6] + 0.10 * features[:, :, 7]
            - 0.15 * features[:, :, 9] - 0.10 * features[:, :, 4]
        )


def _masked_task_loss(logits, labels, masks, propensity, positive_weight):
    losses = []
    inverse = propensity.reciprocal().clamp(max=5.0)
    for index, task in enumerate(LOCAL_SEARCH_TASKS):
        mask = masks[:, :, index]
        point = functional.binary_cross_entropy_with_logits(
            logits[task], labels[:, :, index],
            pos_weight=positive_weight[index], reduction="none",
        )
        point = (point * mask * inverse).sum() / (
            mask * inverse
        ).sum().clamp_min(1.0)
        positive = (labels[:, :, index] * mask).sum(1) > 0
        listwise = torch.zeros((), device=labels.device)
        if positive.any():
            listwise = -(
                labels[positive, :, index]
                * functional.log_softmax(logits[task][positive], dim=1)
            ).sum(1).mean()
        losses.append(point + 0.20 * listwise)
    weights = (0.20, 0.30, 0.15, 0.35)
    return sum(weight * loss for weight, loss in zip(weights, losses))


def _offline_tasks(model, features, semantic, history, labels, masks, mean, scale):
    predictions = {task: [] for task in LOCAL_SEARCH_TASKS}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), 20_000):
            outputs = model(
                (features[start : start + 20_000] - mean) / scale,
                semantic[start : start + 20_000], history[start : start + 20_000],
            )
            for task in LOCAL_SEARCH_TASKS:
                predictions[task].append(torch.sigmoid(outputs[task]).cpu())
    report = {}
    for index, task in enumerate(LOCAL_SEARCH_TASKS):
        mask = masks[:, :, index].bool().cpu().numpy().reshape(-1)
        target = labels[:, :, index].cpu().numpy().reshape(-1)[mask]
        score = torch.cat(predictions[task]).numpy().reshape(-1)[mask]
        report[task] = {
            "auc": float(roc_auc_score(target, score)),
            "pr_auc": float(average_precision_score(target, score)),
            "positive_rate": float(target.mean()),
            "observable_rows": int(mask.sum()),
        }
    return report


def _train_torch_rankers(config, features, semantic, history, examples):
    top = examples.exposed_indices
    exposed_features = gather_candidates(features, top)
    exposed_semantic = gather_candidates(semantic, top)
    labels = gather_candidates(examples.labels, top)
    masks = gather_candidates(examples.label_masks, top)
    first, second = int(config.requests * 0.70), int(config.requests * 0.85)
    mean = exposed_features[:first].flatten(0, 1).mean(0)
    scale = exposed_features[:first].flatten(0, 1).std(0).clamp_min(1e-4)
    flat_labels = labels[:first].flatten(0, 1)
    flat_masks = masks[:first].flatten(0, 1)
    positive = (flat_labels * flat_masks).sum(0).clamp_min(1.0)
    negative = (flat_masks.sum(0) - positive).clamp_min(1.0)
    positive_weight = (negative / positive).clamp(max=30.0)
    bundles = {}
    for offset, (name, factory) in enumerate(MODEL_FACTORIES.items()):
        torch.manual_seed(config.seed + 800 + offset)
        model = factory(features.shape[2], config.semantic_dim).to(features.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=1e-5
        )
        generator = torch.Generator(device=features.device).manual_seed(
            config.seed + 810 + offset
        )
        losses = []
        for _ in range(config.train_epochs):
            order = torch.randperm(first, generator=generator, device=features.device)
            for start in range(0, first, config.train_batch_requests):
                request = order[start : start + config.train_batch_requests]
                outputs = model(
                    (exposed_features[request] - mean) / scale,
                    exposed_semantic[request], history[request],
                )
                loss = _masked_task_loss(
                    outputs, labels[request], masks[request],
                    examples.position_propensity[request], positive_weight,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
        offsets = {
            task: positive_weight[index].log()
            for index, task in enumerate(LOCAL_SEARCH_TASKS)
        }
        offline = {
            "parameters": sum(value.numel() for value in model.parameters()),
            "final_loss": float(np.mean(losses[-max(first // config.train_batch_requests, 1):])),
            "weighted_loss_logit_offsets": {
                task: float(value) for task, value in offsets.items()
            },
            "metrics": _offline_tasks(
                model, exposed_features[first:second], exposed_semantic[first:second],
                history[first:second], labels[first:second], masks[first:second],
                mean, scale,
            ),
        }
        bundles[name] = TorchRankerBundle(
            name, model, mean, scale, offsets, offline
        )
    return bundles


def _train_xgboost(config, features, examples):
    top = examples.exposed_indices
    exposed = gather_candidates(features, top)
    labels = gather_candidates(examples.labels, top)
    first, second = int(config.requests * 0.70), int(config.requests * 0.85)
    grade = (
        labels[:, :, 0] + 2 * labels[:, :, 1]
        + 2 * labels[:, :, 2] + 4 * labels[:, :, 3]
    )
    train_x = _xgb_array(exposed[:first].flatten(0, 1))
    train_y = _xgb_array(grade[:first].flatten())
    qid_tensor = torch.arange(first, device=features.device).repeat_interleave(
        config.exposed_candidates
    )
    qid = _xgb_array(qid_tensor)
    model = XGBRanker(
        objective="rank:pairwise", n_estimators=90, max_depth=6,
        learning_rate=0.08, subsample=0.85, colsample_bytree=0.85,
        tree_method="hist", device="cuda" if config.device.startswith("cuda") else "cpu",
        random_state=config.seed + 850, n_jobs=8,
    )
    model.fit(train_x, train_y, qid=qid, verbose=False)
    validation_x = _xgb_array(exposed[first:second].flatten(0, 1))
    score = _numpy(model.predict(validation_x)).reshape(
        -1, config.exposed_candidates
    )
    target = grade[first:second].cpu().numpy()
    binary = (target > 0).reshape(-1)
    return XGBoostRankerBundle("xgboost_pairwise", model, {
        "parameters": int(sum(tree.count("leaf") for tree in model.get_booster().get_dump())),
        "ranking_auc": float(roc_auc_score(binary, score.reshape(-1))),
        "ndcg_at_8": float(ndcg_score(target, score, k=config.exposed_candidates)),
        "positive_rate": float(binary.mean()),
    })


def train_rankers(config, world, candidates, features, response):
    semantic = world.catalog.semantic[candidates.poi_ids]
    bundles = _train_torch_rankers(
        config, features, semantic, world.requests.history_sequence,
        response["examples"],
    )
    bundles["xgboost_pairwise"] = _train_xgboost(
        config, features, response["examples"]
    )
    return bundles


def save_ranker(bundle, path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(bundle, XGBoostRankerBundle):
        bundle.model.save_model(path)
    else:
        torch.save({
            "schema": "local-search-ranker-v1", "name": bundle.name,
            "feature_width": len(bundle.mean), "semantic_dim": config.semantic_dim,
            "state_dict": bundle.model.state_dict(), "mean": bundle.mean,
            "scale": bundle.scale, "logit_offsets": bundle.logit_offsets,
            "offline": bundle.offline,
        }, path)
    return {"artifact_file": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}


def load_ranker(path, device="cpu"):
    if path.suffix == ".json":
        target_device = "cuda" if str(device).startswith("cuda") else "cpu"
        model = XGBRanker(device=target_device)
        model.load_model(path)
        model.set_params(device=target_device)
        model.get_booster().set_param({"device": target_device})
        return XGBoostRankerBundle("xgboost_pairwise", model, {})
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "local-search-ranker-v1":
        raise ValueError("unsupported Local Search ranker artifact")
    model = MODEL_FACTORIES[payload["name"]](
        payload["feature_width"], payload["semantic_dim"]
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return TorchRankerBundle(
        payload["name"], model, payload["mean"].to(device),
        payload["scale"].to(device),
        {name: value.to(device) for name, value in payload["logit_offsets"].items()},
        payload["offline"],
    )
