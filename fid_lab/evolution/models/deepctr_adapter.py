"""Thin data and model adapter for the mature DeepCTR-Torch model zoo."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import io

import numpy as np
import torch
from deepctr_torch.callbacks import EarlyStopping
from deepctr_torch.inputs import DenseFeat, SparseFeat, VarLenSparseFeat
from deepctr_torch.models import DCNMix, DIN, DeepFM, MMOE, PLE, WDL


MODEL_BUILDERS = {
    "wide_deep": WDL,
    "deepfm": DeepFM,
    "dcnv2": DCNMix,
    "din": DIN,
    "mmoe": MMOE,
    "ple": PLE,
}


def supported_deepctr_models() -> tuple[str, ...]:
    return tuple(MODEL_BUILDERS)


@dataclass(frozen=True)
class DeepCTRFeatureBundle:
    linear_columns: tuple[object, ...]
    deep_columns: tuple[object, ...]
    inputs: dict[str, np.ndarray]


def build_feature_bundle(
    sparse: np.ndarray,
    dense: np.ndarray,
    bucket_size: int = 1_024,
) -> DeepCTRFeatureBundle:
    sparse_columns = tuple(
        SparseFeat(f"fid_{index}", vocabulary_size=bucket_size, embedding_dim=8)
        for index in range(sparse.shape[1])
    )
    dense_columns = tuple(DenseFeat(f"dense_{index}", 1) for index in range(dense.shape[1]))
    inputs = {
        **{
            f"fid_{index}": sparse[:, index].astype(np.int64) % bucket_size
            for index in range(sparse.shape[1])
        },
        **{
            f"dense_{index}": dense[:, index].astype(np.float32)
            for index in range(dense.shape[1])
        },
    }
    columns = sparse_columns + dense_columns
    return DeepCTRFeatureBundle(columns, columns, inputs)


def build_din_bundle(
    sparse: np.ndarray,
    dense: np.ndarray,
    history_item_ids: np.ndarray,
    sequence_mask: np.ndarray,
    bucket_size: int = 1_024,
) -> DeepCTRFeatureBundle:
    item = SparseFeat("item_id", bucket_size, embedding_dim=8)
    columns = (
        SparseFeat("viewer_id", bucket_size, embedding_dim=8),
        item,
        SparseFeat("category_id", 128, embedding_dim=8),
        *(DenseFeat(f"dense_{index}", 1) for index in range(dense.shape[1])),
        VarLenSparseFeat(
            SparseFeat("hist_item_id", bucket_size, embedding_dim=8, embedding_name="item_id"),
            maxlen=history_item_ids.shape[1],
            length_name="sequence_length",
        ),
    )
    inputs = {
        "viewer_id": sparse[:, 0].astype(np.int64) % bucket_size,
        "item_id": sparse[:, 2].astype(np.int64) % bucket_size,
        "category_id": sparse[:, 4].astype(np.int64) % 128,
        "hist_item_id": history_item_ids.astype(np.int64) % bucket_size,
        "sequence_length": sequence_mask.sum(axis=1).astype(np.int64),
        **{
            f"dense_{index}": dense[:, index].astype(np.float32)
            for index in range(dense.shape[1])
        },
    }
    return DeepCTRFeatureBundle((), columns, inputs)


def _create_model(
    name: str,
    bundle: DeepCTRFeatureBundle,
    tasks: tuple[str, ...],
    device: str,
    seed: int,
):
    if name == "din":
        return DIN(
            list(bundle.deep_columns),
            history_feature_list=["item_id"],
            device=device,
            seed=seed,
            l2_reg_embedding=1e-4,
            l2_reg_dnn=1e-4,
            dnn_dropout=0.15,
        )
    if name in {"mmoe", "ple"}:
        if len(tasks) < 2:
            raise ValueError("multi-task model requires at least two tasks")
        return MODEL_BUILDERS[name](
            list(bundle.deep_columns),
            task_types=tuple("binary" for _ in tasks),
            task_names=tasks,
            device=device,
            seed=seed,
            l2_reg_embedding=1e-4,
            l2_reg_dnn=1e-4,
            dnn_dropout=0.15,
        )
    return MODEL_BUILDERS[name](
        bundle.linear_columns,
        bundle.deep_columns,
        device=device,
        seed=seed,
        l2_reg_embedding=1e-4,
        l2_reg_dnn=1e-4,
        dnn_dropout=0.15,
    )


class DeepCTRModelAdapter:
    def __init__(
        self,
        name: str,
        bundle: DeepCTRFeatureBundle,
        tasks: tuple[str, ...] = ("target",),
        device: str = "cpu",
        seed: int = 20260823,
    ) -> None:
        if name not in MODEL_BUILDERS:
            raise ValueError(f"unsupported DeepCTR model: {name}")
        with redirect_stdout(io.StringIO()):
            self.model = _create_model(name, bundle, tasks, device, seed)
        self.name = name
        self.tasks = tasks
        self.loss_history: list[float] = []
        self.model.compile("adam", "binary_crossentropy", metrics=["auc"])

    def fit(
        self,
        inputs: dict[str, np.ndarray],
        labels: np.ndarray,
        batch_size: int = 1_024,
        epochs: int = 1,
        validation: tuple[dict[str, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        callbacks = []
        if validation is not None:
            callbacks.append(
                EarlyStopping(
                    monitor="val_auc",
                    patience=2,
                    mode="max",
                    restore_best_weights=True,
                )
            )
        with redirect_stdout(io.StringIO()):
            history = self.model.fit(
                inputs,
                labels,
                batch_size=batch_size,
                epochs=epochs,
                verbose=0,
                validation_data=validation,
                callbacks=callbacks,
            )
        self.loss_history.extend(float(value) for value in history.history.get("loss", ()))

    def predict(
        self,
        inputs: dict[str, np.ndarray],
        batch_size: int = 4_096,
    ) -> np.ndarray:
        with redirect_stdout(io.StringIO()):
            values = self.model.predict(inputs, batch_size=batch_size)
        return np.asarray(values).reshape(len(next(iter(inputs.values()))), -1)

    def fit_distilled(
        self,
        inputs: dict[str, np.ndarray],
        labels: np.ndarray,
        teacher_logits: np.ndarray,
        teacher_rank: np.ndarray,
        epochs: int = 1,
        batch_size: int = 1_024,
    ) -> None:
        if self.name != "dcnv2":
            raise ValueError("distillation is owned by the DCNv2 adapter")
        matrix = np.concatenate(
            [inputs[name].reshape(-1, 1) for name in self.model.feature_index], axis=1
        ).astype(np.float32)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        device = next(self.model.parameters()).device
        for _ in range(epochs):
            for start in range(0, len(matrix), batch_size):
                stop = min(start + batch_size, len(matrix))
                values = torch.from_numpy(matrix[start:stop]).to(device)
                target = torch.from_numpy(labels[start:stop].astype(np.float32)).to(device)
                teacher = torch.from_numpy(teacher_logits[start:stop].astype(np.float32)).to(device)
                ranks = torch.from_numpy(teacher_rank[start:stop]).to(device)
                student = self.model(values).squeeze(1)
                hard = torch.nn.functional.binary_cross_entropy(student, target)
                soft = torch.nn.functional.binary_cross_entropy(student, torch.sigmoid(teacher))
                order = torch.argsort(ranks)
                count = max(len(order) // 4, 1)
                pairwise = torch.relu(0.05 - student[order[:count]].mean() + student[order[-count:]].mean())
                loss = 0.50 * hard + 0.30 * soft + 0.20 * pairwise
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss = float(loss.detach().cpu())
            self.loss_history.append(batch_loss)

    @property
    def parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())
