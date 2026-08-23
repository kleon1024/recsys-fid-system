"""DeepCTR-backed W&D, DeepFM, and DCNv2-style Feed policies."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory

from deepctr_torch.callbacks import EarlyStopping
from deepctr_torch.inputs import DenseFeat, SparseFeat
from deepctr_torch.models import DCNMix, DeepFM, WDL
import numpy as np
import torch


MODEL_BUILDERS = {"wide_deep": WDL, "deepfm": DeepFM, "dcnv2": DCNMix}
DENSE_INDICES = (*range(14), *range(18, 28))
SPARSE_SPECS = (
    ("item_id", 15, 4_096),
    ("author_id", 16, 1_024),
    ("category_id", 17, 12),
)


class FeedDeepPolicy:
    def __init__(
        self, name: str, device: str, seed: int = 20260823,
        task: str = "binary",
    ) -> None:
        if name not in MODEL_BUILDERS:
            raise ValueError(f"unsupported Feed deep model: {name}")
        self.name = name
        self.device = device
        self.seed = seed
        self.task = task
        sparse = tuple(
            SparseFeat(field, vocabulary, embedding_dim=8)
            for field, _, vocabulary in SPARSE_SPECS
        )
        dense = tuple(DenseFeat(f"dense_{index}", 1) for index in DENSE_INDICES)
        self.columns = (*sparse, *dense)
        self.model = MODEL_BUILDERS[name](
            self.columns,
            self.columns,
            device=device,
            seed=seed,
            dnn_hidden_units=(128, 64),
            l2_reg_embedding=1e-4,
            l2_reg_dnn=1e-4,
            dnn_dropout=0.15,
            task=task,
        )
        loss = "binary_crossentropy" if task == "binary" else "mse"
        metrics = ["auc"] if task == "binary" else []
        self.model.compile("adam", loss, metrics=metrics)
        self.loss_history: list[float] = []

    @staticmethod
    def inputs(features: np.ndarray) -> dict[str, np.ndarray]:
        inputs = {
            field: np.rint(features[:, index] * (vocabulary - 1)).astype(np.int64)
            for field, index, vocabulary in SPARSE_SPECS
        }
        inputs.update(
            {
                f"dense_{index}": features[:, index].astype(np.float32)
                for index in DENSE_INDICES
            }
        )
        return inputs

    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
        epochs: int,
    ) -> None:
        monitor = "val_auc" if self.task == "binary" else "val_loss"
        mode = "max" if self.task == "binary" else "min"
        callback = EarlyStopping(
            monitor=monitor, patience=2, mode=mode, restore_best_weights=True
        )
        with redirect_stdout(io.StringIO()):
            history = self.model.fit(
                self.inputs(train_features),
                train_labels,
                batch_size=1_024,
                epochs=epochs,
                verbose=0,
                validation_data=(self.inputs(validation_features), validation_labels),
                callbacks=[callback],
            )
        self.loss_history = [float(value) for value in history.history.get("loss", ())]

    def score(self, features: np.ndarray) -> np.ndarray:
        with redirect_stdout(io.StringIO()):
            return self.model.predict(
                self.inputs(features), batch_size=4_096
            ).reshape(-1)

    @property
    def parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def replay_delta(self, features: np.ndarray) -> float:
        before = self.score(features)
        with TemporaryDirectory() as directory:
            path = Path(directory) / f"{self.name}.pt"
            torch.save(self.model.state_dict(), path)
            replay = FeedDeepPolicy(self.name, self.device, self.seed, self.task)
            replay.model.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
            after = replay.score(features)
        return float(np.max(np.abs(before - after)))
