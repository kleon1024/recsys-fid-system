"""Adapter from the synthetic Feed catalog to the KuaiRand sequence kernel."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from math import pi
from pathlib import Path

import torch

from ....world_model.external.kuairand.contracts import HASH_VOCABULARIES
from ....world_model.external.kuairand.evaluation.policy import calibrate_response
from ....world_model.external.kuairand.kernel import KuaiBehaviorKernel
from ...graph.random import uniform
from ..ranking_sequence import initialize_ranking_sequence
from .mixture import HiddenResponseMixture


class ExternalSequenceMixtureWorld:
    """External sequence evidence plus hidden multi-population response mechanisms."""

    def __init__(
        self,
        artifact: Path,
        calibration_report: Path,
        dataset_dir: Path,
        device: str,
        seed: int,
        inference_batch: int = 4_096,
        calibration_key: str = "sequence_randomized_adapter",
    ) -> None:
        self.device = torch.device(device)
        report = json.loads(calibration_report.read_text())
        artifact_hash = sha256(artifact.read_bytes()).hexdigest()
        if report.get("artifact_sha256") != artifact_hash:
            raise ValueError("Feed behavior artifact and calibration lineage differ")
        dataset_manifest = json.loads((dataset_dir / "manifest.json").read_text())
        catalog_path = dataset_dir / "random_item_catalog.pt"
        catalog_hash = sha256(catalog_path.read_bytes()).hexdigest()
        if dataset_manifest.get("catalog_sha256") != catalog_hash:
            raise ValueError("Feed behavior catalog does not match its manifest")
        random_path = dataset_dir / "random_test.pt"
        random_hash = sha256(random_path.read_bytes()).hexdigest()
        if dataset_manifest["splits"]["random_test"]["sha256"] != random_hash:
            raise ValueError("Feed behavior profiles do not match their manifest")
        self.kernel = KuaiBehaviorKernel.load(artifact, device)
        self.calibration = report["randomized_calibration"][calibration_key]
        self.external_catalog = torch.load(
            catalog_path, map_location="cpu", weights_only=False
        )
        self.external_profiles = torch.load(
            random_path, map_location="cpu", weights_only=False
        )
        self.residual = HiddenResponseMixture(self.device, seed)
        self.inference_batch = inference_batch
        self.artifact_sha256 = artifact_hash
        self.catalog_sha256 = catalog_hash
        self.profile_sha256 = random_hash
        self._build_stay_sampler()

    @torch.inference_mode()
    def _build_stay_sampler(self, bins=16, quantiles=65):
        """Fit a conditional residual bootstrap from randomized holdout data."""
        predicted = []
        profiles = self.external_profiles
        for start in range(0, len(profiles["sparse"]), self.inference_batch):
            stop = min(start + self.inference_batch, len(profiles["sparse"]))
            logits = self.kernel.model(
                profiles["sparse"][start:stop].to(self.device),
                profiles["dense"][start:stop].to(self.device),
                profiles["history_items"][start:stop].to(self.device),
                profiles["history_feedback"][start:stop].to(self.device),
            )
            predicted.append(torch.sigmoid(logits[:, 7]).cpu())
        prediction = torch.cat(predicted)
        labels = profiles["labels"].float()
        played = labels[:, 0] > 0
        if played.sum() < bins * 32:
            raise ValueError("randomized holdout cannot identify stay residuals")
        prediction = prediction[played]
        observed = labels[played, 7]
        probability = torch.linspace(0.0, 1.0, bins + 1)
        edges = torch.quantile(prediction, probability)
        grid = torch.linspace(0.0, 1.0, quantiles)
        global_values = torch.quantile(observed, grid)
        tables = []
        bucket = torch.bucketize(prediction, edges[1:-1])
        for index in range(bins):
            values = observed[bucket == index]
            tables.append(
                torch.quantile(values, grid) if len(values) >= 32 else global_values
            )
        self.stay_sampler_edges = edges.to(self.device)
        self.stay_sampler_quantiles = torch.stack(tables).to(self.device)
        self.source_completion_rate = float((observed >= 0.95).float().mean())

    def describe(self):
        return {
            "authority": "external-sequence-mixture-v4",
            "artifact_sha256": self.artifact_sha256,
            "catalog_sha256": self.catalog_sha256,
            "profile_sha256": self.profile_sha256,
            "sequence_length": int(
                self.external_profiles["history_items"].shape[1]
            ),
            "hidden_mixture_experts": 4,
            "stay_sampler": {
                "authority": "randomized-holdout-conditional-residual-bootstrap",
                "bins": len(self.stay_sampler_quantiles),
                "quantiles": self.stay_sampler_quantiles.shape[1],
                "source_completion_rate_given_play": self.source_completion_rate,
            },
        }

    def initialize_state(self, state):
        rows = len(self.external_profiles["user_ids"])
        profile = torch.remainder(
            state["user_ids"].cpu() * 1_103_515_245 + 12_345, rows
        ).long()
        state["behavior_history_items"] = self.external_profiles[
            "history_items"
        ][profile].to(self.device).long()
        state["behavior_history_feedback"] = self.external_profiles[
            "history_feedback"
        ][profile].to(self.device).to(torch.uint8)
        state["behavior_user_sparse"] = self.external_profiles[
            "sparse"
        ][profile].to(self.device).long()
        state["behavior_user_dense"] = self.external_profiles[
            "dense"
        ][profile].to(self.device).float()
        history_topic = self.video_topic_lookup[
            state["behavior_history_items"]
        ]
        feedback = state["behavior_history_feedback"].float()
        weight = (
            (state["behavior_history_items"] > 0).float()
            * (0.20 + feedback[:, :, 0] + 1.5 * feedback[:, :, 1]
               + 0.8 * feedback[:, :, 2] - 0.8 * feedback[:, :, 6])
        ).clamp_min(0.0)
        profile_interest = torch.zeros_like(state["interest"])
        profile_interest.scatter_add_(1, history_topic, weight)
        profile_interest = torch.nn.functional.normalize(
            profile_interest + 0.05, dim=1
        )
        state["interest"] = torch.nn.functional.normalize(
            0.75 * profile_interest + 0.25 * state["interest"], dim=1
        )
        state["observed_interest"] = torch.nn.functional.normalize(
            0.80 * profile_interest + 0.20 * state["observed_interest"], dim=1
        )
        state["local_observed_interest"] = torch.nn.functional.normalize(
            0.70 * profile_interest + 0.30 * state["local_observed_interest"],
            dim=1,
        )
        initialize_ranking_sequence(state, self.video_topic_lookup)

    def attach_catalog(self, catalog, config):
        eligible = torch.nonzero(
            self.external_catalog["standard_exposure_count"] >= 5
        ).flatten()
        if not len(eligible):
            raise ValueError("external Feed catalog has no supported items")
        item = torch.arange(catalog.size)
        profile = eligible[torch.remainder(
            item * 1_103_515_245 + 12_345, len(eligible)
        )]
        sparse = self.external_catalog["sparse"][profile].to(self.device)
        dense = self.external_catalog["dense"][profile].to(self.device)
        category = torch.remainder(sparse[:, 3], config.topics)
        self.video_topic_lookup = torch.zeros(
            HASH_VOCABULARIES[1], device=self.device, dtype=torch.long
        )
        self.video_topic_lookup[sparse[:, 1]] = category
        self.topic_profile_pools = tuple(
            torch.nonzero(category == topic).flatten()
            for topic in range(config.topics)
        )
        basis = catalog.topics[: config.topics]
        topics = torch.nn.functional.normalize(
            basis[category] + 0.15 * catalog.topics, dim=1
        )
        duration = torch.expm1(
            dense[:, 0] * torch.log(torch.tensor(300_001.0, device=self.device))
        ).div(1_000.0).clamp(1.0, 180.0)
        standard_count = self.external_catalog["standard_exposure_count"][
            profile
        ].to(self.device).float()
        popularity = torch.log1p(standard_count)
        popularity /= popularity.max().clamp_min(1.0)
        content_type = torch.where(
            sparse[:, 4] == 1,
            torch.full_like(catalog.content_type, 2),
            torch.where(
                dense[:, 9] > 0.5,
                torch.ones_like(catalog.content_type),
                torch.zeros_like(catalog.content_type),
            ),
        )
        return replace(
            catalog,
            topics=topics,
            category=category,
            popularity=popularity,
            duration_seconds=duration,
            content_type=content_type,
            behavior_sparse=sparse,
            behavior_dense=dense,
        )

    def decorate_new_supply(self, catalog, slots, categories, day):
        behavior_sparse = catalog.behavior_sparse.clone()
        behavior_dense = catalog.behavior_dense.clone()
        for topic, pool in enumerate(self.topic_profile_pools):
            mask = categories == topic
            if not mask.any():
                continue
            target = slots[mask]
            choice = torch.remainder(target * 48_271 + day * 7_919, len(pool))
            source = pool[choice]
            behavior_sparse[target] = catalog.behavior_sparse[source]
            behavior_dense[target] = catalog.behavior_dense[source]
        duration = catalog.duration_seconds.clone()
        duration[slots] = torch.expm1(
            behavior_dense[slots, 0]
            * torch.log(torch.tensor(300_001.0, device=self.device))
        ).div(1_000.0).clamp(1.0, 180.0)
        return replace(
            catalog, behavior_sparse=behavior_sparse,
            behavior_dense=behavior_dense, duration_seconds=duration,
        )

    @staticmethod
    def _sparse(state, selected):
        if "behavior_sparse" in selected:
            sparse = selected["behavior_sparse"].clone()
            sparse[:, 0] = state["behavior_user_sparse"][:, 0]
            return sparse
        user = torch.remainder(state["user_ids"], HASH_VOCABULARIES[0] - 1) + 1
        item = torch.remainder(selected["item_ids"], HASH_VOCABULARIES[1] - 1) + 1
        author = torch.remainder(selected["author"], HASH_VOCABULARIES[2] - 1) + 1
        tag = torch.remainder(selected["candidate_topic"], HASH_VOCABULARIES[3] - 1) + 1
        video_type = torch.where(
            selected["content_type"] == 2,
            torch.ones_like(item),
            torch.full_like(item, 2),
        )
        upload = torch.remainder(item * 17 + author * 3, HASH_VOCABULARIES[5] - 1) + 1
        music = torch.remainder(item * 29 + tag * 5, HASH_VOCABULARIES[6] - 1) + 1
        return torch.stack((user, item, author, tag, video_type, upload, music), dim=1)

    @staticmethod
    def _dense(state, selected, step):
        hour = torch.remainder(state["region_bucket"] * 2 + step, 24).float()
        duration = torch.log1p(selected["duration"]) / torch.log(
            torch.tensor(300_001.0, device=hour.device)
        )
        activity = state["historical_activity"]
        age = state["account_age_days"]
        generated = torch.stack((
            duration,
            torch.sin(2.0 * pi * hour / 24.0),
            torch.cos(2.0 * pi * hour / 24.0),
            0.25 + 0.75 * torch.remainder(selected["item_ids"], 97).float() / 96.0,
            torch.log1p(age) / torch.log(torch.tensor(4_001.0, device=hour.device)),
            torch.log1p(activity * 25.0) / torch.log(torch.tensor(10_001.0, device=hour.device)),
            torch.log1p(activity * 18.0) / torch.log(torch.tensor(10_001.0, device=hour.device)),
            torch.log1p(activity * 3.0) / torch.log(torch.tensor(1_001.0, device=hour.device)),
            (state["lifecycle_bucket"] <= 1).float(),
            (selected["content_type"] == 1).float(),
            (state["historical_activity"] > 25).float(),
        ), dim=1)
        if "behavior_dense" not in selected:
            return generated
        dense = selected["behavior_dense"].clone()
        dense[:, 0:3] = generated[:, 0:3]
        dense[:, 4:] = state["behavior_user_dense"][:, 4:]
        return dense

    @staticmethod
    def _hidden_inputs(state, selected, step):
        affinity = (selected["topics"] * state["interest"]).sum(dim=1)
        topic_rate = state["topic_counts"].gather(
            1, selected["candidate_topic"][:, None]
        ).squeeze(1) / state["topic_counts"].sum(dim=1).clamp_min(1.0)
        author_head = (selected["author"] < 128).float()
        return torch.stack((
            affinity,
            selected["quality"],
            selected["freshness"],
            selected["popularity"],
            state["hidden_satisfaction"],
            state["hidden_fatigue"],
            state["lifecycle_bucket"].float() / 3.0,
            state["region_bucket"].float() / 9.0,
            torch.full_like(affinity, step / 64.0),
            topic_rate,
            author_head,
            state["hidden_novelty"],
            state["hidden_patience"],
            selected["content_type"].float() / 2.0,
        ), dim=1)

    @staticmethod
    def _candidate_hidden_inputs(state, candidates, step):
        affinity = torch.einsum(
            "bkd,bd->bk", candidates["topics"], state["interest"]
        )
        topic_rate = state["topic_counts"].gather(
            1, candidates["candidate_topic"]
        ) / state["topic_counts"].sum(dim=1, keepdim=True).clamp_min(1.0)
        users, count = affinity.shape

        def repeated(value):
            return value[:, None].expand(users, count)

        return torch.stack((
            affinity, candidates["quality"], candidates["freshness"],
            candidates["popularity"], repeated(state["hidden_satisfaction"]),
            repeated(state["hidden_fatigue"]),
            repeated(state["lifecycle_bucket"].float() / 3.0),
            repeated(state["region_bucket"].float() / 9.0),
            torch.full_like(affinity, step / 64.0), topic_rate,
            (candidates["author"] < 128).float(),
            repeated(state["hidden_novelty"]),
            repeated(state["hidden_patience"]),
            candidates["content_type"].float() / 2.0,
        ), dim=2)

    @staticmethod
    def _apply_residual(probability, stay, residual):
        probability = torch.sigmoid(
            torch.logit(probability.clamp(1e-5, 1 - 1e-5)) + residual[..., :7]
        )
        stay = torch.sigmoid(
            torch.logit(stay.clamp(1e-5, 1 - 1e-5)) + residual[..., 7]
        )
        return probability, stay

    def _predict(self, state, selected, step):
        sparse = self._sparse(state, selected)
        dense = self._dense(state, selected, step)
        probabilities, stay = [], []
        for start in range(0, len(sparse), self.inference_batch):
            stop = min(start + self.inference_batch, len(sparse))
            response = self.kernel.score_slate(
                sparse[start:stop], dense[start:stop],
                sparse[start:stop, None], dense[start:stop, None],
                state["behavior_history_items"][start:stop],
                state["behavior_history_feedback"][start:stop],
            )
            calibrated = calibrate_response(response, self.calibration)
            probabilities.append(calibrated.probabilities[:, 0])
            stay.append(calibrated.stay_norm[:, 0])
        probability = torch.cat(probabilities)
        stay_norm = torch.cat(stay)
        residual = self.residual(
            self._hidden_inputs(state, selected, step), state["hidden_mixture"]
        )
        probability, stay_norm = self._apply_residual(
            probability, stay_norm, residual
        )
        return probability, stay_norm, sparse[:, 1]

    def predict(self, state, selected, step):
        return self._predict(state, selected, step)

    @torch.inference_mode()
    def score_candidates(self, state, candidates, step):
        probabilities, stays = [], []
        for start in range(0, len(state["user_ids"]), self.inference_batch):
            stop = min(start + self.inference_batch, len(state["user_ids"]))
            sparse = candidates["behavior_sparse"][start:stop].clone()
            dense = candidates["behavior_dense"][start:stop].clone()
            request_sparse = state["behavior_user_sparse"][start:stop]
            request_dense = state["behavior_user_dense"][start:stop].clone()
            hour = torch.remainder(
                state["region_bucket"][start:stop] * 2 + step, 24
            ).float()
            request_dense[:, 1] = torch.sin(2.0 * pi * hour / 24.0)
            request_dense[:, 2] = torch.cos(2.0 * pi * hour / 24.0)
            response = self.kernel.score_slate(
                request_sparse, request_dense, sparse, dense,
                state["behavior_history_items"][start:stop],
                state["behavior_history_feedback"][start:stop],
            )
            calibrated = calibrate_response(response, self.calibration)
            probabilities.append(calibrated.probabilities)
            stays.append(calibrated.stay_norm)
        probability, stay = torch.cat(probabilities), torch.cat(stays)
        hidden = self._candidate_hidden_inputs(state, candidates, step)
        residual = self.residual(
            hidden.flatten(0, 1),
            state["hidden_mixture"][:, None].expand(hidden.shape[:2]).reshape(-1),
        ).reshape(*hidden.shape[:2], 8)
        probability, stay = self._apply_residual(probability, stay, residual)
        utility = (
            0.20 * probability[:, :, 0]
            + 0.30 * probability[:, :, 1]
            + 0.10 * probability[:, :, 2]
            - 0.10 * probability[:, :, 6]
            + 0.40 * stay
        )
        return {"probabilities": probability, "stay_norm": stay, "utility": utility}

    @torch.inference_mode()
    def sample(self, config, state, selected, step):
        probability, stay_norm, item_ids = self._predict(state, selected, step)
        draws = torch.stack(tuple(
            uniform(state["user_ids"], step, 230 + index, config.seed)
            for index in range(7)
        ), dim=1)
        played = (draws[:, 0] < probability[:, 0]) & state["active"]
        conditional = (
            probability[:, 1:6] / probability[:, :1].clamp_min(1e-5)
        ).clamp_max(1.0)
        downstream = (draws[:, 1:6] < conditional) & played[:, None]
        stay_bucket = torch.bucketize(stay_norm, self.stay_sampler_edges[1:-1])
        stay_quantile = torch.floor(
            uniform(state["user_ids"], step, 238, config.seed)
            * self.stay_sampler_quantiles.shape[1]
        ).long().clamp_max(self.stay_sampler_quantiles.shape[1] - 1)
        sampled_stay_norm = self.stay_sampler_quantiles[
            stay_bucket, stay_quantile
        ]
        stay = torch.expm1(
            sampled_stay_norm * torch.log1p(selected["duration"])
        ).clamp_min(0.0)
        stay = torch.minimum(stay, selected["duration"]) * played
        long_view, like, comment, share, follow = downstream.unbind(dim=1)
        negative = (draws[:, 6] < probability[:, 6]) & state["active"]
        feedback = torch.stack(
            (played, long_view, like, comment, share, follow, negative), dim=1
        )
        return {
            "stay": stay,
            "played": played,
            "play_draw": played,
            "long_view": long_view,
            "quality_view": long_view & (stay >= torch.minimum(
                torch.full_like(stay, 30.0), selected["duration"]
            )),
            "like": like,
            "comment": comment,
            "share": share,
            "follow": follow,
            "negative": negative,
            "probabilities": probability,
            "history_item": item_ids,
            "history_feedback": feedback,
        }
