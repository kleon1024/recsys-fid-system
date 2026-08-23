"""Train every V3 ladder model on one propensity-corrected snapshot."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from ...models.artifact import publish_policy
from ...models.deep_policy import FeedDeepPolicy
from ...models.feed_multitask import FeedMultiTaskPolicy
from ...models.multitask_policy import FeedMMoEPolicy
from ....simulation.policies import fit_logistic_policy, fit_xgboost_policy
from .data import ExposedSplit


LONG_VIEW = 5
QUALITY_VIEW = 6
NEGATIVE = 8


def _deep_models(train, validation, epochs, device, seed):
    rng = np.random.default_rng(seed)
    probability = train.weights / train.weights.sum()
    indices = rng.choice(len(train.features), len(train.features), p=probability)
    models = []
    timing = {}
    for name in ("wide_deep", "deepfm", "dcnv2"):
        model = FeedDeepPolicy(name, device, seed)
        started = perf_counter()
        model.fit(
            train.features[indices], train.labels[indices, LONG_VIEW],
            validation.features, validation.labels[:, LONG_VIEW], epochs,
        )
        timing[name] = perf_counter() - started
        models.append(model)
    mmoe = FeedMMoEPolicy(train.features.shape[1], device, seed)
    task_labels = train.labels[:, (LONG_VIEW, QUALITY_VIEW, NEGATIVE)]
    validation_labels = validation.labels[:, (LONG_VIEW, QUALITY_VIEW, NEGATIVE)]
    started = perf_counter()
    mmoe.fit(
        train.features[indices], task_labels[indices],
        validation.features, validation_labels, epochs,
    )
    timing[mmoe.name] = perf_counter() - started
    multitask = FeedMultiTaskPolicy(train.features.shape[1], device, seed)
    started = perf_counter()
    multitask.fit(
        train.features[indices], train.labels[indices],
        validation.features, validation.labels, epochs,
    )
    timing[multitask.name] = perf_counter() - started
    return (*models, mmoe, multitask), timing


def train_models(train, validation, epochs, device, seed):
    started = perf_counter()
    lr = fit_logistic_policy(
        "lr_v3_long_view", train.features, train.labels[:, LONG_VIEW],
        tuple(range(train.features.shape[1])), seed, train.weights,
    )
    timing = {lr.name: perf_counter() - started}
    started = perf_counter()
    xgboost = fit_xgboost_policy(
        train.features, train.labels[:, LONG_VIEW], seed, train.weights,
        "xgboost_v3_long_view",
    )
    timing[xgboost.name] = perf_counter() - started
    deep, deep_timing = _deep_models(train, validation, epochs, device, seed)
    timing.update(deep_timing)
    return (lr, xgboost, *deep), timing


def publish_models(models, audit_features, artifact_dir, signal_version):
    published = {}
    replay = {}
    for model in models:
        serving, delta = publish_policy(
            model, audit_features, signal_version, artifact_dir
        )
        published[model.name] = serving
        replay[model.name] = delta
    return published, replay
