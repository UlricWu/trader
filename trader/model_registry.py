#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : model_registry.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/8/22 18:11

#!filepath: trader/model_registry.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

@dataclass
class ModelConfig:
    name: str                      # "logistic_regression" | "random_forest" | ...
    params: Dict[str, Any]         # estimator-specific kwargs

def build_estimator(cfg: ModelConfig) -> ClassifierMixin:
    model_name = cfg.name.lower()
    params = cfg.params or {}

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            random_state=params.get("random_state", 42),
        )
    elif model_name == "logistic_regression":
        return LogisticRegression(
            solver=params.get("solver", "lbfgs"),
            max_iter=params.get("max_iter", 200),
            random_state=params.get("random_state", 42),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

#     p = cfg.params or {}
#     if name in {"logreg", "logistic", "logistic_regression"}:
#         # Scale → LR with probabilities
#         return Pipeline([
#             ("scaler", StandardScaler(with_mean=False)),  # safe for sparse/robust
#             ("clf", LogisticRegression(max_iter=p.get("max_iter", 1000),
#                                        class_weight=p.get("class_weight", "balanced"),
#                                        solver=p.get("solver", "lbfgs"),
#                                        C=p.get("C", 1.0)))
#         ])
#     if name in {"rf", "random_forest"}:
#         return RandomForestClassifier(
#             n_estimators=p.get("n_estimators", 200),
#             max_depth=p.get("max_depth", 6),
#             min_samples_leaf=p.get("min_samples_leaf", 2),
#             random_state=p.get("random_state", 42),
#             n_jobs=p.get("n_jobs", -1),
#         )
#     raise ValueError(f"Unknown model name: {cfg.name}")

