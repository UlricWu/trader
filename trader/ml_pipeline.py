#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : ml_pipeline.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/8/22 18:12
# !filepath: trader/ml_pipeline.py
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import deque
from typing import Deque, Dict, List, Tuple, Optional, Callable
from sklearn.base import ClassifierMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold
from trader.events import FeatureEvent, MLFeatureEvent
from utilts.logs import logs


# A tiny helper to log importance for trees and coefficients for linear models
def _log_feature_importance(prefix: str, estimator: ClassifierMixin, feature_names: List[str]) -> None:
    try:
        # Pipeline support: get the final step if wrapped
        model = estimator
        if hasattr(estimator, "named_steps"):
            # assume the last step is classifier
            model = list(estimator.named_steps.values())[-1]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            pairs = sorted(zip(feature_names, importances), key=lambda x: -x[1])
            logs.record_log(f"{prefix} feature_importances_:", 1)
            for k, v in pairs[:20]:
                logs.record_log(f"  {k}: {v:.4f}", 1)
        elif hasattr(model, "coef_"):
            coefs = np.ravel(model.coef_)
            pairs = sorted(zip(feature_names, np.abs(coefs)), key=lambda x: -x[1])
            logs.record_log(f"{prefix} |coef| (magnitude):", 1)
            for k, v in pairs[:20]:
                logs.record_log(f"  {k}: {v:.4f}", 1)
        else:
            logs.record_log(f"{prefix} model has no importances/coefficients.", 2)
    except Exception as exc:
        logs.record_log(f"{prefix} importance logging failed: {exc}", 3)


class MLPipeline:
    """
    Rolling ML pipeline (model-agnostic):
      - collects FeatureEvent rows
      - constructs rolling train set with next-bar target
      - periodic CV + training
      - logs CV metrics & feature importances
      - emits MLFeatureEvent (prediction + probability thresholding left to strategy)
    """

    def __init__(
            self,
            bus,
            model_builder: Callable[[], ClassifierMixin],
            train_window: int = 200,
            retrain_freq: int = 50,
            cv_folds: int = 3,
            min_train_rows: Optional[int] = None,
    ):
        self.bus = bus
        self.model_builder = model_builder
        self.train_window = train_window
        self.retrain_freq = retrain_freq
        self.cv_folds = cv_folds
        self.min_train_rows = 30
        # self.min_train_rows = min_train_rows or max(50, train_window // 2)

        self.rows: Dict[str, Deque[dict]] = {}
        self.estimator: Dict[str, ClassifierMixin] = {}
        self.counter: Dict[str, int] = {}

    # ---------- Event entry ----------
    def on_feature(self, event: FeatureEvent) -> None:
        # print(f"start ={event}")
        if event.is_empty():
            return
        s = event.symbol
        if s not in self.rows:
            self.rows[s] = deque(maxlen=self.train_window)
            self.counter[s] = 0
            self.estimator[s] = self.model_builder()

        # Append one row (features dict must include a "Close" for target construction)
        self.rows[s].append(dict(event.features))
        # self.counter[s] += 1

        # Retrain periodically
        # if self.counter[s] % self.retrain_freq == 0 and len(self.rows[s]) >= self.min_train_rows:
        if len(self.rows[s]) <= self.min_train_rows:
            return

        # self._train(s)

        X, y, feat_names = self._make_dataset(s)
        print(f"start training {len(X)}")
        if len(np.unique(y)) < 2 or len(y) < self.min_train_rows:
            # logs.record_log(f"[MLPipeline] {s} not enough class variety/rows for training.", 2)
            return

        # model = self.model_builder()  # fresh instance each train
        # print(model)
        model = self.estimator[s]
        print(f'model={model}')
        model.fit(X, y)

        # CV

        # cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        # try:
        #     acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        #     logs.record_log(f"[MLPipeline] {s} CV accuracy: {acc.mean():.3f} ± {acc.std():.3f}", 1)
        # except Exception as exc:
        #     logs.record_log(f"[MLPipeline] {s} CV failed: {exc}", 3)

        # Fit on all data
        self.estimator[s] = model


        # Predict latest (if trained, else HOLD)
        pred, proba = self._predict_latest(s)
        self.bus.emit(MLFeatureEvent(
            symbol=s,
            datetime=event.datetime,
            features=event.features,
            prediction=pred,
            probability=proba,
        ))

    # ---------- Internal: train / predict ----------
    def _make_dataset(self, s: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        df = pd.DataFrame(list(self.rows[s]))
        df = df.dropna()
        if "Close" not in df.columns:
            raise KeyError("FeatureEvent must include 'Close' to form targets.")
        # Binary target: next return up (1) vs down/flat (0)
        ret = df["Close"].pct_change().shift(-1)
        y = (ret > 0).astype(int).fillna(0)
        X = df.drop(columns=["Close"], errors="ignore")
        feat_names = list(X.columns)
        return X, y, feat_names

    def _train(self, s: str) -> None:
        try:
            X, y, feat_names = self._make_dataset(s)
            print(f"start training {len(X)}")
            if len(np.unique(y)) < 2 or len(y) < self.min_train_rows:
                logs.record_log(f"[MLPipeline] {s} not enough class variety/rows for training.", 2)
                return

            # model = self.model_builder()  # fresh instance each train
            # print(model)
            model = self.estimator[s]
            print(f'model={model}')
            # CV

            # cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            # try:
            #     acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            #     logs.record_log(f"[MLPipeline] {s} CV accuracy: {acc.mean():.3f} ± {acc.std():.3f}", 1)
            # except Exception as exc:
            #     logs.record_log(f"[MLPipeline] {s} CV failed: {exc}", 3)

            # Fit on all data
            model.fit(X, y)
            self.estimator[s] = model

            _log_feature_importance(f"[MLPipeline] {s}", model, feat_names)
        except Exception as exc:
            logs.record_log(f"[MLPipeline] {s} train error: {exc}", 3)

    def _predict_latest(self, s: str) -> Tuple[int, float]:
        try:
            if s not in self.estimator:
                return 0, 0.0
            model = self.estimator[s]
            if model is None:
                return 0, 0.0
            X, y, _ = self._make_dataset(s)
            if X.empty:
                return 0, 0.0
            x_last = X.iloc[[-1]]
            # Try to get probabilities; if not available, fallback
            proba = None
            if hasattr(model, "predict_proba"):
                proba = float(np.max(model.predict_proba(x_last)))
                pred_cls = int(model.predict(x_last)[0])
                # map 0→-1 or keep 0? We keep 0/1; strategy can map to BUY/SELL policy
                pred = 1 if pred_cls == 1 else -1
                return pred, proba
            # Decision function fallback
            if hasattr(model, "decision_function"):
                score = float(np.ravel(model.decision_function(x_last))[0])
                pred = 1 if score > 0 else -1
                return pred, abs(score)
            # Final fallback
            pred_cls = int(model.predict(x_last)[0])
            pred = 1 if pred_cls == 1 else -1
            return pred, 0.0
        except Exception as exc:
            logs.record_log(f"[MLPipeline] {s} predict error: {exc}", 3)
            return 0, 0.0
