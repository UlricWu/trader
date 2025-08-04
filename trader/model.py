#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : models.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/27 11:43
import json
import os
import pickle
from datetime import datetime

# trader/model.py
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import re


class Model:
    def __init__(self, settings, symbol: str):
        self.settings = settings
        self.symbol = symbol
        self.model = None

    def create_model(self):
        if self.settings.model.model_type == "RandomForest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.settings.model.model_type}")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model = self.create_model()
        self.model.fit(X_train, y_train)

        # Evaluate if validation set is provided
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
            # feature_names = X_train.columns.tolist() if hasattr(X_train, "columns") else []
            self._save_metrics(metrics)

            # Early stopping logic
            if metrics["log_loss"] > self.settings.model.early_stopping_logloss_threshold:
                print(f"[{self.symbol}] 🚫 Early Stopping: LogLoss {metrics['log_loss']:.4f} too high.")
                return

        self.save_model()

    def predict(self, X, prob=False):
        if self.model is None:
            self.model = self.load_model()

        if prob:
            return self.model.predict_proba(X)
        return self.model.predict(X)

    def save_model(self):
        """Save model, optionally auto-increment version."""

        full_dir = self._resolve_path(self.settings.model.model_dir)
        os.makedirs(full_dir, exist_ok=True)

        symbol = self.symbol

        if not self.model:
            raise ValueError("No model to save.")

        # Determine filename
        if self.settings.model.auto_version:
            filename = self._get_next_version_filename(full_dir, symbol)
        else:
            filename = f"{symbol}_model.pkl"

        full_path = os.path.join(full_dir, filename)

        with open(full_path, "wb") as f:
            pickle.dump(self.model, f)

        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, "wb") as f:
            pickle.dump(self.model, f)
            print(f"✅ Model saved: {full_path}")

        # Save "latest" shortcut
        if self.settings.model.save_latest:
            latest_path = os.path.join(full_dir, f"{symbol}_model_latest.pkl")
            with open(latest_path, "wb") as f:
                pickle.dump(self.model, f)
            print(f"[Latest Updated] {latest_path}")

    def load_model(self, model_name: str = None):
        """Load a model from a file."""

        relative_path = self._resolve_path(self.settings.model.model_dir)

        if model_name is None:
            model_name = f"{self.symbol}_latest.pkl"

        full_path = os.path.join(relative_path, model_name)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file does not exist: {full_path}")

        with open(full_path, "rb") as f:
            self.model = pickle.load(f)

    def _resolve_path(self, relative_path: str) -> str:
        """Resolve a relative path to absolute, relative to project root."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.join(project_root, relative_path)

    def _get_next_version_filename(self, full_dir: str, symbol: str) -> str:
        """Determine next versioned filename for saving."""
        pattern = re.compile(rf"{re.escape(symbol)}_model_v(\d+)\.pkl")
        versions = []

        for file in os.listdir(full_dir):
            match = pattern.match(file)
            if match:
                versions.append(int(match.group(1)))

        next_version = max(versions, default=0) + 1
        return f"{symbol}_model_v{next_version}.pkl"

    def evaluate(self, X_val, y_val):
        y_pred = self.predict(X_val)
        y_proba = self.predict(X_val, prob=True)

        acc = accuracy_score(y_val, y_pred)
        ll = log_loss(y_val, y_proba)
        try:
            auc = roc_auc_score(y_val, y_proba[:, 1])
        except ValueError:
            auc = None

        print(f"[{self.symbol}] 📊 Validation Accuracy: {acc:.4f}")
        print(f"[{self.symbol}] 📊 Validation LogLoss: {ll:.4f}")
        if auc is not None:
            print(f"[{self.symbol}] 📊 Validation ROC AUC: {auc:.4f}")

        return {"accuracy": acc, "log_loss": ll, "roc_auc": auc}

    def _save_metrics(self, metrics: dict):
        """Save model evaluation metrics with full training metadata."""
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Metadata
        metadata = {
            "symbol": self.symbol,
            "model_name": self.settings.model.model_type,
            "model_params": {
                "n_estimators": self.settings.model.n_estimators,
                "random_state": self.settings.model.random_state,
            },
            "settings": {
                "early_stopping_logloss_threshold": self.settings.model.early_stopping_logloss_threshold
            },
            "metrics": metrics,
            "trained_at": now
        }
        # "train_samples": len(X_train) if X_train is not None else None,
        # "val_samples": len(X_val) if X_val is not None else None,
        # "feature_names": feature_names if feature_names is not None else [],

        full_dir = self._resolve_path(self.settings.model.model_dir)
        full_path = os.path.join(full_dir, "metrics.json")
        if os.path.exists(full_path):
            # Load existing metrics if exists
            with open(full_path, "r") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}

        all_metrics.append(metadata)

        with open(full_path, "w+") as f:
            json.dump(all_metrics, f, indent=4)

        print(f"[{self.symbol}] 📁 Metrics saved and appended to {full_path}")
