# # src/classifier/pairwise_classifier.py
#
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import List, Optional, Dict, Any, Tuple
#
# import numpy as np
# import pandas as pd
#
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     roc_auc_score, average_precision_score,
#     precision_recall_curve, precision_score, recall_score
# )
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
#
# # Optional XGBoost
# _HAS_XGB = True
# try:
#     from xgboost import XGBClassifier
# except Exception:
#     _HAS_XGB = False
#
#
# @dataclass
# class TrainedMatcher:
#     """Trained ER matcher with OOF metrics, threshold, and prediction helpers."""
#     model_name: str
#     model: Any
#     scaler: Optional[StandardScaler]
#     feature_cols: List[str]
#     best_threshold: float
#     metrics: Dict[str, Any]
#     oof_prob: np.ndarray
#
#     def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
#         X = df[self.feature_cols].astype(float).fillna(0.0).values
#         if self.scaler is not None:
#             X = self.scaler.transform(X)
#         return self.model.predict_proba(X)[:, 1]
#
#     def predict(self, df: pd.DataFrame) -> np.ndarray:
#         p = self.predict_proba(df)
#         return (p >= self.best_threshold).astype(int)
#
#
# def _select_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
#     """Pick probability threshold maximizing F1."""
#     prec, rec, thr = precision_recall_curve(y_true, y_prob)
#     f1_vals, thrs = [], []
#     for i in range(len(thr)):
#         p, r = prec[i+1], rec[i+1]
#         f1 = (2*p*r) / (p+r) if (p+r) > 0 else 0.0
#         f1_vals.append(f1); thrs.append(thr[i])
#     if not thrs:
#         return 0.5, {"f1": 0.0, "precision": 0.0, "recall": 0.0}
#     j = int(np.argmax(f1_vals))
#     best_thr = float(thrs[j])
#     preds = (y_prob >= best_thr).astype(int)
#     return best_thr, {
#         "f1": float(f1_vals[j]),
#         "precision": float(precision_score(y_true, preds, zero_division=0)),
#         "recall": float(recall_score(y_true, preds, zero_division=0)),
#     }
#
#
# def _build_model(model_name: str, class_weight: Dict[int, float], scale_pos_weight: Optional[float]):
#     """Return an unfitted estimator. Scaling is handled outside for ALL models."""
#     if model_name == "logreg":
#         return LogisticRegression(max_iter=2000, solver="liblinear", class_weight=class_weight)
#     if model_name == "rf":
#         return RandomForestClassifier(
#             n_estimators=400, max_depth=None, min_samples_split=2,
#             n_jobs=-1, class_weight=class_weight, random_state=42
#         )
#     if model_name == "xgb":
#         if not _HAS_XGB:
#             raise RuntimeError("XGBoost not installed. Install with: pip install xgboost")
#         return XGBClassifier(
#             n_estimators=600, max_depth=6, learning_rate=0.05,
#             subsample=0.9, colsample_bytree=0.9,
#             reg_lambda=1.0, objective="binary:logistic",
#             tree_method="hist", n_jobs=-1, eval_metric="logloss",
#             scale_pos_weight=(scale_pos_weight or 1.0),
#             random_state=42
#         )
#     raise ValueError(f"Unknown model: {model_name}. Choose from: logreg | rf | xgb")
#
#
# def train_pairwise_matcher(
#     df: pd.DataFrame,
#     feature_cols: List[str],
#     label_col: str = "label",
#     model_name: str = "logreg",
#     n_folds: int = 5,
#     random_state: int = 42
# ) -> TrainedMatcher:
#     """
#     Train a pairwise ER matcher with stratified OOF estimates, select an OOF F1-optimal
#     threshold, and refit on all data. Returns a TrainedMatcher.
#
#     NOTE: Features are standardized (mean=0, std=1) for ALL models (LR/RF/XGB).
#     """
#     # Extract X, y
#     X = df[feature_cols].astype(float).fillna(0.0).values
#     y = df[label_col].astype(int).values
#
#     # Handle class imbalance
#     classes = np.array([0, 1])
#     cw_vals = compute_class_weight(class_weight="balanced", classes=classes, y=y)
#     class_weight = {int(k): float(v) for k, v in zip(classes, cw_vals)}
#     scale_pos_weight = float(cw_vals[0] / cw_vals[1])
#
#     # Estimator + scaler (always)
#     clf = _build_model(model_name, class_weight, scale_pos_weight)
#     scaler = StandardScaler()
#
#     # CV training (fit scaler ONLY on training folds)
#     skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
#     oof_prob = np.zeros(len(df), dtype=float)
#     fold_reports = []
#
#     for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
#         X_tr, X_va = X[tr], X[va]
#         y_tr, y_va = y[tr], y[va]
#
#         scaler.fit(X_tr)
#         X_tr_s = scaler.transform(X_tr)
#         X_va_s = scaler.transform(X_va)
#
#         clf.fit(X_tr_s, y_tr)
#         prob_va = clf.predict_proba(X_va_s)[:, 1]
#         oof_prob[va] = prob_va
#
#         roc = roc_auc_score(y_va, prob_va)
#         pr  = average_precision_score(y_va, prob_va)
#         thr, f1m = _select_threshold_by_f1(y_va, prob_va)
#         fold_reports.append({
#             "fold": fold,
#             "roc_auc": float(roc),
#             "pr_auc": float(pr),
#             "best_thr": float(thr),
#             "f1_at_best": float(f1m["f1"]),
#             "prec_at_best": float(f1m["precision"]),
#             "rec_at_best": float(f1m["recall"]),
#         })
#
#     # Global OOF metrics
#     roc_all = roc_auc_score(y, oof_prob)
#     pr_all  = average_precision_score(y, oof_prob)
#     best_thr_global, f1m_global = _select_threshold_by_f1(y, oof_prob)
#
#     metrics = {
#         "cv_folds": fold_reports,
#         "oof_roc_auc": float(roc_all),
#         "oof_pr_auc": float(pr_all),
#         "oof_best_thr": float(best_thr_global),
#         "oof_f1_at_best": float(f1m_global["f1"]),
#         "oof_prec_at_best": float(f1m_global["precision"]),
#         "oof_rec_at_best": float(f1m_global["recall"]),
#         "pos_frac": float(y.mean()),
#         "model": model_name,
#         "features": list(feature_cols),
#     }
#
#     # Retrain on ALL data with scaler fitted on ALL data
#     scaler.fit(X)
#     X_full = scaler.transform(X)
#     clf.fit(X_full, y)
#
#     return TrainedMatcher(
#         model_name=model_name,
#         model=clf,
#         scaler=scaler,                    # always present now
#         feature_cols=list(feature_cols),
#         best_threshold=float(best_thr_global),
#         metrics=metrics,
#         oof_prob=oof_prob
#     )
# src/classifier/pairwise_classifier.py



#zakomentiraniov kod e so povekje classifiers, stariot pred da se dodade ovoj dole:git

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional XGBoost
_HAS_XGB = True
try:
    from xgboost import XGBClassifier
except Exception:
    _HAS_XGB = False


@dataclass
class TrainedMatcher:
    """Trained ER matcher with OOF metrics, threshold, and prediction helpers."""
    model_name: str
    model: Any
    scaler: StandardScaler
    feature_cols: List[str]
    best_threshold: float
    metrics: Dict[str, Any]
    oof_prob: np.ndarray

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted probabilities for new pairs."""
        X = df[self.feature_cols].astype(float).fillna(0.0).values
        X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary predictions (0/1) using the best threshold."""
        p = self.predict_proba(df)
        return (p >= self.best_threshold).astype(int)


# ----------------- helpers -----------------

def _select_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Pick probability threshold maximizing F1."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1_vals, thrs = [], []
    for i in range(len(thr)):
        p, r = prec[i+1], rec[i+1]
        f1 = (2*p*r) / (p+r) if (p+r) > 0 else 0.0
        f1_vals.append(f1); thrs.append(thr[i])
    if not thrs:
        return 0.5, {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    j = int(np.argmax(f1_vals))
    best_thr = float(thrs[j])
    preds = (y_prob >= best_thr).astype(int)
    return best_thr, {
        "f1": float(f1_vals[j]),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
    }


def _build_model(model_name: str, class_weight: Dict[int, float], scale_pos_weight: Optional[float]):
    """Return an unfitted estimator."""
    if model_name == "logreg":
        return LogisticRegression(max_iter=2000, solver="liblinear", class_weight=class_weight)
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_split=2,
            n_jobs=-1, class_weight=class_weight, random_state=42
        )
    if model_name == "xgb":
        if not _HAS_XGB:
            raise RuntimeError("XGBoost not installed. Install with: pip install xgboost")
        return XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, objective="binary:logistic",
            tree_method="hist", n_jobs=-1, eval_metric="logloss",
            scale_pos_weight=(scale_pos_weight or 1.0),
            random_state=42
        )
    raise ValueError(f"Unknown model: {model_name}. Choose from: logreg | rf | xgb")


# ----------------- main training -----------------

def train_pairwise_matcher(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    model_name: str = "xgb",
    n_folds: int = 5,
    random_state: int = 42
) -> TrainedMatcher:
    """
    Train a single ER matcher model with stratified OOF evaluation,
    pick the best threshold by F1, and refit on all data.
    """
    # Extract features and labels
    X = df[feature_cols].astype(float).fillna(0.0).values
    y = df[label_col].astype(int).values

    # Handle class imbalance
    classes = np.array([0, 1])
    cw_vals = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight = {int(k): float(v) for k, v in zip(classes, cw_vals)}
    scale_pos_weight = float(cw_vals[0] / cw_vals[1])

    # Estimator and scaler
    clf = _build_model(model_name, class_weight, scale_pos_weight)
    scaler = StandardScaler()

    # Cross-validation for OOF predictions
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_prob = np.zeros(len(df), dtype=float)
    fold_reports = []

    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        scaler.fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)

        clf.fit(X_tr_s, y_tr)
        prob_va = clf.predict_proba(X_va_s)[:, 1]
        oof_prob[va] = prob_va

        roc = roc_auc_score(y_va, prob_va)
        pr  = average_precision_score(y_va, prob_va)
        thr, f1m = _select_threshold_by_f1(y_va, prob_va)
        fold_reports.append({
            "fold": fold,
            "roc_auc": float(roc),
            "pr_auc": float(pr),
            "best_thr": float(thr),
            "f1_at_best": float(f1m["f1"]),
            "prec_at_best": float(f1m["precision"]),
            "rec_at_best": float(f1m["recall"]),
        })

    # Global OOF metrics
    roc_all = roc_auc_score(y, oof_prob)
    pr_all  = average_precision_score(y, oof_prob)
    best_thr_global, f1m_global = _select_threshold_by_f1(y, oof_prob)

    metrics = {
        "cv_folds": fold_reports,
        "oof_roc_auc": float(roc_all),
        "oof_pr_auc": float(pr_all),
        "oof_best_thr": float(best_thr_global),
        "oof_f1_at_best": float(f1m_global["f1"]),
        "oof_prec_at_best": float(f1m_global["precision"]),
        "oof_rec_at_best": float(f1m_global["recall"]),
        "pos_frac": float(y.mean()),
        "model": model_name,
        "features": list(feature_cols),
    }

    # Retrain on full dataset
    scaler.fit(X)
    X_full = scaler.transform(X)
    clf.fit(X_full, y)

    return TrainedMatcher(
        model_name=model_name,
        model=clf,
        scaler=scaler,
        feature_cols=list(feature_cols),
        best_threshold=float(best_thr_global),
        metrics=metrics,
        oof_prob=oof_prob
    )
