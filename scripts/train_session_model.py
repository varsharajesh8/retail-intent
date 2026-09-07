import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, classification_report, precision_recall_curve
)
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint, uniform

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURE_COLS_BASE = [
    "n_events", "n_views", "n_carts", "n_distinct_products",
    "n_distinct_categories", "avg_price_viewed", "max_price_viewed",
    "session_duration_sec", "pct_missing_category", "has_cart_add",
]
FEATURE_COLS_WITH_HISTORY = FEATURE_COLS_BASE + [
    "user_past_orders", "user_past_spend", "user_days_since_last_purchase",
]
LABEL_COL = "purchased"


def load_and_split(path: Path, feature_cols: list):
    """Load a session feature table, split features/label, then train/test split."""
    df = pd.read_parquet(path)
    X = df[feature_cols].copy()
    y = df[LABEL_COL].astype(int)

    X["has_cart_add"] = X["has_cart_add"].astype(int)
    if "user_days_since_last_purchase" in X.columns:
        # NaN = no prior purchase; fill with a large sentinel so logistic
        # regression can use it too (LightGBM handles NaN natively, but we
        # keep inputs consistent across both models for a fair comparison)
        X["user_days_since_last_purchase"] = X["user_days_since_last_purchase"].fillna(9999)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=38, stratify=y
    )
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    print(f"\n--- {model_name} ---")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k}: {v:.4f}")
    return metrics

def find_best_threshold(model, X_test, y_test) -> dict:
    """Sweep classification thresholds to find the one maximizing F1,
    rather than relying on the default 0.5 cutoff."""
    y_proba = model.predict_proba(X_test)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = f1_scores[:-1].argmax()  # last precision/recall point has no matching threshold

    return {
        "best_threshold": thresholds[best_idx],
        "precision_at_best": precisions[best_idx],
        "recall_at_best": recalls[best_idx],
        "f1_at_best": f1_scores[best_idx],
    }

def train_logistic_regression(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(class_weight="balanced", random_state=38, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return evaluate_model(model, X_test_scaled, y_test, "Logistic Regression")


def train_lightgbm(X_train, X_test, y_train, y_test):
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        class_weight="balanced",
        random_state=38,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test, "LightGBM")

def tune_lightgbm(X_train, y_train, n_iter: int = 40) -> LGBMClassifier:
    """
    Randomized hyperparameter search over LightGBM, covering the parameters
    that matter most for tabular, imbalanced classification:
    - n_estimators/learning_rate: (more trees, smaller steps)
    - max_depth/num_leaves: controls overfitting
    - min_child_samples: another overfitting control
    - subsample/colsample_bytree: row/column subsampling, reduces overfitting
      and speeds up training
    - reg_alpha/reg_lambda: L1/L2 regularization on leaf weights

    RandomizedSearchCV (not GridSearchCV) is used because a full grid over
    this many parameters would require thousands of fits; random sampling
    covers the space far more efficiently for a fixed budget.
    """
    param_distributions = {
        "n_estimators": randint(100, 500),
        "learning_rate": uniform(0.01, 0.19),       # 0.01 to 0.20
        "max_depth": randint(3, 12),
        "num_leaves": randint(15, 127),
        "min_child_samples": randint(10, 100),
        "subsample": uniform(0.6, 0.4),              # 0.6 to 1.0
        "colsample_bytree": uniform(0.6, 0.4),       # 0.6 to 1.0
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.0, 1.0),
    }

    base_model = LGBMClassifier(class_weight="balanced", random_state=38, verbose=-1)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="average_precision",
        cv=3,
        n_jobs=-1,
        random_state=38,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV PR-AUC: {search.best_score_:.4f}")
    return search.best_estimator_


if __name__ == "__main__":
    print("=== ORACLE MODEL (full session, includes leakage) ===")
    X_train, X_test, y_train, y_test = load_and_split(DATA_DIR / "session_features.parquet", FEATURE_COLS_BASE)
    oracle_lr = train_logistic_regression(X_train, X_test, y_train, y_test)
    oracle_gbm = train_lightgbm(X_train, X_test, y_train, y_test)

    print("\n=== REAL-TIME MODEL — BASE FEATURES (leakage-safe, 10 cols) ===")
    X_train_rt, X_test_rt, y_train_rt, y_test_rt = load_and_split(
        DATA_DIR / "session_features_realtime.parquet", FEATURE_COLS_BASE
    )
    realtime_lr = train_logistic_regression(X_train_rt, X_test_rt, y_train_rt, y_test_rt)
    realtime_gbm = train_lightgbm(X_train_rt, X_test_rt, y_train_rt, y_test_rt)

    print("\n=== REAL-TIME MODEL — WITH USER HISTORY (leakage-safe, 13 cols) ===")
    X_train_h, X_test_h, y_train_h, y_test_h = load_and_split(
        DATA_DIR / "session_features_realtime.parquet", FEATURE_COLS_WITH_HISTORY
    )
    history_lr = train_logistic_regression(X_train_h, X_test_h, y_train_h, y_test_h)
    history_gbm = train_lightgbm(X_train_h, X_test_h, y_train_h, y_test_h)

    # Threshold tuning: is 0.5 actually the best cutoff, given class imbalance?
    print("\n=== THRESHOLD TUNING (with-history LightGBM) ===")
    model_h_final = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        class_weight="balanced", random_state=38, verbose=-1
    )
    model_h_final.fit(X_train_h, y_train_h)
    threshold_result = find_best_threshold(model_h_final, X_test_h, y_test_h)
    print(f"Default (0.5) F1 was: {history_gbm['f1']:.4f}")
    print(f"Best threshold: {threshold_result['best_threshold']:.4f}")
    print(f"At best threshold — precision: {threshold_result['precision_at_best']:.4f}, "
          f"recall: {threshold_result['recall_at_best']:.4f}, f1: {threshold_result['f1_at_best']:.4f}")

    # Hyperparameter search: can we beat the hand-picked settings above?
    print("\n=== HYPERPARAMETER SEARCH (with-history features) ===")
    best_model = tune_lightgbm(X_train_h, y_train_h)
    tuned_results = evaluate_model(best_model, X_test_h, y_test_h, "LightGBM (tuned)")
    tuned_threshold_result = find_best_threshold(best_model, X_test_h, y_test_h)
    print(f"Tuned model, tuned threshold — precision: {tuned_threshold_result['precision_at_best']:.4f}, "
          f"recall: {tuned_threshold_result['recall_at_best']:.4f}, f1: {tuned_threshold_result['f1_at_best']:.4f}")
    
    results = pd.DataFrame([oracle_lr, oracle_gbm, realtime_lr, realtime_gbm, history_lr, history_gbm])
    results.insert(0, "version", ["oracle", "oracle", "realtime_base", "realtime_base", "realtime_history", "realtime_history"])
    print("\n=== SUMMARY ===")
    print(results)
    results.to_csv(DATA_DIR / "session_model_results.csv", index=False)

    # Feature importance from the richer LightGBM model, to see whether
    # the new user-history features actually get used
    importances = pd.Series(
        LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                        class_weight="balanced", random_state=38, verbose=-1)
        .fit(X_train_h, y_train_h)
        .feature_importances_,
        index=FEATURE_COLS_WITH_HISTORY,
    ).sort_values(ascending=False)
    print("\n=== FEATURE IMPORTANCE (with history) ===")
    print(importances)