import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, classification_report
)
from lightgbm import LGBMClassifier

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