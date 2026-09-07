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

FEATURE_COLS = [
    "n_events", "n_views", "n_carts", "n_distinct_products",
    "n_distinct_categories", "avg_price_viewed", "max_price_viewed",
    "session_duration_sec", "pct_missing_category", "has_cart_add",
]
LABEL_COL = "purchased"


def load_and_split(path: Path):
    """Load a session feature table, split features/label, then train/test split."""
    df = pd.read_parquet(path)
    X = df[FEATURE_COLS].copy()
    y = df[LABEL_COL].astype(int)

    # bool -> int for sklearn/lightgbm compatibility
    X["has_cart_add"] = X["has_cart_add"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=38, stratify=y
    )
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of the positive class

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
        n_estimators = 200,
        learning_rate = 0.05,
        max_depth = 6,
        class_weight = "balanced",
        random_state = 38,
        verbose = 1,
    )
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test, "LightGBM")

if __name__ == "__main__":
    print("=== ORACLE MODEL (full session, includes leakage) ===")
    X_train, X_test, y_train, y_test = load_and_split(DATA_DIR / "session_features.parquet")
    oracle_lr = train_logistic_regression(X_train, X_test, y_train, y_test)
    oracle_gbm = train_lightgbm(X_train, X_test, y_train, y_test)

    print("\n=== REAL-TIME MODEL (pre-purchase events only, leakage-safe) ===")
    X_train_rt, X_test_rt, y_train_rt, y_test_rt = load_and_split(DATA_DIR / "session_features_realtime.parquet")
    realtime_lr = train_logistic_regression(X_train_rt, X_test_rt, y_train_rt, y_test_rt)
    realtime_gbm = train_lightgbm(X_train_rt, X_test_rt, y_train_rt, y_test_rt)

    results = pd.DataFrame([oracle_lr, oracle_gbm, realtime_lr, realtime_gbm])
    results.insert(0, "version", ["oracle", "oracle", "realtime", "realtime"])
    print("\n=== SUMMARY ===")
    print(results)
    results.to_csv(DATA_DIR / "session_model_results.csv", index=False)