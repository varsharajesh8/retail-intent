from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FEATURE_COLS = [
    "recency_days",
    "days_since_first_seen",
    "n_sessions",
    "n_events",
    "n_purchase_events",
    "n_orders",
    "total_spend",
    "avg_price_viewed",
    "avg_order_value",
]
LABEL_COL = "will_purchase_next_7d"


def load_and_split_by_cutoff(
        path: Path,
        validation_cutoffs: list,
        test_cutoffs: list,
) -> tuple:
    """Splitting into train, validation, and test sets. Validation is for the purpose of model and threshold selection."""

    df = pd.read_parquet(path)

    X = df[FEATURE_COLS].copy()
    y = df[LABEL_COL].astype(int)

    # avg_order_value is NaN for users with no prior orders; fill with sentinel distinct from real vals
    X["avg_order_value"] = X["avg_order_value"].fillna(-1)

    is_validation = df["cutoff_date"].isin(validation_cutoffs)
    is_test = df["cutoff_date"].isin(test_cutoffs)
    is_train = ~(is_validation | is_test)

    X_train = X.loc[is_train].copy()
    y_train = y.loc[is_train].copy()

    X_val = X.loc[is_validation].copy()
    y_val = y.loc[is_validation].copy()

    X_test = X.loc[is_test].copy()
    y_test = y.loc[is_test].copy()

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

def evaluate_model(model, X, y, model_name: str) -> dict:
    """
    Evaluate ranking and default-threshold performance
    on a supplied dataset.
    """

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "model": model_name,
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }

    print(f"\n--- {model_name} ---")

    for k, v in metrics.items():
        if k != "model":
            print(f"{k}: {v:.4f}")

    return metrics

def find_best_threshold(model, X_val, y_val) -> dict:
    """Sweep classification thresholds on validation data to find threshold maximizing F1."""
    y_proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = f1_scores[:-1].argmax()

    return {
        "best_threshold": thresholds[best_idx],
        "precision_at_best": precisions[best_idx],
        "recall_at_best": recalls[best_idx],
        "f1_at_best": f1_scores[best_idx],
    }

def evaluate_at_threshold(
    model,
    X,
    y,
    threshold: float,
    model_name: str,
) -> dict:
    """
    Evaluate using a threshold selected previously on validation data.
    """

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    business_metrics = ranking_metrics_at_k(
        y,
        y_proba,
        k=0.05,
    )

    metrics = {
        "model": model_name,
        "threshold": threshold,
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }

    metrics.update(business_metrics)

    print(f"\n--- {model_name} ---")

    for k, v in metrics.items():
        if k != "model":
            print(f"{k}: {v:.4f}")

    return metrics


def ranking_metrics_at_k(
    y_true,
    y_proba,
    k: float = 0.05,
) -> dict:
    """
    Evaluate the highest-scoring k fraction of users.

    Example:
        k=0.05 evaluates the top 5% of users ranked
        by predicted purchase propensity. useful when marketing can only target top k% of users.
    """

    results = pd.DataFrame({
        "actual": pd.Series(y_true).reset_index(drop=True),
        "score": y_proba,
    })

    results = results.sort_values(
        "score",
        ascending=False,
    )

    n_top = max(1, int(len(results) * k))

    top_k = results.head(n_top)

    precision_at_k = top_k["actual"].mean()

    total_positives = results["actual"].sum()

    if total_positives > 0:
        recall_at_k = (
            top_k["actual"].sum()
            / total_positives
        )
    else:
        recall_at_k = 0.0

    base_rate = results["actual"].mean()

    if base_rate > 0:
        lift_at_k = precision_at_k / base_rate
    else:
        lift_at_k = 0.0

    return {
        f"precision_at_{int(k * 100)}pct": precision_at_k,
        f"recall_at_{int(k * 100)}pct": recall_at_k,
        f"lift_at_{int(k * 100)}pct": lift_at_k,
    }

def fit_lightgbm_with_weight(
    X_train,
    y_train,
    scale_pos_weight: float,
):
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        random_state=38,
        verbose=-1,
    )

    model.fit(X_train, y_train)

    return model

ALL_CUTOFFS = pd.date_range(
    "2019-10-04",
    "2019-10-24",
    freq="2D",
    tz="UTC",
)

VALIDATION_CUTOFFS = list(ALL_CUTOFFS[-4:-2])
TEST_CUTOFFS = list(ALL_CUTOFFS[-2:])

if __name__ == "__main__":

    print(
        "Validation cutoffs:",
        [str(c.date()) for c in VALIDATION_CUTOFFS],
    )

    print(
        "Test cutoffs:",
        [str(c.date()) for c in TEST_CUTOFFS],
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load_and_split_by_cutoff(
        DATA_DIR / "user_features.parquet",
        VALIDATION_CUTOFFS,
        TEST_CUTOFFS,
    )

    print("\n=== DATA SPLIT ===")
    print(f"Train rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Test rows: {len(X_test):,}")

    print(f"Train positive rate: {y_train.mean():.4%}")
    print(f"Validation positive rate: {y_val.mean():.4%}")
    print(f"Test positive rate: {y_test.mean():.4%}")

    print("\n=== LOGISTIC REGRESSION BASELINE ===")

    scaler = StandardScaler()

    # Learn scaling parameters from train only
    X_train_scaled = scaler.fit_transform(X_train)
    # Apply same scaling to validation and test
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)


    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=38,
    )

    lr_model.fit(X_train_scaled, y_train)

    lr_validation_results = evaluate_model(
        lr_model,
        X_val_scaled,
        y_val,
        "Logistic Regression — Validation",
    )

    print("\n=== LOGISTIC REGRESSION THRESHOLD SELECTION ===")

    lr_threshold_result = find_best_threshold(
        lr_model,
        X_val_scaled,
        y_val,
    )

    lr_best_threshold = float(
        lr_threshold_result["best_threshold"]
    )

    print(
        f"Best Logistic Regression validation threshold: "
        f"{lr_best_threshold:.4f}"
    )

    print(
        "Validation performance at selected threshold — "
        f"precision: {lr_threshold_result['precision_at_best']:.4f}, "
        f"recall: {lr_threshold_result['recall_at_best']:.4f}, "
        f"f1: {lr_threshold_result['f1_at_best']:.4f}"
    )

    print("\n=== SCALE_POS_WEIGHT VALIDATION SWEEP ===")
    # Evaluate different positive-class weights to address class imbalance. Model selection based only on validation PR-AUC
    weight_candidates = [
        1,
        5,
        10,
        20,
        50,
        100,
        115,
    ]

    validation_results = []

    for w in weight_candidates:

        model = fit_lightgbm_with_weight(
            X_train,
            y_train,
            scale_pos_weight=w,
        )

        result = evaluate_model(
            model,
            X_val,
            y_val,
            f"LightGBM weight={w} — Validation",
        )

        result["scale_pos_weight"] = w
        validation_results.append(result)

    validation_df = pd.DataFrame(validation_results)

    print("\n=== VALIDATION WEIGHT SUMMARY ===")

    print(
        validation_df[
            [
                "scale_pos_weight",
                "roc_auc",
                "pr_auc",
                "precision",
                "recall",
                "f1",
            ]
        ]
    )

    best_row = validation_df.loc[
        validation_df["pr_auc"].idxmax()
    ]

    best_weight = float(
        best_row["scale_pos_weight"]
    )

    print(
        f"\nBest validation scale_pos_weight by PR-AUC: "
        f"{best_weight}"
    )

    best_model = fit_lightgbm_with_weight(
        X_train,
        y_train,
        scale_pos_weight=best_weight,
    )

    print("\n=== VALIDATION THRESHOLD TUNING ===")

    lgbm_threshold_result = find_best_threshold(
        best_model,
        X_val,
        y_val,
    )

    lgbm_best_threshold = float(
        lgbm_threshold_result["best_threshold"]
    )

    print(
        f"Best validation threshold: "
        f"{lgbm_best_threshold:.4f}"
    )

    print(
        "At validation threshold — "
        f"precision: {lgbm_threshold_result['precision_at_best']:.4f}, "
        f"recall: {lgbm_threshold_result['recall_at_best']:.4f}, "
        f"f1: {lgbm_threshold_result['f1_at_best']:.4f}"
    )

    print("\n=== FINAL UNTOUCHED TEST EVALUATION ===")

    lr_test_results = evaluate_at_threshold(
        lr_model,
        X_test_scaled,
        y_test,
        lr_best_threshold,
        "Logistic Regression — Test",
    )

    lgbm_test_results = evaluate_at_threshold(
        best_model,
        X_test,
        y_test,
        lgbm_best_threshold,
        "LightGBM — Test",
    )

    validation_df.to_csv(
        DATA_DIR / "weekly_validation_weight_results.csv",
        index=False,
    )

    final_comparison_df = pd.DataFrame([
        lr_test_results,
        lgbm_test_results,
    ])

    print("\n=== FINAL MODEL COMPARISON ===")

    comparison_cols = [
        "model",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "precision_at_5pct",
        "recall_at_5pct",
        "lift_at_5pct",
        "threshold",
    ]

    print(
        final_comparison_df[
            comparison_cols
        ].to_string(index=False)
    )

    final_comparison_df.to_csv(
        DATA_DIR / "weekly_final_test_results.csv",
        index=False,
    )
