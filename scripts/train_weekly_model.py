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

FEATURE_COLS_CORE = [
    "recency_days",
    "days_since_first_seen",
    "n_sessions",
    "n_events",
    "avg_price_viewed",
]

FEATURE_COLS_ENGAGEMENT_ONLY = [
    "n_views",
    "n_carts",
    "events_per_session",
    "views_per_session",
    "carts_per_session",
    "cart_view_ratio",
    "unique_products",
    "unique_categories",
]

FEATURE_COLS_PURCHASE_HISTORY = [
    "n_purchase_events",
    "n_orders",
    "total_spend",
    "avg_order_value",
    "historical_conversion_rate",
    "days_since_last_purchase",
]

FEATURE_COLS_CORE_ENGAGEMENT = (
    FEATURE_COLS_CORE
    + FEATURE_COLS_ENGAGEMENT_ONLY
)

FEATURE_COLS_FULL_STRUCTURED = (
    FEATURE_COLS_CORE
    + FEATURE_COLS_ENGAGEMENT_ONLY
    + FEATURE_COLS_PURCHASE_HISTORY
)

FEATURE_COLS_PRODUCT_CATEGORY = [
    "unique_products_viewed",
    "unique_categories_viewed",
    "unique_brands",
    "max_views_same_product",
    "repeat_view_ratio",
    "category_diversity_ratio",
    "brand_diversity_ratio",
    "min_price_viewed",
    "median_price_viewed",
    "max_price_viewed",
    "std_price_viewed",
    "price_range_viewed",
]

FEATURE_COLS_FULL_PLUS_PRODUCT = (
    FEATURE_COLS_FULL_STRUCTURED
    + FEATURE_COLS_PRODUCT_CATEGORY
)

FEATURE_COLS_SEMANTIC = [
    f"user_text_dim_{i}"
    for i in range(20)
]

FEATURE_COLS_FULL_PLUS_PRODUCT_SEMANTIC = (
    FEATURE_COLS_FULL_PLUS_PRODUCT
    + FEATURE_COLS_SEMANTIC
)
LABEL_COL = "will_purchase_next_7d"


def load_and_split_by_cutoff(
    path: Path,
    validation_cutoffs: list,
    test_cutoffs: list,
) -> tuple:
    """
    Split the user-cutoff dataset chronologically into
    train, validation, and test sets.

    Feature columns are selected later so that different
    feature sets can be compared on the exact same rows.
    """

    df = pd.read_parquet(path)

    # avg_order_value is NaN for users with no prior orders.
    # Use -1 as a sentinel distinct from real order values.
    df["avg_order_value"] = df["avg_order_value"].fillna(-1)

    is_validation = df["cutoff_date"].isin(validation_cutoffs)
    is_test = df["cutoff_date"].isin(test_cutoffs)
    is_train = ~(is_validation | is_test)

    train_df = df.loc[is_train].copy()
    val_df = df.loc[is_validation].copy()
    test_df = df.loc[is_test].copy()

    return train_df, val_df, test_df

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

# Purged / embargoed temporal split.
#
# Train:
#   Oct 4 label -> Oct 4-11
#   Oct 6 label -> Oct 6-13
#   Oct 8 label -> Oct 8-15
#
# Validation:
#   Oct 16 label -> Oct 16-23
#
# Test:
#   Oct 24 label -> Oct 24-31
#
# The unused cutoffs act as an embargo so label windows
# do not overlap across train / validation / test.
TRAIN_CUTOFFS = [
    ALL_CUTOFFS[0],   # Oct 4
    ALL_CUTOFFS[1],   # Oct 6
    ALL_CUTOFFS[2],   # Oct 8
]

VALIDATION_CUTOFFS = [
    ALL_CUTOFFS[6],   # Oct 16
]

TEST_CUTOFFS = [
    ALL_CUTOFFS[10],  # Oct 24
]


if __name__ == "__main__":

    print(
        "Train cutoffs:",
        [str(c.date()) for c in TRAIN_CUTOFFS],
    )

    print(
        "Validation cutoffs:",
        [str(c.date()) for c in VALIDATION_CUTOFFS],
    )

    print(
        "Test cutoffs:",
        [str(c.date()) for c in TEST_CUTOFFS],
    )

    # =========================================================
    # LOAD DATA
    # =========================================================

    df = pd.read_parquet(
        DATA_DIR / "user_features.parquet"
    )

    # Users with no previous orders have undefined AOV.
    # Use -1 as a sentinel so this remains distinguishable
    # from a real zero-dollar value.
    df["avg_order_value"] = (
        df["avg_order_value"].fillna(-1)
    )

    # =========================================================
    # PURGED TEMPORAL SPLIT
    # =========================================================

    is_train = df["cutoff_date"].isin(
        TRAIN_CUTOFFS
    )

    is_validation = df["cutoff_date"].isin(
        VALIDATION_CUTOFFS
    )

    is_test = df["cutoff_date"].isin(
        TEST_CUTOFFS
    )

    train_df = df.loc[is_train].copy()
    val_df = df.loc[is_validation].copy()
    test_df = df.loc[is_test].copy()

    # Labels
    y_train = train_df[
        LABEL_COL
    ].astype(int)

    y_val = val_df[
        LABEL_COL
    ].astype(int)

    y_test = test_df[
        LABEL_COL
    ].astype(int)

    # =========================================================
    # FEATURE MATRICES
    # =========================================================

    # -------------------------
    # Core features
    # -------------------------

    X_train_core = train_df[
        FEATURE_COLS_CORE
    ].copy()

    X_val_core = val_df[
        FEATURE_COLS_CORE
    ].copy()

    # -------------------------
    # Core + Engagement/Funnel
    # -------------------------

    X_train_core_engagement = train_df[
        FEATURE_COLS_CORE_ENGAGEMENT
    ].copy()

    X_val_core_engagement = val_df[
        FEATURE_COLS_CORE_ENGAGEMENT
    ].copy()

    # -------------------------
    # Full structured set:
    # Core + Engagement + Purchase History
    # -------------------------

    X_train_full = train_df[
        FEATURE_COLS_FULL_STRUCTURED
    ].copy()

    X_val_full = val_df[
        FEATURE_COLS_FULL_STRUCTURED
    ].copy()

    # FULL PRODUCT
    X_train_full_product = train_df[
        FEATURE_COLS_FULL_PLUS_PRODUCT
    ]

    X_val_full_product = val_df[
        FEATURE_COLS_FULL_PLUS_PRODUCT
    ]

    # FULL PRODUCT SEMANTIC
    X_train_full_product_semantic = train_df[
        FEATURE_COLS_FULL_PLUS_PRODUCT_SEMANTIC
    ]

    X_val_full_product_semantic = val_df[
        FEATURE_COLS_FULL_PLUS_PRODUCT_SEMANTIC
    ]

    # =========================================================
    # DATA SPLIT SUMMARY
    # =========================================================

    print("\n=== DATA SPLIT ===")

    print(
        f"Train rows: {len(train_df):,}"
    )

    print(
        f"Validation rows: {len(val_df):,}"
    )

    print(
        f"Test rows: {len(test_df):,}"
    )

    print(
        f"Train positive rate: "
        f"{y_train.mean():.4%}"
    )

    print(
        f"Validation positive rate: "
        f"{y_val.mean():.4%}"
    )

    print(
        f"Test positive rate: "
        f"{y_test.mean():.4%}"
    )

    # =========================================================
    # FEATURE ABLATION
    # =========================================================

    print("\n" + "=" * 70)
    print(
        "FEATURE ABLATION: "
        "CORE -> ENGAGEMENT -> PURCHASE HISTORY -> PRODUCT/CATEGORY -> SEMANTIC"
    )
    print("=" * 70)

    # =========================================================
    # MODEL 1: CORE ONLY
    # =========================================================

    core_model = fit_lightgbm_with_weight(
        X_train_core,
        y_train,
        scale_pos_weight=1,
    )

    core_val_proba = (
        core_model.predict_proba(
            X_val_core
        )[:, 1]
    )

    core_metrics = {
        "model": "Core",
        "n_features": len(
            FEATURE_COLS_CORE
        ),
        "roc_auc": roc_auc_score(
            y_val,
            core_val_proba,
        ),
        "pr_auc": average_precision_score(
            y_val,
            core_val_proba,
        ),
    }

    core_metrics.update(
        ranking_metrics_at_k(
            y_val,
            core_val_proba,
            k=0.05,
        )
    )

    # =========================================================
    # MODEL 2: CORE + ENGAGEMENT
    # =========================================================

    engagement_model = (
        fit_lightgbm_with_weight(
            X_train_core_engagement,
            y_train,
            scale_pos_weight=1,
        )
    )

    engagement_val_proba = (
        engagement_model.predict_proba(
            X_val_core_engagement
        )[:, 1]
    )

    engagement_metrics = {
        "model": "Core + Engagement",
        "n_features": len(
            FEATURE_COLS_CORE_ENGAGEMENT
        ),
        "roc_auc": roc_auc_score(
            y_val,
            engagement_val_proba,
        ),
        "pr_auc": average_precision_score(
            y_val,
            engagement_val_proba,
        ),
    }

    engagement_metrics.update(
        ranking_metrics_at_k(
            y_val,
            engagement_val_proba,
            k=0.05,
        )
    )

    # =========================================================
    # MODEL 3:
    # CORE + ENGAGEMENT + PURCHASE HISTORY
    # =========================================================

    full_model = fit_lightgbm_with_weight(
        X_train_full,
        y_train,
        scale_pos_weight=1,
    )

    full_val_proba = (
        full_model.predict_proba(
            X_val_full
        )[:, 1]
    )

    full_metrics = {
        "model":
            "Core + Engagement + Purchase History",

        "n_features": len(
            FEATURE_COLS_FULL_STRUCTURED
        ),

        "roc_auc": roc_auc_score(
            y_val,
            full_val_proba,
        ),

        "pr_auc": average_precision_score(
            y_val,
            full_val_proba,
        ),
    }

    full_metrics.update(
        ranking_metrics_at_k(
            y_val,
            full_val_proba,
            k=0.05,
        )
    )

        # =========================================================
    # MODEL 4:
    # FULL STRUCTURED + PRODUCT/CATEGORY
    # =========================================================

    full_product_model = fit_lightgbm_with_weight(
        X_train_full_product,
        y_train,
        scale_pos_weight=1,
    )

    full_product_val_proba = (
        full_product_model.predict_proba(
            X_val_full_product
        )[:, 1]
    )

    full_product_metrics = {
        "model":
            "Full Structured + Product/Category",

        "n_features": len(
            FEATURE_COLS_FULL_PLUS_PRODUCT
        ),

        "roc_auc": roc_auc_score(
            y_val,
            full_product_val_proba,
        ),

        "pr_auc": average_precision_score(
            y_val,
            full_product_val_proba,
        ),
    }

    full_product_metrics.update(
        ranking_metrics_at_k(
            y_val,
            full_product_val_proba,
            k=0.05,
        )
    )

    # =========================================================
    # MODEL 5:
    # FULL STRUCTURED + PRODUCT/CATEGORY + SEMANTIC
    # =========================================================

    full_product_semantic_model = fit_lightgbm_with_weight(
        X_train_full_product_semantic,
        y_train,
        scale_pos_weight=1,
    )

    full_product_semantic_val_proba = (
        full_product_semantic_model.predict_proba(
            X_val_full_product_semantic
        )[:, 1]
    )

    full_product_semantic_metrics = {
        "model":
            "Full + Product/Category + Semantic",

        "n_features": len(
            FEATURE_COLS_FULL_PLUS_PRODUCT_SEMANTIC
        ),

        "roc_auc": roc_auc_score(
            y_val,
            full_product_semantic_val_proba,
        ),

        "pr_auc": average_precision_score(
            y_val,
            full_product_semantic_val_proba,
        ),
    }

    full_product_semantic_metrics.update(
        ranking_metrics_at_k(
            y_val,
            full_product_semantic_val_proba,
            k=0.05,
        )
    )

    # =========================================================
    # BUILD COMPARISON TABLE
    # =========================================================

    ablation_df = pd.DataFrame([
        core_metrics,
        engagement_metrics,
        full_metrics,
        full_product_metrics,
        full_product_semantic_metrics,
    ])

    print(
        "\n=== VALIDATION FEATURE ABLATION ==="
    )

    comparison_cols = [
        "model",
        "n_features",
        "roc_auc",
        "pr_auc",
        "precision_at_5pct",
        "recall_at_5pct",
        "lift_at_5pct",
    ]

    print(
        ablation_df[
            comparison_cols
        ].to_string(index=False)
    )

    # =========================================================
    # INCREMENTAL IMPROVEMENT
    # =========================================================

    core_pr = core_metrics[
        "pr_auc"
    ]

    engagement_pr = engagement_metrics[
        "pr_auc"
    ]

    full_pr = full_metrics[
        "pr_auc"
    ]

    full_product_pr = full_product_metrics[
        "pr_auc"
    ]

    semantic_pr = full_product_semantic_metrics[
        "pr_auc"
    ]


    engagement_pr_change = (
        (engagement_pr - core_pr)
        / core_pr
        * 100
    )

    purchase_history_pr_change = (
        (full_pr - engagement_pr)
        / engagement_pr
        * 100
    )

    total_pr_change = (
        (full_pr - core_pr)
        / core_pr
        * 100
    )

    product_pr_change = (
        (full_product_pr - full_pr)
        / full_pr
        * 100
    )


    semantic_pr_change = (
        (semantic_pr - full_product_pr)
        / full_product_pr
        * 100
    )

    print(
        "\n=== INCREMENTAL PR-AUC IMPROVEMENT ==="
    )

    print(
        "Core -> Core + Engagement: "
        f"{engagement_pr_change:+.2f}%"
    )

    print(
        "Core + Engagement -> "
        "Full Structured: "
        f"{purchase_history_pr_change:+.2f}%"
    )

    print(
        "Core -> Full Structured: "
        f"{total_pr_change:+.2f}%"
    )

    print(
        "Full Structured -> "
        "Full + Product/Category: "
        f"{product_pr_change:+.2f}%"
    )

    print(
        "Full + Product/Category -> "
        "Full + Product/Category + Semantic: "
        f"{semantic_pr_change:+.2f}%"
    )

    # =========================================================
    # CURRENT VALIDATION CHAMPION
    # =========================================================

    best_idx = ablation_df[
        "pr_auc"
    ].idxmax()

    best_feature_set = (
        ablation_df.loc[
            best_idx,
            "model",
        ]
    )

    best_pr_auc = (
        ablation_df.loc[
            best_idx,
            "pr_auc",
        ]
    )

    best_lift = (
        ablation_df.loc[
            best_idx,
            "lift_at_5pct",
        ]
    )

    print(
        "\n=== CURRENT VALIDATION CHAMPION ==="
    )

    print(
        f"Feature set: "
        f"{best_feature_set}"
    )

    print(
        f"PR-AUC: "
        f"{best_pr_auc:.6f}"
    )

    print(
        f"Lift@5%: "
        f"{best_lift:.4f}x"
    )

    # =========================================================
    # SAVE VALIDATION RESULTS
    # =========================================================

    ablation_df.to_csv(
        DATA_DIR
        / "weekly_feature_group_ablation.csv",
        index=False,
    )

    print(
        "\nFeature-group ablation complete."
    )

    print(
        "The final test set was NOT evaluated."
    )