from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# -----------------------------------------------------------------------------
# Feature groups
# -----------------------------------------------------------------------------

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

FEATURE_COLS_CORE_PURCHASE_PRODUCT = list(
    dict.fromkeys(
        FEATURE_COLS_CORE
        + FEATURE_COLS_PURCHASE_HISTORY
        + FEATURE_COLS_PRODUCT_CATEGORY
    )
)

FEATURE_COLS_SEMANTIC = [
    f"user_text_dim_{i}"
    for i in range(20)
]


# Common combinations used in model development.
FEATURE_COLS_CORE_ENGAGEMENT = (
    FEATURE_COLS_CORE
    + FEATURE_COLS_ENGAGEMENT_ONLY
)

FEATURE_COLS_CORE_PURCHASE = (
    FEATURE_COLS_CORE
    + FEATURE_COLS_PURCHASE_HISTORY
)

FEATURE_COLS_CORE_PRODUCT = (
    FEATURE_COLS_CORE
    + FEATURE_COLS_PRODUCT_CATEGORY
)

FEATURE_COLS_CORE_SEMANTIC = (
    FEATURE_COLS_CORE
    + FEATURE_COLS_SEMANTIC
)

LABEL_COL = "will_purchase_next_7d"


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def ranking_metrics_at_k(
    y_true,
    y_score,
    k: float = 0.05,
) -> dict:
    """
    Evaluate the highest-scoring k fraction of users.

    Example:
        k=0.05 evaluates the top 5% of users ranked by predicted
        purchase propensity.
    """

    results = pd.DataFrame(
        {
            "actual": pd.Series(y_true).reset_index(drop=True),
            "score": y_score,
        }
    )

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

    pct = int(k * 100)

    return {
        f"precision_at_{pct}pct": precision_at_k,
        f"recall_at_{pct}pct": recall_at_k,
        f"lift_at_{pct}pct": lift_at_k,
    }


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

def fit_logistic_regression(
    X_train,
    y_train,
):
    """
    Fit the Logistic Regression baseline.

    All preprocessing is learned from the training data only.
    """

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight=None,
                    random_state=38,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def fit_lightgbm_baseline(
    X_train,
    y_train,
    scale_pos_weight: float = 1.0,
):
    """
    Fit the fixed LightGBM baseline used during feature ablation.

    Hyperparameters stay fixed during ablation so performance changes
    can be attributed to feature information rather than tuning changes.
    """

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        scale_pos_weight=scale_pos_weight,
        random_state=38,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def evaluate_ranking_model(
    model,
    X,
    y,
    model_name: str,
) -> dict:
    """
    Evaluate a model using ranking-oriented validation metrics.
    """

    scores = model.predict_proba(X)[:, 1]

    metrics = {
        "model": model_name,
        "roc_auc": roc_auc_score(y, scores),
        "average_precision": average_precision_score(
            y,
            scores,
        ),
    }

    metrics.update(
        ranking_metrics_at_k(
            y,
            scores,
            k=0.05,
        )
    )

    return metrics


# -----------------------------------------------------------------------------
# Feature ablation
# -----------------------------------------------------------------------------

def run_feature_ablation(
    train_df,
    val_df,
    y_train,
    y_val,
    feature_sets: dict,
) -> pd.DataFrame:
    """
    Run feature-group ablation using the same fixed LightGBM baseline.
    """

    results = []

    for model_name, feature_cols in feature_sets.items():
        X_train = train_df[feature_cols].copy()
        X_val = val_df[feature_cols].copy()

        model = fit_lightgbm_baseline(
            X_train,
            y_train,
            scale_pos_weight=1.0,
        )

        metrics = evaluate_ranking_model(
            model=model,
            X=X_val,
            y=y_val,
            model_name=model_name,
        )

        metrics["n_features"] = len(feature_cols)
        results.append(metrics)

    results_df = pd.DataFrame(results)

    ordered_cols = [
        "model",
        "n_features",
        "average_precision",
        "roc_auc",
        "precision_at_5pct",
        "recall_at_5pct",
        "lift_at_5pct",
    ]

    return results_df[ordered_cols]


# -----------------------------------------------------------------------------
# Temporal split
# -----------------------------------------------------------------------------

ALL_CUTOFFS = pd.date_range(
    "2019-10-04",
    "2019-10-24",
    freq="2D",
    tz="UTC",
)

# Purged / embargoed temporal split.
#
# Train label windows:
#   Oct 4  -> [Oct 4, Oct 11)
#   Oct 6  -> [Oct 6, Oct 13)
#   Oct 8  -> [Oct 8, Oct 15)
#
# Validation label window:
#   Oct 16 -> [Oct 16, Oct 23)
#
# Test label window:
#   Oct 24 -> [Oct 24, Oct 31)
#
# Intermediate cutoffs are intentionally unused so target windows do
# not overlap across train, validation, and test partitions.

TRAIN_CUTOFFS = [
    ALL_CUTOFFS[0],  # Oct 4
    ALL_CUTOFFS[1],  # Oct 6
    ALL_CUTOFFS[2],  # Oct 8
]

VALIDATION_CUTOFFS = [
    ALL_CUTOFFS[6],  # Oct 16
]

TEST_CUTOFFS = [
    ALL_CUTOFFS[10],  # Oct 24
]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

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

    # Load the user-cutoff feature table.
    df = pd.read_parquet(
        DATA_DIR / "user_features.parquet"
    )

    # Users with no previous purchase-containing sessions have undefined
    # average order value. Keep a sentinel value for model-development
    # consistency. We will revisit missing-value handling before final tuning.
    df["avg_order_value"] = (
        df["avg_order_value"].fillna(-1)
    )

    # Explicit cutoff membership preserves the embargo dates.
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

    y_train = train_df[LABEL_COL].astype(int)
    y_val = val_df[LABEL_COL].astype(int)

    print("\nData split")
    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print(f"Reserved test rows: {len(test_df):,}")
    print(
        f"Train positive rate: "
        f"{y_train.mean():.4%}"
    )
    print(
        f"Validation positive rate: "
        f"{y_val.mean():.4%}"
    )
    print(
        "Reserved Oct 24 test labels are not inspected "
        "during model development."
    )

    # -------------------------------------------------------------------------
    # Phase A: Marginal feature-group value
    # -------------------------------------------------------------------------
    #
    # Each optional feature family is added separately to Core.
    # This answers whether each information source adds signal beyond Core
    # without assuming Engagement must be included before Purchase History,
    # Product/Category, or Semantic features are tested.

    PHASE_A_FEATURE_SETS = {
        "Core": FEATURE_COLS_CORE,
        "Core + Engagement": FEATURE_COLS_CORE_ENGAGEMENT,
        "Core + Purchase History": FEATURE_COLS_CORE_PURCHASE,
        "Core + Product/Category": FEATURE_COLS_CORE_PRODUCT,
        "Core + Semantic": FEATURE_COLS_CORE_SEMANTIC,
    }

    print("\nPhase A: Marginal feature-group ablation")

    phase_a_results = run_feature_ablation(
        train_df=train_df,
        val_df=val_df,
        y_train=y_train,
        y_val=y_val,
        feature_sets=PHASE_A_FEATURE_SETS,
    )

    phase_a_results = phase_a_results.sort_values(
        "average_precision",
        ascending=False,
    ).reset_index(drop=True)

    print(
        phase_a_results.to_string(
            index=False
        )
    )

    phase_a_results.to_csv(
        DATA_DIR / "phase_a_feature_ablation.csv",
        index=False,
    )

    best_phase_a = phase_a_results.iloc[0]

    print("\nBest Phase A candidate")
    print(f"Feature set: {best_phase_a['model']}")
    print(
        f"Average Precision: "
        f"{best_phase_a['average_precision']:.6f}"
    )
    print(
        f"Lift@5%: "
        f"{best_phase_a['lift_at_5pct']:.4f}x"
    )


    print(
        "\nPhase A complete. "
    )
    print(
        "The final Oct 24 test set was NOT evaluated."
    )


    # -----------------------------------------------------------------------------
    # Phase B feature combinations
    # -----------------------------------------------------------------------------

    # Phase A showed that Purchase History had the strongest
    # marginal contribution beyond the Core features.
    #
    # Phase B therefore keeps Core + Purchase History as the
    # common reference and adds each remaining feature group
    # separately.
    #
    # This tests conditional value:
    # Does the group still add useful information once purchase
    # history is already known?


    def combine_features(*groups):
        """
        Combine feature lists while preserving their original order
        and avoiding duplicate feature names.
        """
        combined = []

        for group in groups:
            for feature in group:
                if feature not in combined:
                    combined.append(feature)

        return combined


    FEATURE_COLS_CORE_PURCHASE_ENGAGEMENT = combine_features(
        FEATURE_COLS_CORE,
        FEATURE_COLS_PURCHASE_HISTORY,
        FEATURE_COLS_ENGAGEMENT_ONLY,
    )

    FEATURE_COLS_CORE_PURCHASE_SEMANTIC = combine_features(
        FEATURE_COLS_CORE,
        FEATURE_COLS_PURCHASE_HISTORY,
        FEATURE_COLS_SEMANTIC,
    )

    # After seeing phase B results, want to test core + purchase history + engagement + product/category 
    FEATURE_COLS_CORE_PURCHASE_ENGAGEMENT_PRODUCT = (
        FEATURE_COLS_CORE
        + FEATURE_COLS_PURCHASE_HISTORY
        + FEATURE_COLS_ENGAGEMENT_ONLY
        + FEATURE_COLS_PRODUCT_CATEGORY
    )


    # -----------------------------------------------------------------------------
    # Phase B: Conditional feature-group ablation
    # -----------------------------------------------------------------------------

    PHASE_B_FEATURE_SETS = {
        "Core + Purchase History": FEATURE_COLS_CORE_PURCHASE,
        "Core + Purchase + Engagement":
            FEATURE_COLS_CORE_PURCHASE_ENGAGEMENT,
        "Core + Purchase + Product/Category":
            FEATURE_COLS_CORE_PURCHASE_PRODUCT,
        "Core + Purchase + Semantic":
            FEATURE_COLS_CORE_PURCHASE_SEMANTIC,
        "Core + Purchase + Engagement + Product/Category":
            FEATURE_COLS_CORE_PURCHASE_ENGAGEMENT_PRODUCT,
    }

    print("\nPhase B: Conditional feature-group ablation")

    phase_b_results = run_feature_ablation(
        train_df=train_df,
        val_df=val_df,
        y_train=y_train,
        y_val=y_val,
        feature_sets=PHASE_B_FEATURE_SETS,
    )

    phase_b_results = phase_b_results.sort_values(
        "average_precision",
        ascending=False,
    ).reset_index(drop=True)

    print(
        phase_b_results.to_string(
            index=False
        )
    )

    phase_b_results.to_csv(
        DATA_DIR / "phase_b_feature_ablation.csv",
        index=False,
    )

    best_phase_b = phase_b_results.iloc[0]

    print("\nBest Phase B candidate")

    print(
        f"Feature set: {best_phase_b['model']}"
    )

    print(
        "Average Precision: "
        f"{best_phase_b['average_precision']:.6f}"
    )

    print(
        "Lift@5%: "
        f"{best_phase_b['lift_at_5pct']:.4f}x"
    )

    print(
        "\nPhase B complete."
    )

    print(
        "Review phase_b_feature_ablation.csv before "
        "testing any larger combined feature set."
    )

    print(
        "Do NOT run hyperparameter tuning yet."
    )

    print(
        "The final Oct 24 test set was NOT evaluated."
    )

    # -------------------------------------------------------------------------
    # Baseline model comparison
    # -------------------------------------------------------------------------
    #
    # Compare Logistic Regression and the untuned LightGBM baseline using
    # the same frozen 23-feature specification and the same temporal
    # train/validation split.
    #
    # This determines whether LightGBM provides enough incremental
    # predictive value to justify its additional model complexity.
    #
    # The Oct 24 test set remains untouched.

    FINAL_FEATURE_COLS = FEATURE_COLS_CORE_PURCHASE_PRODUCT

    X_train_final = train_df[FINAL_FEATURE_COLS].copy()
    X_val_final = val_df[FINAL_FEATURE_COLS].copy()

    print("\nStage 13.5: Baseline model comparison")
    print(f"Frozen feature count: {len(FINAL_FEATURE_COLS)}")


    # Logistic Regression baseline
    logistic_model = fit_logistic_regression(
        X_train_final,
        y_train,
    )

    logistic_metrics = evaluate_ranking_model(
        model=logistic_model,
        X=X_val_final,
        y=y_val,
        model_name="Logistic Regression",
    )


    # Untuned LightGBM baseline
    lightgbm_model = fit_lightgbm_baseline(
        X_train_final,
        y_train,
        scale_pos_weight=1.0,
    )

    lightgbm_metrics = evaluate_ranking_model(
        model=lightgbm_model,
        X=X_val_final,
        y=y_val,
        model_name="LightGBM",
    )


    baseline_results = pd.DataFrame(
        [
            logistic_metrics,
            lightgbm_metrics,
        ]
    )

    baseline_results["n_features"] = len(FINAL_FEATURE_COLS)

    baseline_results = baseline_results[
        [
            "model",
            "n_features",
            "average_precision",
            "roc_auc",
            "precision_at_5pct",
            "recall_at_5pct",
            "lift_at_5pct",
        ]
    ]

    baseline_results = baseline_results.sort_values(
        "average_precision",
        ascending=False,
    ).reset_index(drop=True)

    print("\nLogistic Regression vs LightGBM")
    print(
        baseline_results.to_string(
            index=False
        )
    )

    baseline_results.to_csv(
        DATA_DIR / "baseline_model_comparison.csv",
        index=False,
    )

    print(
        "\nBaseline comparison complete."
    )
    print(
        "The final Oct 24 test set was NOT evaluated."
    )