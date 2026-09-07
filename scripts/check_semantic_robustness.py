from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from train_weekly_model import (
    DATA_DIR,
    LABEL_COL,
    FEATURE_COLS_FULL_PLUS_PRODUCT,
    FEATURE_COLS_FULL_PLUS_PRODUCT_SEMANTIC,
    fit_lightgbm_with_weight,
    ranking_metrics_at_k,
)


# =========================================================
# PURPOSE
# =========================================================
# Compare:
#
# Model A: 31 structured + product/category features
# Model B: same 31 features + 20 semantic features = 51
#
# across several rolling historical validation points.
#
# IMPORTANT:
# The Oct 24 final test cutoff is NEVER used here.
# =========================================================


# ---------------------------------------------------------
# Historical rolling-origin folds
# ---------------------------------------------------------
#
# Because the prediction horizon is 7 days, training
# cutoffs must end before the validation label window starts.
#
# Fold 1:
#   Train cutoff Oct 4
#   Its label window ends before Oct 12 validation begins.
#
# Fold 2:
#   Train cutoffs Oct 4, Oct 6
#   Latest training label window ends before Oct 14 begins.
#
# Fold 3:
#   Train cutoffs Oct 4, Oct 6, Oct 8
#   Latest training label window ends before Oct 16 begins.
#
# Oct 24 remains untouched final test data.
# ---------------------------------------------------------

ROBUSTNESS_FOLDS = [
    {
        "fold": "Fold 1",
        "train_cutoffs": [
            pd.Timestamp("2019-10-04", tz="UTC"),
        ],
        "validation_cutoff":
            pd.Timestamp("2019-10-12", tz="UTC"),
    },
    {
        "fold": "Fold 2",
        "train_cutoffs": [
            pd.Timestamp("2019-10-04", tz="UTC"),
            pd.Timestamp("2019-10-06", tz="UTC"),
        ],
        "validation_cutoff":
            pd.Timestamp("2019-10-14", tz="UTC"),
    },
    {
        "fold": "Fold 3",
        "train_cutoffs": [
            pd.Timestamp("2019-10-04", tz="UTC"),
            pd.Timestamp("2019-10-06", tz="UTC"),
            pd.Timestamp("2019-10-08", tz="UTC"),
        ],
        "validation_cutoff":
            pd.Timestamp("2019-10-16", tz="UTC"),
    },
]


def evaluate_feature_set(
    train_df,
    val_df,
    feature_cols,
):
    """
    Train LightGBM on one feature set and evaluate
    on the supplied temporal validation cutoff.
    """

    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()

    y_train = train_df[LABEL_COL].astype(int)
    y_val = val_df[LABEL_COL].astype(int)

    model = fit_lightgbm_with_weight(
        X_train,
        y_train,
        scale_pos_weight=1,
    )

    val_proba = model.predict_proba(
        X_val
    )[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(
            y_val,
            val_proba,
        ),
        "pr_auc": average_precision_score(
            y_val,
            val_proba,
        ),
    }

    metrics.update(
        ranking_metrics_at_k(
            y_val,
            val_proba,
            k=0.05,
        )
    )

    return metrics


def main():

    # =====================================================
    # LOAD FEATURE TABLE
    # =====================================================

    df = pd.read_parquet(
        DATA_DIR / "user_features.parquet"
    )

    # Same treatment as the main weekly model.
    df["avg_order_value"] = (
        df["avg_order_value"].fillna(-1)
    )

    results = []

    # =====================================================
    # RUN EACH TEMPORAL FOLD
    # =====================================================

    for fold_info in ROBUSTNESS_FOLDS:

        fold_name = fold_info["fold"]
        train_cutoffs = fold_info["train_cutoffs"]
        validation_cutoff = (
            fold_info["validation_cutoff"]
        )

        print("\n" + "=" * 70)
        print(fold_name)
        print("=" * 70)

        print(
            "Train cutoffs:",
            [
                str(c.date())
                for c in train_cutoffs
            ],
        )

        print(
            "Validation cutoff:",
            str(validation_cutoff.date()),
        )

        train_df = df[
            df["cutoff_date"].isin(
                train_cutoffs
            )
        ].copy()

        val_df = df[
            df["cutoff_date"]
            == validation_cutoff
        ].copy()

        print(
            f"Train rows: {len(train_df):,}"
        )

        print(
            f"Validation rows: {len(val_df):,}"
        )

        print(
            "Train positive rate: "
            f"{train_df[LABEL_COL].mean():.4%}"
        )

        print(
            "Validation positive rate: "
            f"{val_df[LABEL_COL].mean():.4%}"
        )

        # -------------------------------------------------
        # MODEL A: 31 FEATURES
        # -------------------------------------------------

        product_metrics = evaluate_feature_set(
            train_df,
            val_df,
            FEATURE_COLS_FULL_PLUS_PRODUCT,
        )

        # -------------------------------------------------
        # MODEL B: 51 FEATURES
        # -------------------------------------------------

        semantic_metrics = evaluate_feature_set(
            train_df,
            val_df,
            FEATURE_COLS_FULL_PLUS_PRODUCT_SEMANTIC,
        )

        product_pr = product_metrics["pr_auc"]
        semantic_pr = semantic_metrics["pr_auc"]

        pr_change_pct = (
            (semantic_pr - product_pr)
            / product_pr
            * 100
        )

        print("\n31-feature model")
        print(
            f"PR-AUC: "
            f"{product_pr:.6f}"
        )
        print(
            f"ROC-AUC: "
            f"{product_metrics['roc_auc']:.6f}"
        )
        print(
            f"Lift@5%: "
            f"{product_metrics['lift_at_5pct']:.4f}x"
        )

        print("\n51-feature semantic model")
        print(
            f"PR-AUC: "
            f"{semantic_pr:.6f}"
        )
        print(
            f"ROC-AUC: "
            f"{semantic_metrics['roc_auc']:.6f}"
        )
        print(
            f"Lift@5%: "
            f"{semantic_metrics['lift_at_5pct']:.4f}x"
        )

        print(
            "\nSemantic PR-AUC change: "
            f"{pr_change_pct:+.2f}%"
        )

        results.append({
            "fold": fold_name,
            "train_cutoffs": ", ".join(
                str(c.date())
                for c in train_cutoffs
            ),
            "validation_cutoff":
                str(validation_cutoff.date()),

            "product_pr_auc":
                product_metrics["pr_auc"],

            "semantic_pr_auc":
                semantic_metrics["pr_auc"],

            "semantic_pr_change_pct":
                pr_change_pct,

            "product_roc_auc":
                product_metrics["roc_auc"],

            "semantic_roc_auc":
                semantic_metrics["roc_auc"],

            "product_lift_5pct":
                product_metrics["lift_at_5pct"],

            "semantic_lift_5pct":
                semantic_metrics["lift_at_5pct"],
        })

    # =====================================================
    # SUMMARY
    # =====================================================

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("SEMANTIC FEATURE ROBUSTNESS SUMMARY")
    print("=" * 70)

    summary_cols = [
        "fold",
        "validation_cutoff",
        "product_pr_auc",
        "semantic_pr_auc",
        "semantic_pr_change_pct",
        "product_lift_5pct",
        "semantic_lift_5pct",
    ]

    print(
        results_df[
            summary_cols
        ].to_string(index=False)
    )

    wins = (
        results_df[
            "semantic_pr_change_pct"
        ] > 0
    ).sum()

    print(
        f"\nSemantic model improved PR-AUC "
        f"in {wins}/{len(results_df)} folds."
    )

    print(
        "\nAverage semantic PR-AUC change: "
        f"{results_df['semantic_pr_change_pct'].mean():+.2f}%"
    )

    # Save for documentation/reporting.
    results_df.to_csv(
        DATA_DIR
        / "semantic_temporal_robustness.csv",
        index=False,
    )

    print(
        "\nSaved results to "
        "data/semantic_temporal_robustness.csv"
    )

    print(
        "\nThe Oct 24 final test set was NOT evaluated."
    )


if __name__ == "__main__":
    main()