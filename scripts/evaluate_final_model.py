import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from train_weekly_model import (
    DATA_DIR,
    FEATURE_COLS_CORE_PURCHASE_PRODUCT,
    LABEL_COL,
    TEST_CUTOFFS,
    TRAIN_CUTOFFS,
    VALIDATION_CUTOFFS,
    ranking_metrics_at_k,
)

# -----------------------------------------------------------------------------
# Final held-out test evaluation
# -----------------------------------------------------------------------------
#
# Model development is complete before this script is run.
#
# The feature set, preprocessing, model family, and hyperparameters were
# selected using only the training and validation periods.
#
# Oct 24 is now opened once for final out-of-time evaluation.
# No model decisions should be changed based on these test results.


FINAL_FEATURE_COLS = FEATURE_COLS_CORE_PURCHASE_PRODUCT


FINAL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 4,
    "num_leaves": 15,
    "min_child_samples": 20,
    "scale_pos_weight": 1,
    "random_state": 38,
    "verbose": -1,
}


def main():
    df = pd.read_parquet(
        DATA_DIR / "user_features.parquet"
    )

    # Keep preprocessing identical to model development.
    df["avg_order_value"] = (
        df["avg_order_value"].fillna(-1)
    )

    # Oct 16 can now be added to training because model development
    # is finished and its full seven-day outcome is known before Oct 24.
    final_train_cutoffs = (
        list(TRAIN_CUTOFFS)
        + list(VALIDATION_CUTOFFS)
    )

    final_train_df = df[
        df["cutoff_date"].isin(final_train_cutoffs)
    ].copy()

    test_df = df[
        df["cutoff_date"].isin(TEST_CUTOFFS)
    ].copy()

    print("\nStage 15: Final held-out test evaluation")

    print(
        f"Final training rows: "
        f"{len(final_train_df):,}"
    )

    print(
        f"Final test rows: "
        f"{len(test_df):,}"
    )

    print(
        "Final training cutoffs:"
    )

    for cutoff in sorted(
        final_train_df["cutoff_date"].unique()
    ):
        print(f"  {cutoff}")

    print(
        "\nFinal test cutoff:"
    )

    for cutoff in sorted(
        test_df["cutoff_date"].unique()
    ):
        print(f"  {cutoff}")

    X_train = final_train_df[
        FINAL_FEATURE_COLS
    ].copy()

    y_train = final_train_df[
        LABEL_COL
    ].astype(int)

    X_test = test_df[
        FINAL_FEATURE_COLS
    ].copy()

    y_test = test_df[
        LABEL_COL
    ].astype(int)

    assert X_train.shape[1] == 23
    assert X_test.shape[1] == 23

    print(
        f"\nFrozen feature count: "
        f"{X_train.shape[1]}"
    )

    print(
        f"Final training positive rate: "
        f"{y_train.mean():.4%}"
    )

    # Test labels are intentionally opened here for the first
    # and only final evaluation.
    print(
        f"Final test positive rate: "
        f"{y_test.mean():.4%}"
    )

    model = LGBMClassifier(
        **FINAL_PARAMS
    )

    model.fit(
        X_train,
        y_train,
    )

    test_scores = model.predict_proba(
        X_test
    )[:, 1]

    average_precision = (
        average_precision_score(
            y_test,
            test_scores,
        )
    )

    roc_auc = roc_auc_score(
        y_test,
        test_scores,
    )

    print(
        "\nFinal test performance"
    )

    print(
        f"Average Precision: "
        f"{average_precision:.6f}"
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc:.6f}"
    )

    results = {
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "test_positive_rate": y_test.mean(),
    }

    # -------------------------------------------------------------------------
# Business targeting curve
# -------------------------------------------------------------------------
#
# The model is already frozen.
# These targeting depths are only used to describe how the final model
# behaves at different campaign sizes.
#
# They are not used to select or tune the model.

    targeting_depths = [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.10,
        0.15,
        0.20,
    ]

    targeting_rows = []

    for k in targeting_depths:
        metrics = ranking_metrics_at_k(
            y_test,
            test_scores,
            k=k,
        )

        pct = int(k * 100)

        precision = metrics[
            f"precision_at_{pct}pct"
        ]

        recall = metrics[
            f"recall_at_{pct}pct"
        ]

        lift = metrics[
            f"lift_at_{pct}pct"
        ]

        targeting_rows.append(
            {
                "targeting_depth": k,
                "targeting_pct": pct,
                "precision": precision,
                "recall": recall,
                "lift": lift,
            }
        )

        # Keep the main console output focused on the
        # three campaign sizes discussed in the report.
        if k in [0.01, 0.05, 0.10]:
            print(
                f"\nTop {pct}% targeting"
            )

            print(
                f"Precision@{pct}%: "
                f"{precision:.6f}"
            )

            print(
                f"Recall@{pct}%: "
                f"{recall:.6f}"
            )

            print(
                f"Lift@{pct}%: "
                f"{lift:.4f}x"
            )

            results.update(metrics)


    # Save the full targeting curve data.
    targeting_df = pd.DataFrame(
        targeting_rows
    )

    targeting_df.to_csv(
        DATA_DIR / "targeting_curve.csv",
        index=False,
    )


    # -------------------------------------------------------------------------
    # Targeting curve figure
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        targeting_df["targeting_pct"],
        targeting_df["lift"],
        marker="o",
    )

    ax.axhline(
        y=1,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Customers Targeted (%)"
    )

    ax.set_ylabel(
        "Lift Over Population Purchase Rate"
    )

    ax.set_title(
        "Purchase Lift by Campaign Targeting Depth"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    figure_path = (
        DATA_DIR
        / "targeting_lift_curve.png"
    )

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "\nSaved targeting curve data to "
        "data/targeting_curve.csv"
    )

    print(
        "Saved targeting curve figure to "
        "data/targeting_lift_curve.png"
    )

    results_df = pd.DataFrame(
        [results]
    )

    results_df.to_csv(
        DATA_DIR / "final_test_results.csv",
        index=False,
    )

    print(
        "\nSaved final results to "
        "data/final_test_results.csv"
    )

    print(
        "\nFinal test evaluation is complete."
    )

    print(
        "Do not change the model based on these test results."
    )


if __name__ == "__main__":
    main()