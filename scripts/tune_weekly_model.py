import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from train_weekly_model import (
    DATA_DIR,
    LABEL_COL,
    FEATURE_COLS_CORE_PURCHASE_PRODUCT,
    TRAIN_CUTOFFS,
    VALIDATION_CUTOFFS,
    TEST_CUTOFFS,
    ranking_metrics_at_k,
)


# -----------------------------------------------------------------------------
# LightGBM hyperparameter tuning
# -----------------------------------------------------------------------------
#
# Feature selection and model-family selection are now frozen.
#
# The selected model uses 23 features:
#   - Core
#   - Purchase History
#   - Product/Category
#
# A 31-feature version that also included Engagement performed only
# slightly better on validation, so I kept the simpler 23-feature model.
#
# Using these same 23 features, untuned LightGBM also outperformed
# Logistic Regression on the Oct 16 validation period.
#
# This script changes only LightGBM hyperparameters.
# The feature set and temporal split stay fixed.
#
# Oct 24 stays reserved for final evaluation and is not scored here.


# -----------------------------------------------------------------------------
# Tuning candidates
# -----------------------------------------------------------------------------
#
# The search is intentionally small because there is only one main
# temporal validation period.
#
# Instead of trying a huge grid, each configuration tests a specific idea:
#   1. Shallower trees
#   2. More regularization through larger minimum leaf sizes
#   3. Slower learning with more boosting trees
#   4. Mild weighting of the positive class
#
# The baseline is the same LightGBM configuration used during feature
# ablation.

PARAM_GRID = [
    {
        "name": "baseline",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 20,
        "scale_pos_weight": 1,
    },
    {
        "name": "depth4_leaves15",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 20,
        "scale_pos_weight": 1,
    },
    {
        "name": "depth5_leaves20",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 5,
        "num_leaves": 20,
        "min_child_samples": 20,
        "scale_pos_weight": 1,
    },
    {
        "name": "minchild50",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 50,
        "scale_pos_weight": 1,
    },
    {
        "name": "minchild100",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 100,
        "scale_pos_weight": 1,
    },
    {
        "name": "300trees_lr03",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 20,
        "scale_pos_weight": 1,
    },
    {
        "name": "400trees_lr03",
        "n_estimators": 400,
        "learning_rate": 0.03,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 20,
        "scale_pos_weight": 1,
    },
    {
        "name": "depth4_lr03",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 20,
        "scale_pos_weight": 1,
    },
    {
        "name": "depth5_lr03",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 5,
        "num_leaves": 20,
        "min_child_samples": 20,
        "scale_pos_weight": 1,
    },
    {
        "name": "weight2",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 20,
        "scale_pos_weight": 2,
    },
    {
        "name": "weight3",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 20,
        "scale_pos_weight": 3,
    },
    {
        "name": "depth5_minchild50_weight2",
        "n_estimators": 250,
        "learning_rate": 0.04,
        "max_depth": 5,
        "num_leaves": 20,
        "min_child_samples": 50,
        "scale_pos_weight": 2,
    },
]


def evaluate_candidate(
    X_train,
    y_train,
    X_val,
    y_val,
    params,
):
    """
    Fit one LightGBM configuration and evaluate it on the
    temporal validation period.
    """

    model = LGBMClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        scale_pos_weight=params["scale_pos_weight"],
        random_state=38,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    # This is a ranking problem, so use the model scores directly.
    val_scores = model.predict_proba(X_val)[:, 1]

    metrics = {
        "name": params["name"],
        "n_estimators": params["n_estimators"],
        "learning_rate": params["learning_rate"],
        "max_depth": params["max_depth"],
        "num_leaves": params["num_leaves"],
        "min_child_samples": params["min_child_samples"],
        "scale_pos_weight": params["scale_pos_weight"],
        "average_precision": average_precision_score(
            y_val,
            val_scores,
        ),
        "roc_auc": roc_auc_score(
            y_val,
            val_scores,
        ),
    }

    metrics.update(
        ranking_metrics_at_k(
            y_val,
            val_scores,
            k=0.05,
        )
    )

    return metrics


def main():
    # Load the same user-cutoff feature table used during feature selection.
    df = pd.read_parquet(
        DATA_DIR / "user_features.parquet"
    )

    # Users without a previous purchase-containing session do not have
    # a defined average order value.
    #
    # Keep the same -1 sentinel used during feature selection so Stage 14
    # changes only hyperparameters.
    df["avg_order_value"] = (
        df["avg_order_value"].fillna(-1)
    )

    # Recreate the same purged temporal split used during model development.
    train_df = df[
        df["cutoff_date"].isin(TRAIN_CUTOFFS)
    ].copy()

    val_df = df[
        df["cutoff_date"].isin(VALIDATION_CUTOFFS)
    ].copy()

    # Keep the reserved test period separate.
    # Do not inspect its labels or score any model on it during tuning.
    test_df = df[
        df["cutoff_date"].isin(TEST_CUTOFFS)
    ].copy()

    print("\nTuning data")
    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print(f"Reserved test rows: {len(test_df):,}")

    print(
        f"Train positive rate: "
        f"{train_df[LABEL_COL].mean():.4%}"
    )

    print(
        f"Validation positive rate: "
        f"{val_df[LABEL_COL].mean():.4%}"
    )

    print(
        "Reserved Oct 24 test labels are not inspected during tuning."
    )

    # Use the frozen 23-feature specification for every tuning candidate.
    X_train = train_df[
        FEATURE_COLS_CORE_PURCHASE_PRODUCT
    ].copy()

    X_val = val_df[
        FEATURE_COLS_CORE_PURCHASE_PRODUCT
    ].copy()

    y_train = train_df[
        LABEL_COL
    ].astype(int)

    y_val = val_df[
        LABEL_COL
    ].astype(int)

    print(
        f"\nFrozen feature count: "
        f"{X_train.shape[1]}"
    )

    # Stop immediately if the frozen feature set changes by accident.
    assert X_train.shape[1] == 23
    assert X_val.shape[1] == 23

    print(
        f"\nTesting {len(PARAM_GRID)} "
        "predefined LightGBM configurations."
    )

    results = []

    for i, params in enumerate(
        PARAM_GRID,
        start=1,
    ):
        print(
            f"\n[{i}/{len(PARAM_GRID)}] "
            f"{params['name']}"
        )

        metrics = evaluate_candidate(
            X_train,
            y_train,
            X_val,
            y_val,
            params,
        )

        results.append(metrics)

        print(
            f"Average Precision: "
            f"{metrics['average_precision']:.6f}"
        )

        print(
            f"ROC-AUC: "
            f"{metrics['roc_auc']:.6f}"
        )

        print(
            f"Precision@5%: "
            f"{metrics['precision_at_5pct']:.6f}"
        )

        print(
            f"Recall@5%: "
            f"{metrics['recall_at_5pct']:.6f}"
        )

        print(
            f"Lift@5%: "
            f"{metrics['lift_at_5pct']:.4f}x"
        )

    # Average Precision was chosen before tuning as the primary metric.
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="average_precision",
        ascending=False,
    ).reset_index(drop=True)

    print(
        "\nTuning results sorted by Average Precision"
    )

    display_cols = [
        "name",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "num_leaves",
        "min_child_samples",
        "scale_pos_weight",
        "average_precision",
        "roc_auc",
        "precision_at_5pct",
        "recall_at_5pct",
        "lift_at_5pct",
    ]

    print(
        results_df[
            display_cols
        ].to_string(index=False)
    )

    # This is the highest-AP candidate, but do not automatically assume
    # that a tiny improvement is worth extra complexity.
    best_candidate = results_df.iloc[0]

    print(
        "\nHighest-AP validation candidate"
    )

    print(
        f"Configuration: "
        f"{best_candidate['name']}"
    )

    print(
        f"Average Precision: "
        f"{best_candidate['average_precision']:.6f}"
    )

    print(
        f"ROC-AUC: "
        f"{best_candidate['roc_auc']:.6f}"
    )

    print(
        f"Precision@5%: "
        f"{best_candidate['precision_at_5pct']:.6f}"
    )

    print(
        f"Recall@5%: "
        f"{best_candidate['recall_at_5pct']:.6f}"
    )

    print(
        f"Lift@5%: "
        f"{best_candidate['lift_at_5pct']:.4f}x"
    )

    print(
        "\nCandidate parameters"
    )

    print(
        f"n_estimators="
        f"{int(best_candidate['n_estimators'])}"
    )

    print(
        f"learning_rate="
        f"{best_candidate['learning_rate']}"
    )

    print(
        f"max_depth="
        f"{int(best_candidate['max_depth'])}"
    )

    print(
        f"num_leaves="
        f"{int(best_candidate['num_leaves'])}"
    )

    print(
        f"min_child_samples="
        f"{int(best_candidate['min_child_samples'])}"
    )

    print(
        f"scale_pos_weight="
        f"{best_candidate['scale_pos_weight']}"
    )

    # Compare against the untuned LightGBM baseline using the same
    # frozen 23 features.
    baseline_row = results_df[
        results_df["name"] == "baseline"
    ].iloc[0]

    ap_improvement = (
        (
            best_candidate["average_precision"]
            - baseline_row["average_precision"]
        )
        / baseline_row["average_precision"]
        * 100
    )

    print(
        "\nAverage Precision change versus "
        "untuned 23-feature baseline: "
        f"{ap_improvement:+.2f}%"
    )

    print(
        "\nReview the size of the improvement and the added complexity "
        "before freezing the final hyperparameters."
    )

    # Save the full tuning table for reproducibility.
    results_df.to_csv(
        DATA_DIR / "lightgbm_tuning_results.csv",
        index=False,
    )

    print(
        "\nSaved results to "
        "data/lightgbm_tuning_results.csv"
    )

    print(
        "The Oct 24 final test set was not evaluated."
    )


if __name__ == "__main__":
    main()

