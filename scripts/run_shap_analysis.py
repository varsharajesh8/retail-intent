import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from train_weekly_model import (
    DATA_DIR,
    FEATURE_COLS_CORE_PURCHASE_PRODUCT,
    LABEL_COL,
    TEST_CUTOFFS,
    TRAIN_CUTOFFS,
    VALIDATION_CUTOFFS,
)

OUTPUT_DIR = DATA_DIR / "shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 38
SHAP_SAMPLE_SIZE = 5000


def main():
    df = pd.read_parquet(DATA_DIR / "user_features.parquet")

    feature_cols = FEATURE_COLS_CORE_PURCHASE_PRODUCT
    assert len(feature_cols) == 23

    train_cutoffs = TRAIN_CUTOFFS + VALIDATION_CUTOFFS

    train_df = df[df["cutoff_date"].isin(train_cutoffs)].copy()
    test_df = df[df["cutoff_date"].isin(TEST_CUTOFFS)].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[LABEL_COL]

    X_test = test_df[feature_cols]

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        min_child_samples=20,
        scale_pos_weight=1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    model.fit(X_train, y_train)

    sample_size = min(SHAP_SAMPLE_SIZE, len(X_test))

    X_sample = X_test.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_values = np.asarray(shap_values)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    importance_df.to_csv(
        OUTPUT_DIR / "shap_global_importance.csv",
        index=False,
    )

    print("\nTop SHAP features:")
    print(importance_df.head(15).to_string(index=False))

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "shap_beeswarm.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "shap_global_bar.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    top_features = importance_df.head(4)["feature"].tolist()

    for feature in top_features:
        shap.dependence_plot(
            feature,
            shap_values,
            X_sample,
            interaction_index=None,
            show=False,
        )

        safe_name = feature.replace("/", "_").replace(" ", "_")

        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"shap_scatter_{safe_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    probabilities = model.predict_proba(X_sample)[:, 1]

    high_idx = int(np.argmax(probabilities))
    median_idx = int(
        np.argsort(probabilities)[len(probabilities) // 2]
    )

    base_value = explainer.expected_value

    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.asarray(base_value).reshape(-1)[-1]

    high_explanation = shap.Explanation(
        values=shap_values[high_idx],
        base_values=base_value,
        data=X_sample.iloc[high_idx].values,
        feature_names=feature_cols,
    )

    shap.plots.waterfall(
        high_explanation,
        max_display=12,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "shap_waterfall_high_score.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    median_explanation = shap.Explanation(
        values=shap_values[median_idx],
        base_values=base_value,
        data=X_sample.iloc[median_idx].values,
        feature_names=feature_cols,
    )

    shap.plots.waterfall(
        median_explanation,
        max_display=12,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "shap_waterfall_median_score.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nSHAP outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()