from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path("data")
FIGURE_DIR = Path("reports/figures")
TABLE_DIR = Path("reports/tables")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 1. Dataset event composition
# ---------------------------------------------------------

event_counts = pd.DataFrame(
    {
        "Event type": ["View", "Cart", "Purchase"],
        "Count": [3_281_506, 74_730, 60_051],
    }
)

event_counts["Percent"] = (
    event_counts["Count"] / event_counts["Count"].sum() * 100
)

event_counts.to_csv(
    TABLE_DIR / "event_composition.csv",
    index=False,
)

print("\nEvent composition")
print(event_counts.to_string(index=False))


plt.figure(figsize=(7, 4.5))

bars = plt.bar(
    event_counts["Event type"],
    event_counts["Count"],
)

plt.title("Event Distribution in the User-Level Sample")
plt.ylabel("Number of Events")
plt.xlabel("Event Type")

plt.ticklabel_format(
    style="plain",
    axis="y",
)

for bar, count in zip(
    bars,
    event_counts["Count"],
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count:,}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "event_distribution.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# 2. Dataset summary table
# ---------------------------------------------------------

dataset_summary = pd.DataFrame(
    {
        "Statistic": [
            "Events",
            "Unique users",
            "Unique sessions",
            "Views",
            "Carts",
            "Purchases",
            "User-cutoff observations",
            "Positive user-cutoff observations",
            "7-day purchase label rate",
        ],
        "Value": [
            "3,416,287",
            "241,783",
            "742,457",
            "3,281,506",
            "74,730",
            "60,051",
            "1,364,491",
            "49,249",
            "3.61%",
        ],
    }
)

dataset_summary.to_csv(
    TABLE_DIR / "dataset_summary.csv",
    index=False,
)

print("\nDataset summary")
print(dataset_summary.to_string(index=False))


# ---------------------------------------------------------
# 3. Temporal modeling split table
# ---------------------------------------------------------

split_summary = pd.DataFrame(
    {
        "Partition": [
            "Training",
            "Validation",
            "Final test",
        ],
        "Prediction cutoffs": [
            "Oct 4, Oct 6, Oct 8",
            "Oct 16",
            "Oct 24",
        ],
        "Rows": [
            172_750,
            142_808,
            198_599,
        ],
        "Positive rate": [
            0.052579,
            0.035775,
            None,
        ],
    }
)

split_summary["Positive rate"] = split_summary[
    "Positive rate"
].map(
    lambda x: (
        f"{x:.2%}"
        if pd.notna(x)
        else "Reserved"
    )
)

split_summary.to_csv(
    TABLE_DIR / "temporal_split_summary.csv",
    index=False,
)

print("\nTemporal split")
print(split_summary.to_string(index=False))


# ---------------------------------------------------------
# 4. Phase A feature-group ablation
# ---------------------------------------------------------

ablation = pd.read_csv(
    DATA_DIR / "phase_a_feature_ablation.csv"
)

# Put Core first so the figure reads naturally.
desired_order = [
    "Core",
    "Core + Semantic",
    "Core + Product/Category",
    "Core + Engagement",
    "Core + Purchase History",
]

ablation["model"] = pd.Categorical(
    ablation["model"],
    categories=desired_order,
    ordered=True,
)

ablation = (
    ablation
    .sort_values("model")
    .reset_index(drop=True)
)

ablation.to_csv(
    TABLE_DIR / "phase_a_ablation_report.csv",
    index=False,
)

print("\nPhase A ablation")
print(
    ablation[
        [
            "model",
            "n_features",
            "average_precision",
            "roc_auc",
            "precision_at_5pct",
            "recall_at_5pct",
            "lift_at_5pct",
        ]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# 5. Main report figure:
# Average Precision by feature group
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

bars = plt.barh(
    ablation["model"].astype(str),
    ablation["average_precision"],
)

plt.title(
    "Feature-Group Ablation on Temporal Validation Set"
)

plt.xlabel("Average Precision")
plt.ylabel("Feature Set")

plt.xlim(
    0,
    ablation["average_precision"].max() * 1.18,
)

for bar, value in zip(
    bars,
    ablation["average_precision"],
):
    plt.text(
        value + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "phase_a_average_precision.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# 6. Business-facing ablation figure:
# Lift at top 5%
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

bars = plt.barh(
    ablation["model"].astype(str),
    ablation["lift_at_5pct"],
)

plt.title(
    "Top-5% Targeting Lift by Feature Set"
)

plt.xlabel("Lift at Top 5%")
plt.ylabel("Feature Set")

plt.xlim(
    0,
    ablation["lift_at_5pct"].max() * 1.18,
)

for bar, value in zip(
    bars,
    ablation["lift_at_5pct"],
):
    plt.text(
        value + 0.08,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}x",
        va="center",
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "phase_a_lift_at_5pct.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# 7. Improvement over Core
# ---------------------------------------------------------

core_ap = float(
    ablation.loc[
        ablation["model"] == "Core",
        "average_precision",
    ].iloc[0]
)

ablation["AP improvement vs Core"] = (
    ablation["average_precision"] - core_ap
)

ablation["Relative AP improvement vs Core (%)"] = (
    ablation["AP improvement vs Core"]
    / core_ap
    * 100
)

improvement = ablation[
    ablation["model"] != "Core"
].copy()

improvement.to_csv(
    TABLE_DIR / "phase_a_improvement_over_core.csv",
    index=False,
)


plt.figure(figsize=(9, 5))

bars = plt.barh(
    improvement["model"].astype(str),
    improvement[
        "Relative AP improvement vs Core (%)"
    ],
)

plt.title(
    "Incremental Value of Feature Groups Beyond Core"
)

plt.xlabel(
    "Relative Improvement in Average Precision (%)"
)

plt.ylabel("Added Feature Group")

for bar, value in zip(
    bars,
    improvement[
        "Relative AP improvement vs Core (%)"
    ],
):
    plt.text(
        value + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f}%",
        va="center",
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "phase_a_relative_improvement.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------
# 8. Compact report table with formatted values
# ---------------------------------------------------------

report_ablation = ablation.copy()

report_ablation["Average Precision"] = (
    report_ablation["average_precision"]
    .map(lambda x: f"{x:.3f}")
)

report_ablation["ROC-AUC"] = (
    report_ablation["roc_auc"]
    .map(lambda x: f"{x:.3f}")
)

report_ablation["Precision@5%"] = (
    report_ablation["precision_at_5pct"]
    .map(lambda x: f"{x:.1%}")
)

report_ablation["Recall@5%"] = (
    report_ablation["recall_at_5pct"]
    .map(lambda x: f"{x:.1%}")
)

report_ablation["Lift@5%"] = (
    report_ablation["lift_at_5pct"]
    .map(lambda x: f"{x:.2f}x")
)

report_ablation = report_ablation[
    [
        "model",
        "n_features",
        "Average Precision",
        "ROC-AUC",
        "Precision@5%",
        "Recall@5%",
        "Lift@5%",
    ]
]

report_ablation.columns = [
    "Feature Set",
    "Features",
    "Average Precision",
    "ROC-AUC",
    "Precision@5%",
    "Recall@5%",
    "Lift@5%",
]

report_ablation.to_csv(
    TABLE_DIR / "phase_a_ablation_formatted.csv",
    index=False,
)

print("\nFormatted report table")
print(report_ablation.to_string(index=False))


print(
    "\nDone. Figures saved in reports/figures "
    "and tables saved in reports/tables."
)