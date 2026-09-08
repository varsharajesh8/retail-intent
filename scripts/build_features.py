from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_PARQUET = DATA_DIR / "events_sample.parquet"
USER_FEATURES_OUT = DATA_DIR / "user_features.parquet"


def load_and_clean(path: Path) -> pd.DataFrame:
    """Load and clean the data from the parquet file."""
    df = pd.read_parquet(path)

    # Remove exact duplicate event records identified during EDA.
    # Similar events at different timestamps are retained as real behavior.
    df = df.drop_duplicates().copy()

    # Missingness treatment: diagnosed in EDA notebook, missingness is systematic, not random, so we keep it as a signal rather than dropping rows
    for col in ["category_code", "brand"]:
        # preserves informative signal
        df[f"{col}_missing"] = df[col].isna()
        # makes column usable for grouping and encoding (rather than NaN)
        df[col] = df[col].fillna("unknown")

    return df

# Build user-level behavioral features at multiple prediction cutoffs.
# Candidate cutoffs are generated every 2 days from October 4 through
# October 24. Train, validation, and test cutoffs are selected later
# using a purged temporal split so their 7-day outcome windows do not overlap.
CUTOFF_DATES = pd.date_range("2019-10-04", "2019-10-24", freq="2D", tz="UTC")
LOOKAHEAD_DAYS = 7

def build_recent_behavior_features(
    history_window: pd.DataFrame,
    suffix: str,
) -> pd.DataFrame:
    """
    Aggregate recent user behavior within one pre-cutoff window.
    Intentionally not counting purchases. Testing recent engagement/funnel behavior separate from historical purchase. 
    """

    if history_window.empty:
        return pd.DataFrame(
            columns=[
                f"events_{suffix}",
                f"views_{suffix}",
                f"carts_{suffix}",
                f"sessions_{suffix}",
            ]
        )

    is_view = history_window["event_type"] == "view"
    is_cart = history_window["event_type"] == "cart"

    g = history_window.groupby("user_id")

    recent = pd.DataFrame({
        f"events_{suffix}": g.size(),
        f"views_{suffix}": is_view.groupby(
            history_window["user_id"]
        ).sum(),
        f"carts_{suffix}": is_cart.groupby(
            history_window["user_id"]
        ).sum(),
        f"sessions_{suffix}": g["user_session"].nunique(),
    })

    return recent

def build_user_features_for_cutoff(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Build RFM features as of cutoff, and a label for whether the user purchased in the next LOOKAHEAD_DAYS."""
    # strict time split: history is all events before cutoff, future is all events in the next LOOKAHEAD_DAYS
    history = df[df["event_time"] < cutoff]
    future = df[(df["event_time"] >= cutoff) & (df["event_time"] < cutoff + pd.Timedelta(days=LOOKAHEAD_DAYS))]

    # Recent pre-cutoff behavior windows.
    # Every window ends strictly at the prediction cutoff,
    # so no future information is used.
    history_1d = history[
        history["event_time"] >= cutoff - pd.Timedelta(days=1)
    ]

    history_3d = history[
        history["event_time"] >= cutoff - pd.Timedelta(days=3)
    ]

    history_7d = history[
        history["event_time"] >= cutoff - pd.Timedelta(days=7)
    ]

    # compute purchase condition once (vectorized) for efficiency, rather than repeatedly in groupby
    is_purchase = history["event_type"] == "purchase"
    purchase_rows = history[is_purchase]

    g = history.groupby("user_id") # all events per user
    g_purchase = purchase_rows.groupby("user_id") # only purchase events per user

    is_view = history["event_type"] == "view"
    is_cart = history["event_type"] == "cart"

    # View-only history for product/category/price behavior.
    historical_views = history[
        history["event_type"] == "view"
    ].copy()


     # =========================================================
    # SEMANTIC CATEGORY PREFERENCE FEATURES
    # =========================================================

    text_feature_path = (
        DATA_DIR / "category_text_features.parquet"
    )

    text_features = pd.read_parquet(
        text_feature_path
    )

    text_cols = [
        c for c in text_features.columns
        if c.startswith("text_dim_")
    ]

    # Attach semantic category vectors to historical views.
    historical_views_semantic = historical_views.merge(
        text_features,
        on="category_code",
        how="left",
    )

    # Average semantic vectors across all categories
    # historically viewed by each user.
    user_text_features = (
        historical_views_semantic
        .groupby("user_id")[text_cols]
        .mean()
        .rename(
            columns={
                col: f"user_{col}"
                for col in text_cols
            }
        )
    )


    g_views = historical_views.groupby("user_id")

    # RFM features: recency, frequency, monetary value, plus some additional behavioral features
    features = pd.DataFrame({
        "recency_days":
            (cutoff - g["event_time"].max()).dt.total_seconds() / 86400,

        "days_since_first_seen":
            (cutoff - g["event_time"].min()).dt.total_seconds() / 86400,

        "n_sessions":
            g["user_session"].nunique(),

        "n_events":
            g.size(),

        "n_views":
            is_view.groupby(history["user_id"]).sum(),

        "n_carts":
            is_cart.groupby(history["user_id"]).sum(),

        "n_purchase_events":
            is_purchase.groupby(history["user_id"]).sum(),

        "n_orders":
            g_purchase["user_session"].nunique(),

        "total_spend":
            g_purchase["price"].sum(),

        "avg_price_viewed":
            g_views["price"].mean(),

        "unique_products":
            g["product_id"].nunique(),

        "unique_categories":
            g["category_id"].nunique(),
    })

    features = features.join(
        user_text_features,
        how="left",
    )
    
    user_text_cols = [
        f"user_text_dim_{i}"
        for i in range(20)
    ]

    features[user_text_cols] = (
        features[user_text_cols]
        .fillna(0) # user with no historical category views has no semantic preference signal
    )

    # g_purchase only contains users who made at least one purchase, so we need to fill NaN for users with no purchases (instead of NaN)
    features["n_orders"] = features["n_orders"].fillna(0)
    features["total_spend"] = features["total_spend"].fillna(0)
    # Average order value is undefined for users with no orders, so we replace 0 with NaN to avoid misleadingly showing $0.00 as their average order value
    features["avg_order_value"] = features["total_spend"] / features["n_orders"].replace(0, np.nan)


    # Engagement and funnel features: 
    features["events_per_session"] = (
        features["n_events"]
        / features["n_sessions"].replace(0, np.nan)
    )

    features["views_per_session"] = (
        features["n_views"]
        / features["n_sessions"].replace(0, np.nan)
    )

    features["carts_per_session"] = (
        features["n_carts"]
        / features["n_sessions"].replace(0, np.nan)
    )

    features["cart_view_ratio"] = (
        features["n_carts"]
        / features["n_views"].replace(0, np.nan)
    )

    features["historical_conversion_rate"] = (
        features["n_orders"]
        / features["n_sessions"].replace(0, np.nan)
    )

    # Ratios are 0 when there is no corresponding observed activity.
    ratio_cols = [
        "events_per_session",
        "views_per_session",
        "carts_per_session",
        "cart_view_ratio",
        "historical_conversion_rate",
    ]

    features[ratio_cols] = (
        features[ratio_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    last_purchase_time = (
        purchase_rows
        .groupby("user_id")["event_time"]
        .max()
    )

    features["days_since_last_purchase"] = (
        (
            cutoff
            - last_purchase_time.reindex(features.index)
        )
        .dt.total_seconds()
        / 86400
    ) # keeping NaN since it means customer has never previously purchased, while 0 would mean purchased at cutoff

    # =========================================================
    # Product / category behavior features
    # =========================================================

    # Distinct products/categories specifically among views.
    unique_products_viewed = (
        g_views["product_id"]
        .nunique()
        .rename("unique_products_viewed")
    )

    unique_categories_viewed = (
        g_views["category_id"]
        .nunique()
        .rename("unique_categories_viewed")
    )

    features = features.merge(
        unique_products_viewed,
        left_index=True,
        right_index=True,
        how="left",
    )

    features = features.merge(
        unique_categories_viewed,
        left_index=True,
        right_index=True,
        how="left",
    )

    # Number of distinct brands viewed.
    unique_brands = (
        g_views["brand"]
        .nunique()
        .rename("unique_brands")
    )

    features = features.merge(
        unique_brands,
        left_index=True,
        right_index=True,
        how="left",
    )


    # ---------------------------------------------------------
    # Maximum views of the same product
    # ---------------------------------------------------------

    product_view_counts = (
        historical_views
        .groupby(["user_id", "product_id"])
        .size()
        .rename("product_view_count")
        .reset_index()
    )

    max_views_same_product = (
        product_view_counts
        .groupby("user_id")["product_view_count"]
        .max()
        .rename("max_views_same_product")
    )

    features = features.merge(
        max_views_same_product,
        left_index=True,
        right_index=True,
        how="left",
    )


    # ---------------------------------------------------------
    # Price behavior
    # ---------------------------------------------------------

    price_features = (
        g_views["price"]
        .agg(
            min_price_viewed="min",
            median_price_viewed="median",
            max_price_viewed="max",
            std_price_viewed="std",
        )
    )

    features = features.merge(
        price_features,
        left_index=True,
        right_index=True,
        how="left",
    )

    features["price_range_viewed"] = (
        features["max_price_viewed"]
        - features["min_price_viewed"]
    )


    # ---------------------------------------------------------
    # Repeat interest and diversity
    # ---------------------------------------------------------

    features["repeat_view_ratio"] = (
        1
        - (
            features["unique_products_viewed"]
            / features["n_views"].replace(0, np.nan)
        )
    )

    features["category_diversity_ratio"] = (
        features["unique_categories_viewed"]
        / features["n_views"].replace(0, np.nan)
    )

    features["brand_diversity_ratio"] = (
        features["unique_brands"]
        / features["n_views"].replace(0, np.nan)
    )


    # ---------------------------------------------------------
    # Clean count / ratio features
    # ---------------------------------------------------------

    features[
        [
            "unique_brands",
            "max_views_same_product",
            "unique_products_viewed",
            "unique_categories_viewed",
        ]
    ] = features[
        [
            "unique_brands",
            "max_views_same_product",
            "unique_products_viewed",
            "unique_categories_viewed",
        ]
    ].fillna(0)


    product_ratio_cols = [
        "repeat_view_ratio",
        "category_diversity_ratio",
        "brand_diversity_ratio",
    ]

    features[product_ratio_cols] = (
        features[product_ratio_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # calculate recent behavior features for each window, and merge them into the main features DataFrame
    recent_1d = build_recent_behavior_features(
        history_1d,
        "1d",
    )

    recent_3d = build_recent_behavior_features(
        history_3d,
        "3d",
    )

    recent_7d = build_recent_behavior_features(
        history_7d,
        "7d",
    )
    
    # Merge the recent behavior features into the main features DataFrame
    features = features.merge(recent_1d, left_index=True, right_index=True, how="left")
    features = features.merge(recent_3d, left_index=True, right_index=True, how="left")
    features = features.merge(recent_7d, left_index=True, right_index=True, how="left")

    # fill missing recent activity with 0
    recent_cols = [
        "events_1d",
        "views_1d",
        "carts_1d",
        "sessions_1d",
        "events_3d",
        "views_3d",
        "carts_3d",
        "sessions_3d",
        "events_7d",
        "views_7d",
        "carts_7d",
        "sessions_7d",
    ]
    features[recent_cols] = features[recent_cols].fillna(0) # users with no events in a recent window get 0 for those features

    # Calculate the number of active days in the last 7 days
    # Distinguishes customers that have been active in the last week from those that have not, even if they have the same number of events (e.g., 5 events in 1 day vs. 5 events spread across 7 days)
    if history_7d.empty:
        features["active_days_7d"] = 0
    else:
        active_days_7d = (
            history_7d.assign(
                activity_date=history_7d[
                    "event_time"
                ].dt.date
            )
            .groupby("user_id")["activity_date"]
            .nunique()
        )

        features["active_days_7d"] = (
            active_days_7d.reindex(
                features.index,
                fill_value=0,
            )
        )
    
    # Label: whether the user purchased in the next LOOKAHEAD_DAYS. This is a binary label, not a count of purchases.
    # The set of ever user who purchased anything in the lookahead period is computed once (vectorized) for efficiency, rather than repeatedly in groupby
    purchasers_future = set(future.loc[future["event_type"] == "purchase", "user_id"].unique())
    # The label is computed by checking if each user_id is in the set of purchasers in the future period
    features["will_purchase_next_7d"] = features.index.isin(purchasers_future)

    features["cutoff_date"] = cutoff

    # user_id is the index, but we want it as a column for downstream processing, so we reset the index
    return features.reset_index()

# Each cutoff has its own RFM features and will_purchase_next_7d label, no cutoff's calculation sees another cutoff's data (each gets own history/future split)
def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    all_features = []
    for cutoff in CUTOFF_DATES:
        print(f"Processing cutoff {cutoff}...")
        all_features.append(build_user_features_for_cutoff(df, cutoff))
    return pd.concat(all_features, ignore_index=True)

if __name__ == "__main__":
    df = load_and_clean(SAMPLE_PARQUET)

    user_features = build_user_features(df)

    print("User feature shape:", user_features.shape)
    print(user_features.head())

    user_features.to_parquet(
        USER_FEATURES_OUT,
        index=False
    )

    # Basic label sanity checks.
    purchase_rate = (
        user_features["will_purchase_next_7d"].mean()
    )

    positive_labels = (
        user_features["will_purchase_next_7d"].sum()
    )

    total_rows = len(user_features)

    print(
        f"7-day purchase label rate: "
        f"{purchase_rate:.4%}"
    )

    print(
        f"Positive user-cutoff observations: "
        f"{positive_labels:,} / {total_rows:,}"
    )

    # Number of distinct users receiving at least one
    # positive label across all generated cutoffs.
    positive_users = (
        user_features.loc[
            user_features["will_purchase_next_7d"],
            "user_id",
        ]
        .nunique()
    )

    total_users = user_features["user_id"].nunique()

    print(
        f"Users positive at least once: "
        f"{positive_users:,} / {total_users:,}"
    )