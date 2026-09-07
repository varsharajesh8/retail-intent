from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_PARQUET = DATA_DIR / "events_sample.parquet"
SESSION_FEATURES_OUT = DATA_DIR / "session_features.parquet"
USER_FEATURES_OUT = DATA_DIR / "user_features.parquet"

def load_and_clean(path: Path) -> pd.DataFrame:
    """Load and clean the data from the parquet file."""
    df = pd.read_parquet(path)

    # Missingness treatment: diagnosed in EDA notebook, missingness is systematic, not random, so we keep it as a signal rather than dropping rows
    for col in ["category_code", "brand"]:
        # preserves informative singal
        df[f"{col}_missing"] = df[col].isna()
        # makes column usable for grouping and encoding (rather than NaN)
        df[col] = df[col].fillna("unknown")

    return df

def build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per session: behavioral features + purchase label. Uses entire, complete session, including post-purchase events. This is the "oracle" version."""
    is_view = df["event_type"] == "view"
    is_cart = df["event_type"] == "cart"
    is_purchase = df["event_type"] == "purchase"

    g = df.groupby("user_session")

    # Feature store
    features = pd.DataFrame({
        "n_events": g.size(),
        "n_views": is_view.groupby(df["user_session"]).sum(),
        "n_carts": is_cart.groupby(df["user_session"]).sum(),
        "n_distinct_products": g["product_id"].nunique(),
        "n_distinct_categories": g["category_id"].nunique(),
        "avg_price_viewed": g["price"].mean(),
        "max_price_viewed": g["price"].max(),
        "session_duration_sec": (g["event_time"].max() - g["event_time"].min()).dt.total_seconds(),
        "pct_missing_category": g["category_code_missing"].mean(),
        "has_cart_add": is_cart.groupby(df["user_session"]).any(),
        # Label: stored with features for traceability, will enforce separation at model training
        "purchased": is_purchase.groupby(df["user_session"]).any(),
    })

    return features.reset_index()

def truncate_before_purchase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Real-time simulation: for each session, keep only events strictly before
    its first purchase (if any). Non-converting sessions are kept whole, since
    there is no purchase event to truncate before. (DEPLOYABLE VERSION)
    """
    df = df.sort_values("event_time")
    is_purchase = df["event_type"] == "purchase"
    # one row per session, with the timestamp of the first purchase in that session (if any)
    first_purchase_time = df[is_purchase].groupby("user_session")["event_time"].min()

    df = df.merge(
        first_purchase_time.rename("first_purchase_time"),
        # left join, sessions with no purchase will have NaN for first_purchase_time 
        on="user_session", how="left"
    )
    # Keep events that are either in sessions with no purchase, or that occur before the first purchase in their session
    keep = df["first_purchase_time"].isna() | (df["event_time"] < df["first_purchase_time"])
    # Drop the first_purchase_time column, since it was only used for filtering and is not a feature we want to keep
    return df[keep].drop(columns=["first_purchase_time"])


def build_session_features_realtime(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature definitions as build_session_features, but computed only
    on pre-purchase events — this is the leakage-safe version for Task C.1."""
    truncated = truncate_before_purchase(df)

    features = build_session_features(truncated)
    # purchased label must come from the FULL session, not the truncated one,
    # since truncation removes the purchase event itself
    full_purchased = (df["event_type"] == "purchase").groupby(df["user_session"]).any()
    features = features.set_index("user_session")
    features["purchased"] = full_purchased
    return features.reset_index()

def build_session_user_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each session, compute the session's user's purchase history using
    ONLY events strictly before this session's own start time. This lets the
    session model see "is this a new visitor or a returning customer,"
    without leaking anything from the current or future sessions.
    """
    session_starts = df.groupby("user_session").agg(
        user_id=("user_id", "first"),
        session_start=("event_time", "min"),
    ).reset_index()

    purchases = df[df["event_type"] == "purchase"][["user_id", "user_session", "event_time", "price"]]
    # collapse to order-level first (one row per session that had a purchase)
    orders = purchases.groupby(["user_id", "user_session"]).agg(
        order_time=("event_time", "min"),
        order_value=("price", "sum"),
    ).reset_index()

    # join each session against its OWN user's orders, then keep only strictly-prior ones
    merged = session_starts.merge(orders, on="user_id", how="left", suffixes=("", "_order"))
    merged["is_prior"] = merged["order_time"] < merged["session_start"]
    prior_orders = merged[merged["is_prior"]]

    agg = prior_orders.groupby("user_session").agg(
        user_past_orders=("user_session_order", "nunique"),
        user_past_spend=("order_value", "sum"),
        user_last_purchase_time=("order_time", "max"),
    ).reset_index()

    result = session_starts[["user_session", "session_start"]].merge(agg, on="user_session", how="left")
    result["user_past_orders"] = result["user_past_orders"].fillna(0)
    result["user_past_spend"] = result["user_past_spend"].fillna(0)
    result["user_days_since_last_purchase"] = (
        (result["session_start"] - result["user_last_purchase_time"]).dt.total_seconds() / 86400
    )

    return result[["user_session", "user_past_orders", "user_past_spend", "user_days_since_last_purchase"]]


# Building RFM features for one cutoff, then rolling across all cutoffs
# cutoffs range from 2019-10-08 to 2019-10-24, every 3 days, with a 7-day lookahead for the label
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
    session_features = build_session_features(df)
    print(session_features.shape)
    print(session_features.head())
    session_features.to_parquet(SESSION_FEATURES_OUT, index = False)

    user_features = build_user_features(df)
    print(user_features.shape)
    print(user_features.head())
    user_features.to_parquet(USER_FEATURES_OUT, index = False)

    session_features_oracle = session_features  # full-session version, already built = "oracle"
    session_features_realtime = build_session_features_realtime(df)

    user_history = build_session_user_history_features(df)    
    session_features_realtime = session_features_realtime.merge(user_history, on="user_session", how="left")
    print(session_features_realtime.shape)
    session_features_realtime.to_parquet(DATA_DIR / "session_features_realtime.parquet", index = False)


    # Sanity check: sessions dropped from realtime (should be sessions whose
    # very first event is a purchase, leaving no pre-purchase history)
    oracle_sessions = set(session_features_oracle["user_session"])
    realtime_sessions = set(session_features_realtime["user_session"])
    missing = oracle_sessions - realtime_sessions
    print(len(missing))

    # overall purchase rate in the sample, for sanity check
    print(user_features["will_purchase_next_7d"].mean() * 100)
    print(user_features["will_purchase_next_7d"].sum())

    # how many distinct users get a positive label at all
    positive_users = user_features.loc[user_features["will_purchase_next_7d"], "user_id"].nunique()
    total_eligible_users = user_features["user_id"].nunique()
    print(positive_users, total_eligible_users, positive_users / total_eligible_users * 100)

    # does this roughly reconcile with known purchase counts from Day 2
    distinct_purchasers_overall = df.loc[df["event_type"] == "purchase", "user_id"].nunique()
    print(distinct_purchasers_overall)

    # spot check user by hand
    example = user_features[user_features["will_purchase_next_7d"]].iloc[0]
    uid, cutoff = example["user_id"], example["cutoff_date"]
    print(uid, cutoff)

    user_events = df[df["user_id"] == uid].sort_values("event_time")
    print(user_events[["event_time", "event_type"]].to_string(index=False))

    oct1_7_purchasers = df.loc[(df["event_type"] == "purchase") & (df["event_time"] < "2019-10-08"), "user_id"].nunique()
    print(oct1_7_purchasers)

    oracle_df = pd.read_parquet(DATA_DIR / "session_features.parquet")
    realtime_df = pd.read_parquet(DATA_DIR / "session_features_realtime.parquet")

    print("Oracle purchase rate:", oracle_df["purchased"].mean())
    print("Realtime purchase rate:", realtime_df["purchased"].mean())