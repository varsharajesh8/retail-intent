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


# Building RFM features for one cutoff, then rolling across all cutoffs
# cutoffs range from 2019-10-08 to 2019-10-24, every 3 days, with a 7-day lookahead for the label
CUTOFF_DATES = pd.date_range("2019-10-04", "2019-10-24", freq="2D", tz="UTC")
LOOKAHEAD_DAYS = 7

def build_user_features_for_cutoff(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Build RFM features as of cutoff, and a label for whether the user purchased in the next LOOKAHEAD_DAYS."""
    # strict time split: history is all events before cutoff, future is all events in the next LOOKAHEAD_DAYS
    history = df[df["event_time"] < cutoff]
    future = df[(df["event_time"] >= cutoff) & (df["event_time"] < cutoff + pd.Timedelta(days=LOOKAHEAD_DAYS))]

    # compute purchase condition once (vectorized) for efficiency, rather than repeatedly in groupby
    is_purchase = history["event_type"] == "purchase"
    purchase_rows = history[is_purchase]

    g = history.groupby("user_id") # all events per user
    g_purchase = purchase_rows.groupby("user_id") # only purchase events per user

    # RFM features: recency, frequency, monetary value, plus some additional behavioral features
    features = pd.DataFrame({
        "recency_days": (cutoff - g["event_time"].max()).dt.total_seconds() / 86400, # R
        "n_sessions": g["user_session"].nunique(), # F, number of sessions in history
        "n_events": g.size(), # F, number of events in history
        "n_purchase_events": is_purchase.groupby(history["user_id"]).sum(), # F, number of purchase events in history
        "n_orders": g_purchase["user_session"].nunique(), # F, number of sessions with at least one purchase in history
        "total_spend": g_purchase["price"].sum(), # M, total spend in history
        "avg_price_viewed": g["price"].mean(), # M: price sensitivity proxy
    })

    # g_purchase only contains users who made at least one purchase, so we need to fill NaN for users with no purchases (instead of NaN)
    features["n_orders"] = features["n_orders"].fillna(0)
    features["total_spend"] = features["total_spend"].fillna(0)
    # Average order value is undefined for users with no orders, so we replace 0 with NaN to avoid misleadingly showing $0.00 as their average order value
    features["avg_order_value"] = features["total_spend"] / features["n_orders"].replace(0, np.nan)

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
    print(session_features_realtime.shape)
    session_features_realtime.to_parquet(DATA_DIR / "session_features_realtime.parquet", index=False)

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