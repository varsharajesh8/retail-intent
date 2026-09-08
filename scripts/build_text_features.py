import json
import os
import re
import time
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_PARQUET = DATA_DIR / "events_sample.parquet"
CATEGORY_TEXT_FEATURES_OUT = DATA_DIR / "category_text_features.parquet"

# loads NLTK's list of common stopwords, wrapped in set for faster membership testing
STOP_WORDS = set(stopwords.words("english"))
# initialize lemmatizer for reducing words to their base form
LEMMATIZER = WordNetLemmatizer()
# only want to fit TF-IDF on events that occurred before this cutoff, to avoid data leakage
TEXT_FIT_END = pd.Timestamp('2019-10-08", tx = "UTC")


def preprocess_text(text: str) -> str:
    """Tokenize, remove stopwords, lemmatize."""
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)  # simple tokenization: words only
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def vectorize_text(
    fit_texts: pd.Series,
    all_texts: pd.Series,
    n_components: int = 20,
) -> np.ndarray:
    """
    Fit TF-IDF and SVD on training-period category text,
    then transform all category descriptions.
    """

    vectorizer = TfidfVectorizer(
        max_features=500
    )

    # Learn vocabulary and IDF weights from training-period text only.
    fit_tfidf = vectorizer.fit_transform(
        fit_texts
    )

    max_components = min(
        n_components,
        fit_tfidf.shape[0] - 1,
        fit_tfidf.shape[1] - 1,
    )

    svd = TruncatedSVD(
        n_components=max_components,
        random_state=38,
    )

    # Learn the latent text dimensions from training-period text only.
    svd.fit(fit_tfidf)

    # Apply the already-fitted TF-IDF representation to all categories.
    all_tfidf = vectorizer.transform(
        all_texts
    )

    # Apply the already-fitted SVD representation.
    reduced = svd.transform(
        all_tfidf
    )

    print(
        f"TF-IDF vocab size: "
        f"{len(vectorizer.vocabulary_)}"
    )

    print(
        f"Explained variance "
        f"(top {max_components} components): "
        f"{svd.explained_variance_ratio_.sum():.2f}"
    )

    return reduced

DESCRIPTIONS_CACHE = DATA_DIR / "category_descriptions_cache.json"
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def load_cache() -> dict:
    """Reads cache file content and parses from JSON text into Python dictionary."""
    if DESCRIPTIONS_CACHE.exists():
        return json.loads(DESCRIPTIONS_CACHE.read_text())
    return {}

def save_cache(cache: dict) -> None:
    """Converts python dictionary to JSON and saves it to a file. Replaces whole file each time, does not append"""
    DESCRIPTIONS_CACHE.write_text(json.dumps(cache, indent=2))

def generate_description(category_code: str) -> str:
    """Generate a description for the given category code using the Anthropic API."""
    # Avoid generating descriptions for unknown categories, as they may not have meaningful descriptions.
    if category_code == "unknown":
        return "unknown miscellaneous product"
    prompt = (
        f"Write one brief, plain product-category description (10-15 words, "
        f"no marketing language) for an e-commerce category  named "
        f"'{category_code}'. Reply with only the description."
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=40,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()

def get_descriptions(categories: list[str]) -> dict:
    """Generate (or load cached) descriptions for each category."""
    cache = load_cache()
    # Determine which categories need to be fetched (not already in cache)
    to_fetch = [c for c in categories if c not in cache]
    print(f"{len(cache)} cached, {len(to_fetch)} to fetch.")

    # Looping through categories that need a description,
    for i, cat in enumerate(to_fetch):
        cache[cat] = generate_description(cat)
        # Provide progress feedback every 20 categories and save the cache to avoid losing progress if interrupted.
        if i % 20 == 0:
            print(f"  {i}/{len(to_fetch)}...")
            save_cache(cache) # save progress periodically
        time.sleep(0.05) # light rate-limit courtesy to avoid overwhelming the API with requests.

    save_cache(cache) # final save
    return cache

if __name__ == "__main__":
    # Load the sample parquet file and extract unique category codes and event time
    df = pd.read_parquet(
        SAMPLE_PARQUET,
        columns=["event_time", "category_code"]
    )

    df["event_time"] = pd.to_datetime(
        df["event_time"],
        utc=True,
    )

    df["category_code"] = (
        df["category_code"]
        .fillna("unknown")
    )
    categories = (
        df["category_code"]
        .unique()
        .tolist()
    )

    training_categories = (
        df.loc[
            df["event_time"] < TEXT_FIT_END,
            "category_code",
        ]
        .unique()
        .tolist()
    )    
    print(f"{len(categories)} unique categories found.")

    # Generate or load cached descriptions for each category
    descriptions = get_descriptions(categories)

    # Create a DataFrame to preview the raw and processed text for each category
    raw_text = pd.Series([descriptions[c] for c in categories])
    processed_text = raw_text.apply(preprocess_text)

    training_raw_text = pd.Series(
        [
            descriptions[c]
            for c in training_categories
        ]
    )

    training_processed_text = (
        training_raw_text.apply(
            preprocess_text
        )
    )
    preview = pd.DataFrame({
        "category_code": categories,
        "raw_text": raw_text,
        "processed_text": processed_text,
    })
    print(preview.head(10))

    # TF-IDF + dimensionality reduction
    # run processed category descriptions through TF_IDF, compressing result to n_components
    text_vectors = vectorize_text(
        fit_texts=training_processed_text,
        all_texts=processed_text,
        n_components=20,
    )
    # list comprehension creates column names for each dimension of the vectorized text features
    vector_cols = [f"text_dim_{i}" for i in range(text_vectors.shape[1])]
    # raw NumPy array -> pandas DF, each row is a catefory with 20 numeric columns describing category's position in compressed text-embedding space
    text_features_df = pd.DataFrame(
    text_vectors,
    columns=vector_cols
)

# Add category_code so the semantic vectors can be merged
# back onto event/category data later.
text_features_df.insert(
    0,
    "category_code",
    categories,
)

print(text_features_df.head(10))

text_features_df.to_parquet(
    CATEGORY_TEXT_FEATURES_OUT,
    index=False
)