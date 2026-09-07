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


def preprocess_text(text: str) -> str:
    """Tokenize, remove stopwords, lemmatize."""
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)  # simple tokenization: words only
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def vectorize_text(processed_texts: pd.Series, n_components: int = 80) -> np.ndarray:
    """TF-IDF vectorize processed text, then reduce to n_components dense dimensions."""
    # caps vocabulary to 500 most frequent/important words across all your category descriptions
    # Chose 500 given the relatively small number of unique categories and their short descriptions, to avoid overfitting and keep the feature space manageable.
    vectorizer = TfidfVectorizer(max_features=500)
    # fit learns vocab and computes importance weights of each word, transform into numeric vector -> sparse matrix
    tfidf_matrix = vectorizer.fit_transform(processed_texts)

    # reduce dimensionality to make easier to join onto feature tables 
    svd = TruncatedSVD(n_components=n_components, random_state=38)
    reduced = svd.fit_transform(tfidf_matrix)

    print(f"TF-IDF vocab size: {len(vectorizer.vocabulary_)}")
    print(f"Explained variance (top {n_components} components): {svd.explained_variance_ratio_.sum():.2}")

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
    # Load the sample parquet file and extract unique category codes
    df = pd.read_parquet(SAMPLE_PARQUET, columns = ["category_code"])
    categories = df["category_code"].fillna("unknown").unique().tolist()
    print(f"{len(categories)} unique categories found.")

    # Generate or load cached descriptions for each category
    descriptions = get_descriptions(categories)

    # Create a DataFrame to preview the raw and processed text for each category
    raw_text = pd.Series([descriptions[c] for c in categories])
    processed_text = raw_text.apply(preprocess_text)

    preview = pd.DataFrame({
        "category_code": categories,
        "raw_text": raw_text,
        "processed_text": processed_text,
    })
    print(preview.head(10))

    # TF-IDF + dimensionality reduction
    # run processed category descriptions through TF_IDF, compressing result to n_components
    text_vectors = vectorize_text(processed_text)
    # list comprehension creates column names for each dimension of the vectorized text features
    vector_cols = [f"text_dim_{i}" for i in range(text_vectors.shape[1])]
    # raw NumPy array -> pandas DF, each row is a catefory with 20 numeric columns describing category's position in compressed text-embedding space
    text_features_df = pd.DataFrame(text_vectors, columns=vector_cols)
    # add category_code as first column to the text_features_df for readability

    print(text_features_df.head(10))
    text_features_df.to_parquet(CATEGORY_TEXT_FEATURES_OUT, index=False)