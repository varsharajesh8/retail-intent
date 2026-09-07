import argparse
import gzip
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

DATA_URL = "https://data.rees46.com/datasets/marketplace/2019-Oct.csv.gz"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

RAW_GZ = DATA_DIR / "2019-Oct.csv.gz"
RAW_CSV = DATA_DIR / "2019-Oct.csv"
FULL_PARQUET = DATA_DIR / "events_full.parquet"
SAMPLE_PARQUET = DATA_DIR / "events_sample.parquet"

DTYPES = {
    "event_type": "category",
    "product_id": "int64",
    "category_id": "int64",
    "category_code": "object",
    "brand": "object",
    "price": "float64",
    "user_id": "int64",
    "user_session": "object",
}

def download(url: str, dest: Path) -> None:
    """Download file from url to dest path."""
    if dest.exists():
        print(f"[skip] {dest} already exists.")
        return
    print(f"Downloading {url} to {dest}")
    with requests.get(url, stream = True, timeout = 60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total = total, unit = "B", unit_scale = True, desc = dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size = 1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))

def unzip(src: Path, dest:Path) -> None:
    """Unzip gz file from src to dest path."""
    if dest.exists():
        print(f"[skip] {dest} already exists.")
        return
    print(f"Unzipping {src} to {dest}")
    with gzip.open(src, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

def csv_to_parquet(src_csv: Path, dest_parquet: Path, chunksize: int = 2_000_000) -> None:
    """Convert CSV file to Parquet file."""
    if dest_parquet.exists():
        print(f"[skip] {dest_parquet} already exists.")
        return
    print(f"Converting {src_csv} to {dest_parquet} (streamed, chunksize={chunksize:,})")
    reader = pd.read_csv(
        src_csv,
        dtype = DTYPES,
        parse_dates = ["event_time"],
        chunksize = chunksize,
    )
    writer = None
    total_rows = 0
    try:
        for i, chunk in enumerate(reader):
            table = pa.Table.from_pandas(chunk, preserve_index = False)
            if writer is None:
                writer = pq.ParquetWriter(dest_parquet, table.schema)
            writer.write_table(table)
            total_rows += len(chunk)
            print(f" wrote chunk {i + 1} ({len(chunk):,} rows, {total_rows:,} total)")

    finally:
        if writer is not None:
            writer.close()
    print(f"Finished writing {total_rows:,} rows to {dest_parquet}")

def make_sample(full_parquet: Path, sample_parquet: Path, frac: float, seed: int = 40) -> None:
    """Build a reproducible user-level sample. 

    Randomly sample users, then retain every event from every session belonging to those selected users.
    This preserves each sampled user's complete observed October history for user-level prediction."""

    import pyarrow.dataset as ds

    if sample_parquet.exists():
        print(f"[skip] {sample_parquet} already exists.")
        return
    print(f"Building {frac:.0%} user-level sample")

    # Read user_id first to sample users without loading entire event dataset to memory
    user_col = pd.read_parquet(full_parquet, columns = ["user_id"])["user_id"]
    users = user_col.unique()

    # Randomly sample users with a fixed seed
    keep_users = set(
        pd.Series(users).sample(frac = frac, random_state = seed).tolist()
    )
    del user_col, users

    # Read only the events belonging to those users
    dataset = ds.dataset(full_parquet, format = "parquet")
    table = dataset.to_table(filter = ds.field("user_id").isin(list(keep_users)))
    sample = table.to_pandas()

    print(
        f"Sample: {len(sample):,} rows, "
        f"{sample['user_id'].nunique():,} users, "
        f"{sample['user_session'].nunique():,} sessions"
    )
    sample.to_parquet(sample_parquet, index = False)


def main():
    # Defining command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-frac", type = float, default = 0.08,
                        help = "Fraction of users to keep in the working sample")
    parser.add_argument("--skip-full-parquet", action = "store_true", help = "Skip building the full parquet (if you only need the sample)")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok = True)


    if RAW_CSV.exists():
        print(f"[skip] {RAW_CSV} already exists.")

    else:
        download(DATA_URL, RAW_GZ)
        unzip(RAW_GZ, RAW_CSV)

    if not args.skip_full_parquet:
        csv_to_parquet(RAW_CSV, FULL_PARQUET)
    else:
        print("Skipped full parquet build (--skip-full-parquet)")

    if FULL_PARQUET.exists():
        make_sample(FULL_PARQUET, SAMPLE_PARQUET, args.sample_frac)
    else:
        print("Cannot build sample - events_full.parquet does not exist yet")
    
    print("\nDone. Use data/events_sample.parquet for iteration, data/events_full.parquet for final run.")


if __name__ == "__main__":
    main()
