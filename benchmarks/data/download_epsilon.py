#!/usr/bin/env python3
"""Download and merge epsilon-normalized dataset parquet files into a single CSV."""

import os
import ssl
import urllib.request
import certifi
import numpy as np
import pandas as pd

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

BASE_URL = "https://huggingface.co/datasets/jxie/epsilon-normalized/resolve/refs%2Fconvert%2Fparquet/default/train"
NUM_FILES = 20
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "epsilon_normalized_train.csv")
TEMP_DIR = os.path.join(SCRIPT_DIR, "epsilon_parquet_temp")

def download_parquet_files():
    """Download all parquet files."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=SSL_CONTEXT)
    )
    urllib.request.install_opener(opener)

    for i in range(NUM_FILES):
        filename = f"{i:04d}.parquet"
        url = f"{BASE_URL}/{filename}"
        local_path = os.path.join(TEMP_DIR, filename)

        if os.path.exists(local_path):
            print(f"[{i+1}/{NUM_FILES}] {filename} already exists, skipping...")
            continue

        print(f"[{i+1}/{NUM_FILES}] Downloading {filename}...")
        urllib.request.urlretrieve(url, local_path)
        print(f"  Downloaded {filename}")

def flatten_inputs(df):
    """Flatten the nested inputs array into separate columns."""
    # Each row's 'inputs' is an array of 2000 single-element arrays
    # Flatten to get actual float values
    inputs_matrix = np.array([
        [x[0] for x in row] for row in df['inputs']
    ])
    # Create DataFrame with numbered columns
    input_cols = [f'f{i}' for i in range(inputs_matrix.shape[1])]
    flat_df = pd.DataFrame(inputs_matrix, columns=input_cols)
    flat_df['label'] = df['label'].values
    return flat_df


def merge_to_csv():
    """Merge all parquet files into a single CSV by streaming one file at a time."""
    print("\nMerging parquet files into CSV...")

    total_rows = 0
    for i in range(NUM_FILES):
        filename = f"{i:04d}.parquet"
        local_path = os.path.join(TEMP_DIR, filename)
        print(f"  [{i+1}/{NUM_FILES}] Reading {filename}...")
        df = pd.read_parquet(local_path)
        flat_df = flatten_inputs(df)
        del df
        total_rows += len(flat_df)

        if i == 0:
            flat_df.to_csv(OUTPUT_CSV, index=False, mode='w')
        else:
            flat_df.to_csv(OUTPUT_CSV, index=False, mode='a', header=False)

        del flat_df

    print(f"  Total rows: {total_rows}")
    print(f"Done! Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    download_parquet_files()
    merge_to_csv()
