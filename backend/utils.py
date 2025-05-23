import os
import glob
import pandas as pd
from sentence_transformers import InputExample

def load_and_filter_lyrics_csv(data_path: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    min_word = 7

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder: {data_path}")

    all_data = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if not all(col in df.columns for col in ["Artist", "Title", "Album", "Lyric"]):
                print(f"Skipping file (missing columns): {file}")
                continue

            df = df[["Artist", "Title", "Album", "Lyric"]].dropna()
            df["Album"] = df["Album"].astype(str).str.strip().str.lower()
            df["Lyric"] = df["Lyric"].astype(str).str.strip()

            df_filtered = df[
                ~df["Album"].str.contains("unreleased", case=False, na=False) &
                (df["Lyric"].str.split().str.len() >= min_word)
            ]

            all_data.append(df_filtered)

        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not all_data:
        raise ValueError("No valid data found after filtering.")

    return pd.concat(all_data, ignore_index=True)


def load_lyrics_pairs_from_csvs(data_path: str ="backend\data\csv", max_per_file: int = 100):
    all_files = glob.glob(os.path.join(data_path, "*.csv"))
    print(f"[DEBUG] Found {len(all_files)} CSV files in: {data_path}")
    examples = []

    for file in all_files:
        print(f"[DEBUG] Reading file: {file}")
        df = pd.read_csv(file)
        df = df.dropna(subset=["Lyric"])

        # ghep cac lyrics trong cung bai hat lam positive pairs
        for i in range(min(len(df), max_per_file)):
            lyric = df.iloc[i]["Lyric"]

            print(f"\n[DEBUG] Raw lyric #{i}: {repr(lyric)}")

            if not isinstance(lyric, str):
                print("[SKIP] Not a string")
                continue

            lines = [line.strip() for line in lyric.split("\n") if len(line.strip()) > 10]

        # Tao cac cap cau (A,B)
        for j in range(len(lines) - 1):
            text1 = lines[j]
            text2 = lines[j+1]
            examples.append(InputExample(texts=[text1,text2], label=0.9))
        print(f"[DEBUG] Generated {len(examples)} examples so far.")

    print(f"[DEBUG] Total training examples generated: {len(examples)}")
    return examples

if __name__ == "__main__":
    folder = "backend\data\csv"
    train_examples = load_lyrics_pairs_from_csvs(folder)
    print(f"Tổng số cặp huấn luyện: {len(train_examples)}")