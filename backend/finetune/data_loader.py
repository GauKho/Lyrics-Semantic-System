import os
import pandas as pd
import glob
from sentence_transformers import InputExample
import pickle
import random


def load_and_prepare_data(file_path):
    all_lyrics = []
    csv_file = glob.glob(os.path.join(file_path, "*.csv"))

    for file in csv_file:
        try:
            df = pd.read_csv(file)
            df = df[["Artist", "Title", "Album", "Lyric"]].dropna()

            df["Album"] = df["Album"].astype(str).str.strip().str.lower()
            df["Lyric"] = df["Lyric"].astype(str).str.strip()

            df_cleaned = df[~df["Album"].str.contains("unreleased", case=False, na=False)].copy()
            df_cleaned = df_cleaned[df_cleaned["Lyric"].str.split().str.len() > 5]


            all_lyrics.append(df_cleaned)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    lyrics_df = pd.concat(all_lyrics, ignore_index=True)
    lyrics_df.drop_duplicates(subset=["Lyric"], inplace=True)
    print(f"Loaded {len(lyrics_df)} unique lyrics")
    return lyrics_df

def create_training_pairs_augmented(df, negative_ratio=1.0):
    train_examples = []
    all_titles = df["Title"].str.strip().str.lower().tolist()

    for _, row in df.iterrows():
        lyric = row["Lyric"].strip().lower()
        title = row["Title"].strip().lower()
        album = row["Album"].strip().lower()
        artist = row["Artist"].strip().lower()

        # Positive pairs
        if len(lyric.split()) > 5:
            train_examples.append(InputExample(texts=[lyric, title]))
            train_examples.append(InputExample(texts=[lyric, album]))
            train_examples.append(InputExample(texts=[lyric, artist]))

            # Hard negative pairs
            for _ in range(int(negative_ratio)):
                wrong_title = random.choice(all_titles)
                if wrong_title != title:
                    train_examples.append(InputExample(texts=[lyric, wrong_title]))

    return train_examples

def save_examples(train_examples, out_file="training_data.pkl"):
    with open(out_file, "wb") as f:
        pickle.dump(train_examples, f)
    print(f"Saved {len(train_examples)} training pairs to {out_file}")


def load_examples(pkl_file="training_data.pkl"):
    with open(pkl_file, "rb") as f:
        return pickle.load(f)