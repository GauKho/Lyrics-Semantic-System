import os
import pandas as pd
import glob
from sentence_transformers import InputExample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import random

def _normalize_text(text):
    return str(text).strip().lower() # text normalization

def load_and_prepare_data(file_path):
    all_lyrics = []
    csv_file = glob.glob(os.path.join(file_path, "*.csv"))

    for file in csv_file:
        try:
            df = pd.read_csv(file)
            df = df[["Artist", "Title", "Album", "Lyric"]].dropna()

            df["Album"] = df["Album"].astype(str).map(_normalize_text)
            df["Lyric"] = df["Lyric"].astype(str).str.strip()
            df["Title"] = df["Title"].astype(str).map(_normalize_text)
            # df["Artist"] = df["Artist"].astype(str).map(_normalize_text)

            df_cleaned = df[~df["Album"].str.contains("unreleased", case=False, na=False)].copy()
            df_cleaned = df_cleaned[df_cleaned["Lyric"].str.split().str.len() > 10]


            all_lyrics.append(df_cleaned)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    lyrics_df = pd.concat(all_lyrics, ignore_index=True)
    lyrics_df.drop_duplicates(subset=["Lyric"], inplace=True)
    print(f"Loaded {len(lyrics_df)} unique lyrics")
    return lyrics_df

# HARD NEGATIVE MINING
def get_hard_negative(title, all_titles, vectorizer, titl_vectors):
    title_vector = vectorizer.transform([title])
    similarity_scores = cosine_similarity(title_vector, titl_vectors)
    rank_idx = similarity_scores.argsort()[0][::-1]
    for idx in rank_idx:
        candidate = all_titles[idx]
        if candidate != title:
            return candidate
    return random.choice(all_titles)

def create_triplet_example_hard(df):
    all_titles = df["Title"].dropna().tolist()
    all_albums = df["Album"].dropna().tolist()

    vectorizer = TfidfVectorizer().fit(all_titles)
    title_vectors = vectorizer.transform(all_titles)
    
    triplet_examples = []
    for _, row in df.iterrows():
        anchor = _normalize_text(row["Lyric"])
        positive_title = _normalize_text(row["Title"])
        negative_title = get_hard_negative(positive_title, all_titles, vectorizer, title_vectors)

        if len(anchor.split()) > 10:
            triplet_examples.append(InputExample(texts=[anchor, positive_title, negative_title]))

        positive_album = _normalize_text(row["Album"])
        negative_album = random.choice([a for a in all_albums if a != positive_album])

        if len(anchor.split()) > 10:
            triplet_examples.append(InputExample(texts=[anchor, positive_album, negative_album]))

    return triplet_examples

def save_examples(train_examples, out_file="training_data_by_triplet.pkl"):
    with open(out_file, "wb") as f:
        pickle.dump(train_examples, f)
    print(f"Saved {len(train_examples)} training pairs to {out_file}")


def load_examples(pkl_file="training_data_by_triplet.pkl"):
    with open(pkl_file, "rb") as f:
        return pickle.load(f)