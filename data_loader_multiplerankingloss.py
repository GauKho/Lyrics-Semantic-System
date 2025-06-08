import os, re
import pandas as pd
import unicodedata
import glob
from sentence_transformers import InputExample
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()
from collections import defaultdict
import pickle
import random
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _normalize_text_improved(text: str) -> str:

    """
    Enhanced text normalization with better lyric processing.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string
    """

    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).strip()
    
    # Remove verse/chorus markers but keep the content
    text = re.sub(r'\[(Verse \d*|Chorus|Bridge|Pre-Chorus|Intro|Outro).*?\]', '', text)
    
    # Clean up repetitive markers and annotations
    text = re.sub(r'\([^)]*\)', '', text)  # Remove parenthetical content
    text = re.sub(r'[""''`]', '"', text)    # Standardize quotes
    
    # Preserve important punctuation for meaning
    text = re.sub(r'\s+', ' ', text)        # Normalize whitespace
    text = unicodedata.normalize('NFKC', text)
    
    return text.strip()

def load_and_prepare_data(file_path, sample_size: int = None, min_lyric_words: int = 25):

    """
    Load and prepare data with improved error handling and validation.
    
    Args:
        file_path: Path to directory containing CSV files
        sample_size: Optional number of samples to use
        min_lyric_words: Minimum number of words in lyrics
        
    Returns:
        Prepared DataFrame with cleaned lyrics data
    """

    all_lyrics = []
    csv_files = glob.glob(os.path.join(file_path, "*.csv"))
    
    if not csv_files:
        logger.error(f"No CSV files found in {file_path}")
        return pd.DataFrame()
    logger.info(f"Found {len(csv_files)} csv files")

    for file in tqdm(csv_files, desc="Loading CSV files"):
        try:
            logger.info(f"Processing file: {file}")
            df = pd.read_csv(file, encoding='utf-8')
            
            # Check if required columns exist
            required_cols = ["Artist", "Title", "Album", "Lyric"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in {file}: {missing_cols}")
                continue
            
            # Clean and prepare data
            df = df[required_cols].copy()
            df = df.dropna(subset=["Artist", "Title", "Lyric"])  # Keep rows with essential data
            
            # Normalize text fields
            df["Album"] = df["Album"].fillna("Unknown Album").apply(_normalize_text_improved)
            df["Title"] = df["Title"].apply(_normalize_text_improved)
            df["Artist"] = df["Artist"].apply(_normalize_text_improved)
            df["Lyric"] = df["Lyric"].fillna("").str.strip()

            # Filter out unwanted records
            mask_unreleased = df["Album"].str.contains("unreleased", case=False, na=False)
            placeholder_pattern = "lyrics for this song have yet to be released please check back once the song has been released"
            mask_placeholder = df["Lyric"].str.contains(placeholder_pattern, case=False, na=False)
            
            df = df[~(mask_unreleased | mask_placeholder)]
            
            # Filter by lyric length (minimum 25 words)
            df = df[df["Lyric"].str.split().str.len() > min_lyric_words]
            
            # Remove empty artists/titles after normalization
            df = df[(df["Artist"] != "") & (df["Title"] != "")]
            
            if len(df) > 0:
                all_lyrics.append(df)
                logger.info(f"Loaded {len(df)} records from {file}")
            else:
                logger.warning(f"No valid records found in {file}")
                
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
            continue

    if not all_lyrics:
        logger.warning("No valid data found in any files.")
        return pd.DataFrame()

    lyrics_df = pd.concat(all_lyrics, ignore_index=True)
    
    # Remove duplicates more carefully
    initial_count = len(lyrics_df)
    lyrics_df = lyrics_df.drop_duplicates(subset=["Title", "Artist", "Lyric"], keep='first')
    logger.info(f"Removed {initial_count - len(lyrics_df)} duplicates")
    
    # Sample data if requested
    if sample_size and len(lyrics_df) > sample_size:
        lyrics_df = lyrics_df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {sample_size} rows from dataset")
    
    # Log dataset statistics
    logger.info(f"Final dataset: {len(lyrics_df)} unique lyrics")
    logger.info(f"Unique artists: {lyrics_df['Artist'].nunique()}")
    logger.info(f"Average lyric length: {lyrics_df['Lyric'].str.len().mean():.1f} characters")
    logger.info(f"Average words per lyric: {lyrics_df['Lyric'].str.split().str.len().mean():.1f}")
    
    return lyrics_df

def create_data_splits(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):

    """
    Create train, validation, and test splits.
    
    Args:
        df: Input DataFrame
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_ratio + test_ratio), 
        random_state=random_state,
        stratify=None  # Can add stratification by artist if needed
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=(1 - val_size), 
        random_state=random_state
    )
    
    logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def extract_lyric_segments(lyric_text, segment_length=150, overlap=50):
    
    """
    Extract meaningful segments from lyrics for better training.
    
    Args:
        lyric_text: Full lyric text
        segment_length: Number of words per segment
        overlap: Number of overlapping words between segments
        
    Returns:
        List of lyric segments
    """
    
    if not lyric_text or len(lyric_text.split()) < 30:
        return []
    
    words = lyric_text.split()
    segments = []
    
    # Create overlapping segments
    for i in range(0, len(words), segment_length - overlap):
        segment = ' '.join(words[i:i + segment_length])
        if len(segment.split()) >= 30:  # Minimum meaningful length
            segments.append(segment)
        
        if i + segment_length >= len(words):
            break
    
    return segments
def create_training_pairs(df: pd.DataFrame, negative_ratio=3.0, max_segments_per_song=4, min_lyric_words=50):
   
    """
    Create sophisticated training pairs with better positive/negative balance.
    
    Args:
        df: DataFrame with lyrics data
        negative_ratio: Ratio of negative to positive examples
        max_segments_per_song: Maximum segments to extract per song
        min_lyric_words: Minimum words required in lyrics
        
    Returns:
        List of InputExample objects for training
    """

    train_examples = []
    
    df = df.dropna(subset=["Lyric", "Title", "Artist"]).copy()
    
    if len(df) == 0:
        logger.warning("No valid data for creating training pairs")
        return []

    # Create artist and title mappings for better negative sampling
    artist_songs = defaultdict(list)
    title_variations = {}
    
    for idx, row in df.iterrows():
        artist = _normalize_text_improved(row["Artist"]).lower()
        title = _normalize_text_improved(row["Title"]).lower()
        artist_songs[artist].append(idx)
        title_variations[title] = idx

    logger.info(f"Creating improved training pairs from {len(df)} records")

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Generating training pairs")):
        try:
            lyric = _normalize_text_improved(row["Lyric"])
            title = _normalize_text_improved(row["Title"]).lower()
            artist = _normalize_text_improved(row["Artist"]).lower()
            album = _normalize_text_improved(row["Album"]).lower() if pd.notna(row["Album"]) else ""
            
            if not lyric or not title or not artist or len(lyric.split()) < min_lyric_words:
                continue

            # Extract meaningful segments from lyrics
            segments = extract_lyric_segments(lyric, segment_length=120, overlap=30)
            if not segments:
                segments = [lyric]  # Use full lyric as fallback
            
            # Limit segments per song
            segments = segments[:max_segments_per_song]

            # Create diverse positive metadata variants
            positive_queries = [
                title,
                f"{title} by {artist}",
                f"{artist} {title}",
                f'song "{title}" {artist}',
            ]
            
            if album and album != "unknown album":
                positive_queries.extend([
                    f"{title} {album}",
                    f"{title} by {artist} from {album}",
                    f"{artist} {title} {album}"
                ])

            # Remove duplicates and empty queries
            positive_queries = list(set([q.strip() for q in positive_queries if q.strip()]))

            # Create positive pairs with segments
            for segment in segments:
                for query in positive_queries[:3]:  # Limit to prevent overfitting
                    # Both directions for better learning
                    train_examples.append(InputExample(texts=[segment, query], label=1))
                    train_examples.append(InputExample(texts=[query, segment], label=0.9))  # Slightly lower for asymmetry

            # Create hard negatives: same artist, different song
            same_artist_songs = [i for i in artist_songs[artist] if i != idx]
            if same_artist_songs:
                n_semi_hard = min(2, len(same_artist_songs))
                sampled_indices = random.sample(same_artist_songs, n_semi_hard)
                
                for neg_idx in sampled_indices:
                    neg_row = df.iloc[neg_idx]
                    neg_title = _normalize_text_improved(neg_row["Title"]).lower()
                    neg_album = _normalize_text_improved(neg_row["Album"]).lower() if pd.notna(neg_row["Album"]) else ""
                    
                    if neg_title != title:
                        neg_query = f"{neg_title} by {artist}"
                        if neg_album:
                            neg_query += f" from {neg_album}"
                        
                        for segment in segments[:2]:  # Limit segments for negatives
                            train_examples.append(InputExample(texts=[segment, neg_query], label=0))

            # Create random hard negatives
            other_indices = [i for i in range(len(df)) if i != idx and df.iloc[i]["Artist"].lower() != artist]
            if other_indices:
                n_hard = min(int(negative_ratio), len(other_indices))
                sampled_indices = random.sample(other_indices, n_hard)
                
                for neg_idx in sampled_indices:
                    neg_row = df.iloc[neg_idx]
                    neg_title = _normalize_text_improved(neg_row["Title"]).lower()
                    neg_artist = _normalize_text_improved(neg_row["Artist"]).lower()
                    neg_album = _normalize_text_improved(neg_row["Album"]).lower() if pd.notna(neg_row["Album"]) else ""
                    
                    neg_query = f"{neg_title} by {neg_artist}"
                    if neg_album:
                        neg_query += f" from {neg_album}"
                    
                    # Use only first segment for hard negatives to balance data
                    train_examples.append(InputExample(texts=[segments[0], neg_query], label=0))
                        
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            continue
    
    logger.info(f"Created {len(train_examples)} training pairs")
    return train_examples


def save_examples(train_examples, out_file="training_data.pkl"):
    """Save training examples with error handling."""
    try:
        with open(out_file, "wb") as f:
            pickle.dump(train_examples, f)
        logger.info(f"Saved {len(train_examples)} training pairs to {out_file}")
    except Exception as e:
        logger.error(f"Error saving training examples: {e}")
        raise

def load_examples(pkl_file="training_data.pkl"):
    """Load training examples with error handling."""
    try:
        with open(pkl_file, "rb") as f:
            examples = pickle.load(f)
        logger.info(f"Loaded {len(examples)} training pairs from {pkl_file}")
        return examples
    except Exception as e:
        logger.error(f"Error loading training examples: {e}")
        raise