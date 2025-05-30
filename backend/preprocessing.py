import re 
import json
import nltk
import os
from nltk.corpus import stopwords, wordnet
import nltk.downloader
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk_packages = [
    ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
    ('tokenizers/punkt', 'punkt'),
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet', 'wordnet'),
    ('corpora/omw-1.4', 'omw-1.4')
]

for path, name in nltk_packages:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name, quiet=True)

# Construct the path to slang_map.json relative to this file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
slang_map_path = os.path.join(current_dir, "slang_map.json")
with open(slang_map_path, "r") as f:
    slang_dict = json.load(f)

stop_word_set = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower() #Lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text) #Remove punctuation and special characters
    text = re.sub(r"\s+", " ", text).strip() #Remove whitespace
    tokens = word_tokenize(text) #Tokenize

    tokens = [slang_dict.get(word, word) for word in tokens]

    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_word_set] #Remove stopwords and lemmatize

    pos_tagged_tokens = nltk.pos_tag(tokens)

    lemmatized_tokens = []
    
    for word, tag in pos_tagged_tokens:
        if word not in stop_word_set:
            word_pos = get_wordnet_pos(tag)
            lemmatized_tokens.append(lemmatizer.lemmatize(word, pos=word_pos))


    return lemmatized_tokens
