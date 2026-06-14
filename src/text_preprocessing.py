"""
text_preprocessing.py — Text preprocessing pipeline for the campus assistant.

Implements the text preprocessing steps required by the assignment (Section 3):
  - Lowercase normalisation
  - Punctuation removal
  - Stop word removal
  - Tokenisation
  - Stemming (Porter Stemmer via NLTK)
  - Lemmatisation (WordNet Lemmatiser via NLTK)
  - Train / validation split utilities

This module is used by:
  1. data_exploration.ipynb  — demonstrates each step with sample output
  2. dataset.py              — optionally pre-cleans text before tokenisation

Note on DistilBERT interaction:
  DistilBERT's sub-word tokeniser operates on raw text and learns to handle
  stop words, capitalisation, and morphology internally. Aggressive stop word
  removal before fine-tuning can hurt performance on intent classification.
  Therefore this module exposes two modes:
    - 'exploration'  — full pipeline (lower + stop words + stem/lemma) for data
                       exploration and vocabulary analysis
    - 'light'        — lower + punctuation only, safe to pass to DistilBERT
"""

from __future__ import annotations

import re
import string
from typing import Literal

# ── Stop words ────────────────────────────────────────────────────────────────
# Extended set covering common query patterns in campus navigation contexts.

STOP_WORDS: set[str] = {
    # Common English function words
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "can", "will", "just", "should", "now",
    # Query-specific filler words
    "tell", "show", "give", "please", "could", "would", "like",
    "want", "need", "looking", "help", "get", "go", "going",
    "hey", "hi", "hello", "ok", "okay",
}


# ── Optional NLTK imports ─────────────────────────────────────────────────────

def _try_import_nltk():
    """Return (stemmer, lemmatiser, available) — falls back gracefully if NLTK absent."""
    try:
        import nltk
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        try:
            nltk.data.find("corpora/omw-1.4")
        except LookupError:
            nltk.download("omw-1.4", quiet=True)

        from nltk.stem import PorterStemmer, WordNetLemmatizer
        return PorterStemmer(), WordNetLemmatizer(), True
    except ImportError:
        return None, None, False


_stemmer, _lemmatiser, _NLTK_AVAILABLE = _try_import_nltk()


# ── Core pipeline ─────────────────────────────────────────────────────────────

def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Remove all ASCII punctuation characters."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_stop_words(tokens: list[str]) -> list[str]:
    """Remove tokens that are in the STOP_WORDS set."""
    return [t for t in tokens if t not in STOP_WORDS]


def tokenise(text: str) -> list[str]:
    """Split cleaned text into word tokens (whitespace split after normalisation)."""
    return text.split()


def stem(tokens: list[str]) -> list[str]:
    """
    Apply Porter Stemming to each token.

    Stemming aggressively strips suffixes:
        'running' -> 'run', 'libraries' -> 'librari', 'opened' -> 'open'

    Requires NLTK. Returns tokens unchanged if NLTK is unavailable.
    """
    if not _NLTK_AVAILABLE or _stemmer is None:
        return tokens
    return [_stemmer.stem(t) for t in tokens]


def lemmatise(tokens: list[str]) -> list[str]:
    """
    Apply WordNet Lemmatisation to each token.

    Lemmatisation maps words to their dictionary base form:
        'running' -> 'running' (verb needs POS tag, defaults to noun here)
        'libraries' -> 'library', 'opened' -> 'opened'

    Less aggressive than stemming; preserves readability.
    Requires NLTK. Returns tokens unchanged if NLTK is unavailable.
    """
    if not _NLTK_AVAILABLE or _lemmatiser is None:
        return tokens
    return [_lemmatiser.lemmatize(t) for t in tokens]


def preprocess(
    text: str,
    mode: Literal["exploration", "light"] = "exploration",
    apply_stemming: bool = False,
    apply_lemmatisation: bool = True,
) -> list[str]:
    """
    Full text preprocessing pipeline.

    Args:
        text               : Raw input string.
        mode               : 'exploration' applies stop word removal (for data analysis);
                             'light' applies only lowercase + punctuation removal
                             (safe for DistilBERT input).
        apply_stemming     : If True, apply Porter Stemming after stop word removal.
                             Only used in 'exploration' mode.
        apply_lemmatisation: If True, apply WordNet Lemmatisation after stop word removal.
                             Only used in 'exploration' mode.

    Returns:
        List of processed tokens.
    """
    text = lowercase(text)
    text = remove_punctuation(text)
    tokens = tokenise(text)

    if mode == "exploration":
        tokens = remove_stop_words(tokens)
        if apply_stemming:
            tokens = stem(tokens)
        elif apply_lemmatisation:
            tokens = lemmatise(tokens)

    return tokens


def preprocess_for_model(text: str) -> str:
    """
    Light preprocessing safe for DistilBERT: lowercase + extra whitespace removal.
    Returns a string (not token list) ready to pass to the HuggingFace tokeniser.
    """
    text = lowercase(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Batch utilities ───────────────────────────────────────────────────────────

def preprocess_series(texts: list[str], **kwargs) -> list[list[str]]:
    """Apply preprocess() to a list of texts. Passes kwargs to preprocess()."""
    return [preprocess(t, **kwargs) for t in texts]


def build_vocabulary(token_lists: list[list[str]]) -> dict[str, int]:
    """
    Build a frequency vocabulary from a list of token lists.

    Returns:
        Dict mapping token -> frequency, sorted descending by frequency.
    """
    from collections import Counter
    flat = [tok for tokens in token_lists for tok in tokens]
    return dict(Counter(flat).most_common())


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo() -> None:
    """Run a step-by-step demonstration of the full text preprocessing pipeline."""
    examples = [
        "Where is the main library?",
        "What time does the gym close on Sundays?",
        "Find a quiet study area near the cafeteria.",
        "Show me events at the student union today.",
        "I am looking for the chemistry department.",
    ]

    print("=" * 65)
    print("Text Preprocessing Pipeline — Step-by-Step Demo")
    print("=" * 65)
    print(f"NLTK available: {_NLTK_AVAILABLE}")
    print()

    for text in examples:
        print(f"Input      : {text}")

        step1 = lowercase(text)
        print(f"Lowercase  : {step1}")

        step2 = remove_punctuation(step1)
        print(f"No punct.  : {step2}")

        step3 = tokenise(step2)
        print(f"Tokens     : {step3}")

        step4 = remove_stop_words(step3)
        print(f"No stops   : {step4}")

        if _NLTK_AVAILABLE:
            step5_stem  = stem(step4)
            step5_lemma = lemmatise(step4)
            print(f"Stemmed    : {step5_stem}")
            print(f"Lemmatised : {step5_lemma}")
        else:
            print("Stemmed    : [nltk not installed — pip install nltk]")
            print("Lemmatised : [nltk not installed — pip install nltk]")

        light = preprocess_for_model(text)
        print(f"For model  : '{light}'")
        print()


if __name__ == "__main__":
    demo()
