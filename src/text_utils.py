from typing import List, Dict
import re

# Expanded generic terms filter — eliminates low-value, non-domain-specific keywords
GENERIC_TERMS = {
    "set", "data", "model", "learning", "use", "used", "using", "system", "process",
    "result", "results", "approach", "based", "method", "methods", "example",
    "given", "show", "shown", "make", "makes", "work", "works", "include",
    "includes", "following", "different", "different", "need", "needs", "like",
    "simply", "also", "actually", "called", "able", "know", "want", "think",
    "right", "just", "thing", "things", "kind", "way", "ways", "case", "cases",
    "look", "looks", "come", "comes", "mean", "means", "point", "points", "form",
}

# Compiled sentence boundary pattern (handles abbreviations gracefully)
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip edges."""
    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences on punctuation boundaries, preserving completeness."""
    raw = _SENT_BOUNDARY.split(text)
    return [s.strip() for s in raw if s.strip()]


def adaptive_chunk_size(word_count: int) -> int:
    """
    Return target chunk size in words, scaled to transcript length.
    Larger transcripts use bigger chunks to reduce the total number of
    intermediate summaries that must be merged.
    """
    if word_count < 500:
        return 180
    if word_count < 1200:
        return 260
    if word_count < 3000:
        return 340
    return 420


def chunk_text_words(text: str, chunk_size: int) -> List[str]:
    """Naive word-count chunker (preserved for backward compatibility)."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def chunk_text_sentences(text: str, target_words: int) -> List[str]:
    """
    Sentence-boundary-aware chunker.

    Groups complete sentences into chunks whose word count stays close to
    `target_words`.  Sentences are never split mid-way, so each chunk is
    grammatically complete and summarisable without artefacts.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return [text]

    chunks: List[str] = []
    current_sents: List[str] = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())
        # Flush current chunk if adding this sentence would exceed the target
        # (but always include at least one sentence per chunk)
        if current_words + sent_words > target_words and current_sents:
            chunks.append(" ".join(current_sents))
            current_sents = []
            current_words = 0
        current_sents.append(sent)
        current_words += sent_words

    if current_sents:
        chunks.append(" ".join(current_sents))

    return chunks


def filter_topics(topics: List[Dict], min_conf: float = 0.45, max_items: int = 3) -> List[Dict]:
    """Return up to `max_items` topics whose confidence meets the threshold."""
    filtered = [t for t in topics if float(t.get("confidence", 0.0)) >= min_conf]
    return filtered[:max_items]


def clean_keywords(keywords: List[str], min_len: int = 4, max_items: int = 8) -> List[str]:
    """
    Remove generic, duplicate, and substring-redundant keywords.

    Improvements over V3:
    - Larger GENERIC_TERMS blocklist
    - Numeric-only tokens removed
    - Substring deduplication: if 'machine' and 'machine learning' both appear,
      'machine' is dropped because it is fully contained within the longer phrase.
    """
    # --- pass 1: basic filters ---
    candidates = []
    seen: set = set()
    for kw in keywords:
        k = kw.lower().strip()
        if len(k) < min_len:
            continue
        if k in GENERIC_TERMS:
            continue
        if k in seen:
            continue
        if re.fullmatch(r'[\d\s.,-]+', k):   # skip pure-numeric tokens
            continue
        seen.add(k)
        candidates.append(k)

    # --- pass 2: drop keywords that are substrings of longer keywords ---
    final = []
    for kw in candidates:
        is_substring = any(
            kw != other and kw in other
            for other in candidates
        )
        if not is_substring:
            final.append(kw)

    return final[:max_items]
