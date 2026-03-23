"""
main.py — AI Study Assistant V3  (CLI entry point)
Usage: python main.py --audio lecture.wav [options]
"""
import argparse
import json
import os
import time
from functools import lru_cache
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import whisper

from src.text_utils import (
    normalize_whitespace,
    adaptive_chunk_size,
    chunk_text_sentences,
    filter_topics,
    clean_keywords,
)
from src.eval_metrics import compute_asr_metrics, compute_rouge
from src.experiment_log import log_run, build_log_row

# ---------------------------------------------------------------------------
# Device selection — auto-detect GPU, fall back to CPU
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Cached model loaders — models are loaded once per process, not per call
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_whisper(model_size: str):
    """Load and cache a Whisper model by size string."""
    return whisper.load_model(model_size, device=DEVICE)


@lru_cache(maxsize=4)
def _load_summarizer(model_name: str):
    """Load and cache a seq2seq tokenizer + model pair."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    return tokenizer, model


# ---------------------------------------------------------------------------
# Pipeline components
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str, model_size: str = "base", language: str = "en") -> str:
    model = _load_whisper(model_size)
    result = model.transcribe(audio_path, language=language)
    return normalize_whitespace(result["text"])


def classify_topics_zero_shot(text: str, labels: List[str], classifier_model: str) -> List[Dict]:
    clf = pipeline("zero-shot-classification", model=classifier_model, device=0 if DEVICE == "cuda" else -1)
    out = clf(sequences=text, candidate_labels=labels, multi_label=True)
    topics = [
        {"label": label, "confidence": round(float(score), 4)}
        for label, score in zip(out["labels"], out["scores"])
    ]
    topics.sort(key=lambda x: x["confidence"], reverse=True)
    return topics


def extract_keywords_tfidf(text: str, top_k: int = 12) -> List[str]:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
    X = vectorizer.fit_transform([text])
    scores = X.toarray()[0]
    feats = np.array(vectorizer.get_feature_names_out())
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [feats[i] for i in top_idx if scores[i] > 0]


def _summarise_one(text: str, summarizer_model: str, max_length: int, min_length: int) -> str:
    """
    Summarise a single text chunk using the cached BART model.
    Input is truncated to the model's maximum token capacity.
    """
    try:
        tokenizer, model = _load_summarizer(summarizer_model)
        inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(DEVICE)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            num_beams=4,          # Upgraded from 2 → 4 for better beam search quality
            length_penalty=2.0,   # Encourages slightly longer, more complete summaries
            early_stopping=True,
        )
        return normalize_whitespace(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
    except Exception as exc:
        print(f"[summarise_one] Error: {exc}")
        sentences = text.split(". ")
        return ". ".join(sentences[:min(3, len(sentences))])


def summarise_hierarchical(transcript: str, summarizer_model: str, max_length: int, min_length: int) -> str:
    """
    Sentence-boundary-aware hierarchical summarisation.

    Chunks are formed at sentence boundaries so that no sentence is split
    mid-way.  Each chunk is summarised independently; the interim summaries
    are then merged and re-summarised to produce the final output.
    """
    words = transcript.split()
    chunk_size = adaptive_chunk_size(len(words))
    # Use sentence-aware chunking (V4 improvement over word-based chunking)
    chunks = chunk_text_sentences(transcript, target_words=chunk_size)

    if len(chunks) == 1:
        return _summarise_one(transcript, summarizer_model, max_length, min_length)

    chunk_summaries = []
    for chunk in chunks:
        c_words = len(chunk.split())
        chunk_min = min(min_length, max(20, c_words // 6))
        chunk_max = min(max_length, max(60, c_words // 2))
        chunk_summaries.append(_summarise_one(chunk, summarizer_model, chunk_max, chunk_min))

    merged = " ".join(chunk_summaries)
    return _summarise_one(merged, summarizer_model, max_length, min_length)


def compression_ratio(source_text: str, summary_text: str) -> float:
    src_words = max(1, len(source_text.split()))
    sum_words = len(summary_text.split())
    return round(sum_words / src_words, 3)


def save_outputs(output_dir: str, payload: Dict) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "result.json")
    txt_path  = os.path.join(output_dir, "summary.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(payload.get("summary", ""))
    return json_path, txt_path


def read_optional_text(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AI Study Assistant V3")
    parser.add_argument("--audio",            required=True,  help="Path to lecture audio file")
    parser.add_argument("--output_dir",       default="outputs/run_latest")
    parser.add_argument("--whisper_model",    default="base",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--language",         default="en")
    parser.add_argument("--classifier_model", default="facebook/bart-large-mnli")
    parser.add_argument("--summary_model",    default="facebook/bart-large-cnn")
    parser.add_argument("--max_summary_len",  type=int,   default=130)
    parser.add_argument("--min_summary_len",  type=int,   default=40)
    parser.add_argument("--version_tag",      default="v3")
    parser.add_argument("--min_topic_conf",   type=float, default=0.45)
    parser.add_argument("--top_topics",       type=int,   default=3)
    parser.add_argument("--top_keywords",     type=int,   default=12)
    parser.add_argument("--ref_transcript",   default="")
    parser.add_argument("--ref_summary",      default="")
    parser.add_argument("--log_csv",          default="outputs/experiments.csv")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    candidate_labels = [
        "artificial intelligence",
        "computer science",
        "data science",
        "software engineering",
        "machine learning",
        "business",
        "mathematics",
        "education",
    ]

    print(f"\n[AI Study Assistant V3]  Device: {DEVICE.upper()}")
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # 1. Transcription
    t = time.perf_counter()
    print("  → Transcribing audio…")
    transcript = transcribe_audio(args.audio, args.whisper_model, args.language)
    timings["transcribe_s"] = round(time.perf_counter() - t, 2)

    # 2. Topic classification
    t = time.perf_counter()
    print("  → Classifying topics…")
    raw_topics = classify_topics_zero_shot(transcript, candidate_labels, args.classifier_model)
    topics = filter_topics(raw_topics, min_conf=args.min_topic_conf, max_items=args.top_topics)
    if not topics:
        topics = raw_topics[:args.top_topics]
    timings["classify_s"] = round(time.perf_counter() - t, 2)

    # 3. Keyword extraction
    t = time.perf_counter()
    print("  → Extracting keywords…")
    raw_keywords = extract_keywords_tfidf(transcript, top_k=args.top_keywords)
    keywords = clean_keywords(raw_keywords, min_len=4, max_items=8)
    timings["keywords_s"] = round(time.perf_counter() - t, 2)

    # 4. Hierarchical summarisation
    t = time.perf_counter()
    print("  → Summarising (hierarchical, sentence-aware)…")
    summary = summarise_hierarchical(transcript, args.summary_model, args.max_summary_len, args.min_summary_len)
    timings["summarise_s"] = round(time.perf_counter() - t, 2)

    timings["total_s"] = round(time.perf_counter() - t0, 2)
    comp = compression_ratio(transcript, summary)

    # 5. Metrics
    reference_transcript = read_optional_text(args.ref_transcript)
    reference_summary    = read_optional_text(args.ref_summary)
    asr_scores   = compute_asr_metrics(reference_transcript, transcript)
    rouge_scores = compute_rouge(reference_summary, summary)

    # 6. Persist
    payload = {
        "audio":            args.audio,
        "device":           DEVICE,
        "transcript":       transcript,
        "topics":           topics,
        "keywords":         keywords,
        "summary":          summary,
        "timings":          timings,
        "compression_ratio": comp,
        "metrics":          {**asr_scores, **rouge_scores},
    }
    json_path, txt_path = save_outputs(args.output_dir, payload)

    top_topic = topics[0]["label"]      if topics else None
    top_conf  = topics[0]["confidence"] if topics else None
    row = build_log_row(
        version=args.version_tag,
        audio=args.audio,
        timings=timings,
        compression_ratio=comp,
        top_topic=top_topic,
        top_conf=top_conf,
        wer_score=asr_scores.get("wer"),
        cer_score=asr_scores.get("cer"),
        rouge1_f=rouge_scores.get("rouge1_f"),
        rouge2_f=rouge_scores.get("rouge2_f"),
        rougeL_f=rouge_scores.get("rougeL_f"),
    )
    log_run(args.log_csv, row)

    print(f"\n=== Results ===")
    print(f"  Device:          {DEVICE.upper()}")
    print(f"  Topics:          {topics}")
    print(f"  Keywords:        {keywords}")
    print(f"  Summary:         {summary}")
    print(f"  Timings:         {timings}")
    print(f"  Compression:     {comp}")
    print(f"  Metrics:         {payload['metrics']}")
    print(f"  Saved JSON:      {json_path}")
    print(f"  Saved summary:   {txt_path}")
    print(f"  Logged CSV:      {args.log_csv}")


if __name__ == "__main__":
    main()
