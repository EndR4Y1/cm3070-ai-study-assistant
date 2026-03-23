"""
run_eval.py – full evaluation run for the CM3070 project.
Run from inside the project folder:  python run_eval.py
Outputs: outputs/eval_results.json
"""
import json, time, re, sys, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import whisper

sys.path.insert(0, os.path.dirname(__file__))
from src.text_utils import (
    normalize_whitespace, adaptive_chunk_size,
    chunk_text_sentences, filter_topics, clean_keywords,
)
from src.eval_metrics import compute_rouge, compute_asr_metrics

AUDIO          = "test_lecture.wav"
CLASSIFIER     = "facebook/bart-large-mnli"
SUMMARIZER     = "facebook/bart-large-cnn"
MAX_SUM        = 130
MIN_SUM        = 40
TOPIC_LABELS   = [
    "Artificial Intelligence", "Computer Science", "Data Science",
    "Software Engineering", "Machine Learning", "Business",
    "Mathematics", "Education",
]


def summarise_one(text, tokenizer, model, max_l, min_l):
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
    ids = model.generate(
        inputs["input_ids"], max_length=max_l, min_length=min_l,
        do_sample=False, num_beams=4, length_penalty=2.0, early_stopping=True,
    )
    return normalize_whitespace(tokenizer.batch_decode(ids, skip_special_tokens=True)[0])


results = {}
t_global = time.perf_counter()

# 1. transcription
print("\n[1/6] Transcribing audio (Whisper base)...")
t = time.perf_counter()
w_model = whisper.load_model("base")
raw = w_model.transcribe(AUDIO, language="en")
transcript = normalize_whitespace(raw["text"])
t_transcribe = round(time.perf_counter() - t, 2)
word_count = len(transcript.split())
print(f"    Done: {t_transcribe}s  |  {word_count} words")

results["transcript"]   = transcript
results["word_count"]   = word_count
results["transcribe_s"] = t_transcribe

# 2. topic classification
print("\n[2/6] Classifying topics (zero-shot BART-MNLI)...")
t = time.perf_counter()
clf = pipeline("zero-shot-classification", model=CLASSIFIER)
out = clf(sequences=transcript, candidate_labels=TOPIC_LABELS, multi_label=True)
topics = sorted(
    [{"label": l, "confidence": round(float(s), 4)}
     for l, s in zip(out["labels"], out["scores"])],
    key=lambda x: x["confidence"], reverse=True,
)
t_classify = round(time.perf_counter() - t, 2)
print(f"    Done: {t_classify}s")
for tp in topics[:5]:
    print(f"      {tp['label']:30s} {tp['confidence']:.4f}")

results["topics"]     = topics
results["classify_s"] = t_classify

# 3. keyword extraction
print("\n[3/6] Extracting keywords (TF-IDF)...")
t = time.perf_counter()
vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
X = vec.fit_transform([transcript])
scores_arr = X.toarray()[0]
feats = np.array(vec.get_feature_names_out())
top_idx = np.argsort(scores_arr)[::-1][:12]
raw_kws = [feats[i] for i in top_idx if scores_arr[i] > 0]
keywords = clean_keywords(raw_kws, min_len=4, max_items=8)
t_keywords = round(time.perf_counter() - t, 2)
print(f"    Done: {t_keywords}s  |  {keywords}")

results["keywords"]   = keywords
results["keywords_s"] = t_keywords

# 4. load summariser
print("\n[4/6] Loading BART summariser...")
tokenizer  = AutoTokenizer.from_pretrained(SUMMARIZER)
sum_model  = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER)

# 5a. one-pass baseline (V1 approach)
print("\n[5/6] One-pass summarisation (baseline V1)...")
t = time.perf_counter()
onepass_summary     = summarise_one(transcript, tokenizer, sum_model, MAX_SUM, MIN_SUM)
t_onepass           = round(time.perf_counter() - t, 2)
onepass_words       = len(onepass_summary.split())
onepass_compression = round(onepass_words / max(1, word_count), 4)
print(f"    Done: {t_onepass}s  |  {onepass_words} words  |  compression {onepass_compression:.1%}")

results["onepass_summary"]     = onepass_summary
results["onepass_words"]       = onepass_words
results["onepass_compression"] = onepass_compression
results["onepass_s"]           = t_onepass

# 5b. hierarchical summarisation (V4)
print("\n[6/6] Hierarchical summarisation (V4)...")
t = time.perf_counter()
chunk_size = adaptive_chunk_size(word_count)
chunks     = chunk_text_sentences(transcript, target_words=chunk_size)
print(f"    Chunks: {len(chunks)}  (chunk_size={chunk_size})")

chunk_sums = []
for i, chunk in enumerate(chunks):
    cw    = len(chunk.split())
    c_min = min(MIN_SUM, max(20, cw // 6))
    c_max = min(MAX_SUM, max(60, cw // 2))
    cs    = summarise_one(chunk, tokenizer, sum_model, c_max, c_min)
    chunk_sums.append(cs)
    print(f"    Chunk {i+1}/{len(chunks)}: {cw} words → {len(cs.split())} words")

merged       = " ".join(chunk_sums)
hier_summary = summarise_one(merged, tokenizer, sum_model, MAX_SUM, MIN_SUM)
t_hier       = round(time.perf_counter() - t, 2)
hier_words   = len(hier_summary.split())
hier_compression = round(hier_words / max(1, word_count), 4)
print(f"    Done: {t_hier}s  |  {hier_words} words  |  compression {hier_compression:.1%}")

results["hier_summary"]     = hier_summary
results["hier_words"]       = hier_words
results["hier_compression"] = hier_compression
results["hier_s"]           = t_hier
results["num_chunks"]       = len(chunks)
results["chunk_size"]       = chunk_size
results["total_s"]          = round(time.perf_counter() - t_global, 2)

# 6. ROUGE (needs reference files)
ref_sum_path = "refs/reference_summary.txt"
ref_txt_path = "refs/reference_transcript.txt"

if os.path.exists(ref_sum_path):
    with open(ref_sum_path) as f:
        ref_sum = f.read().strip()
    rouge_hier    = compute_rouge(ref_sum, hier_summary)
    rouge_onepass = compute_rouge(ref_sum, onepass_summary)
    results["rouge_hier"]    = rouge_hier
    results["rouge_onepass"] = rouge_onepass
    print(f"\n  ROUGE (hierarchical):  R1={rouge_hier['rouge1_f']}  R2={rouge_hier['rouge2_f']}  RL={rouge_hier['rougeL_f']}")
    print(f"  ROUGE (one-pass):      R1={rouge_onepass['rouge1_f']}  R2={rouge_onepass['rouge2_f']}  RL={rouge_onepass['rougeL_f']}")
else:
    print(f"\n  [ROUGE skipped — create {ref_sum_path} with a reference summary to enable]")
    results["rouge_hier"]    = None
    results["rouge_onepass"] = None

if os.path.exists(ref_txt_path):
    with open(ref_txt_path) as f:
        ref_txt = f.read().strip()
    asr = compute_asr_metrics(ref_txt, transcript)
    results["wer"] = asr["wer"]
    results["cer"] = asr["cer"]
    print(f"  WER: {asr['wer']}   CER: {asr['cer']}")
else:
    print(f"  [WER skipped — create {ref_txt_path} with a reference transcript to enable]")
    results["wer"] = None

# save
os.makedirs("outputs", exist_ok=True)
with open("outputs/eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Audio duration:      ~{660}s (11 min)")
print(f"  Transcript words:    {word_count}")
print(f"  Transcription time:  {t_transcribe}s")
print(f"  Classification time: {t_classify}s")
print(f"  Keywords time:       {t_keywords}s")
print(f"  One-pass summ time:  {t_onepass}s  ({onepass_words} words, {onepass_compression:.1%})")
print(f"  Hier summ time:      {t_hier}s  ({hier_words} words, {hier_compression:.1%})")
print(f"  Total time:          {results['total_s']}s")
print(f"  Chunks used:         {len(chunks)}")
print(f"\n  Top topic:           {topics[0]['label']} ({topics[0]['confidence']})")
print(f"  Keywords:            {', '.join(keywords)}")
print(f"\n  One-pass summary:\n    {onepass_summary[:200]}...")
print(f"\n  Hierarchical summary:\n    {hier_summary[:200]}...")
print(f"\nFull results saved to outputs/eval_results.json")
print("="*60)
