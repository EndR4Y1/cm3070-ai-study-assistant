"""
run_v1_baseline.py  — V1 one-pass baseline evaluation for ablation comparison.

Run from inside the project folder (with venv active):
    python run_v1_baseline.py

What it does
------------
Runs the V1 approach: a single BART-CNN call on the full (truncated) transcript,
with NO chunking or hierarchical merging — exactly mirroring the original Version 1
behaviour.  Records timing, compression ratio, word count, and optionally ROUGE
scores if a reference summary is present in refs/reference_summary.txt.

Outputs
-------
outputs/v1_baseline_results.json   — all metrics
(also prints a side-by-side comparison with the V4 result if
outputs/eval_results.json already exists)
"""
import json, time, os, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import whisper

sys.path.insert(0, os.path.dirname(__file__))
from src.text_utils import normalize_whitespace, clean_keywords
from src.eval_metrics import compute_rouge

# ── Config ────────────────────────────────────────────────────────────────────
AUDIO      = "test_lecture.wav"
SUMMARIZER = "facebook/bart-large-cnn"
MAX_SUM    = 130
MIN_SUM    = 40

results = {}
t_global = time.perf_counter()

# ── 1. Transcription ──────────────────────────────────────────────────────────
print("\n[1/4] Transcribing audio (Whisper base)...")
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

# ── 2. Keyword extraction (TF-IDF — same as V4) ───────────────────────────────
print("\n[2/4] Extracting keywords (TF-IDF)...")
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

# ── 3. Load BART summariser ───────────────────────────────────────────────────
print("\n[3/4] Loading BART summariser...")
tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER)
sum_model  = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER)

# ── 4. ONE-PASS summarisation (V1 approach — no chunking) ────────────────────
print("\n[4/4] One-pass summarisation (V1 — no chunking)...")
t = time.perf_counter()
inputs = tokenizer(transcript, max_length=1024, truncation=True, return_tensors="pt")
ids = sum_model.generate(
    inputs["input_ids"], max_length=MAX_SUM, min_length=MIN_SUM,
    do_sample=False, num_beams=4, length_penalty=2.0, early_stopping=True,
)
onepass_summary = normalize_whitespace(tokenizer.batch_decode(ids, skip_special_tokens=True)[0])
t_onepass = round(time.perf_counter() - t, 2)
onepass_words       = len(onepass_summary.split())
onepass_compression = round(onepass_words / max(1, word_count), 4)
print(f"    Done: {t_onepass}s  |  {onepass_words} words  |  compression {onepass_compression:.1%}")
print(f"\n  Summary:\n    {onepass_summary}")

results["summary"]            = onepass_summary
results["summary_words"]      = onepass_words
results["compression"]        = onepass_compression
results["onepass_s"]          = t_onepass
results["total_s"]            = round(time.perf_counter() - t_global, 2)

# ── Optional ROUGE ────────────────────────────────────────────────────────────
ref_sum_path = "refs/reference_summary.txt"
if os.path.exists(ref_sum_path):
    with open(ref_sum_path) as f:
        ref_sum = f.read().strip()
    rouge = compute_rouge(ref_sum, onepass_summary)
    results["rouge"] = rouge
    print(f"\n  ROUGE: R1={rouge['rouge1_f']}  R2={rouge['rouge2_f']}  RL={rouge['rougeL_f']}")
else:
    print(f"\n  [ROUGE skipped — create {ref_sum_path} to enable]")
    results["rouge"] = None

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
out_path = "outputs/v1_baseline_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

# ── Side-by-side comparison with V4 ──────────────────────────────────────────
v4_path = "outputs/eval_results.json"
print("\n" + "="*60)
print("V1 BASELINE RESULTS")
print("="*60)
print(f"  Words:            {word_count}")
print(f"  Transcription:    {t_transcribe}s")
print(f"  Keywords time:    {t_keywords}s")
print(f"  Summarisation:    {t_onepass}s")
print(f"  Total:            {results['total_s']}s")
print(f"  Summary words:    {onepass_words}")
print(f"  Compression:      {onepass_compression:.3f} ({onepass_compression:.1%})")
if results.get("rouge"):
    r = results["rouge"]
    print(f"  ROUGE-1 F1:       {r['rouge1_f']}")
    print(f"  ROUGE-2 F1:       {r['rouge2_f']}")
    print(f"  ROUGE-L F1:       {r['rougeL_f']}")

if os.path.exists(v4_path):
    with open(v4_path) as f:
        v4 = json.load(f)
    print("\n" + "-"*60)
    print("COMPARISON — V1 One-Pass  vs  V4 Hierarchical")
    print("-"*60)
    fmt = "  {:<25} {:>12}  {:>12}"
    print(fmt.format("Metric", "V1 Baseline", "V4 Hierarchical"))
    print("  " + "-"*54)

    def _r(v): return f"{v:.3f}" if isinstance(v, float) else str(v)

    print(fmt.format("Transcription (s)",   _r(t_transcribe),            _r(v4.get("transcribe_s", "—"))))
    print(fmt.format("Summarisation (s)",   _r(t_onepass),               _r(v4.get("hier_s", "—"))))
    print(fmt.format("Total (s)",           _r(results["total_s"]),      _r(v4.get("total_s", "—"))))
    print(fmt.format("Summary words",       str(onepass_words),          str(v4.get("hier_words", "—"))))
    print(fmt.format("Compression",         f"{onepass_compression:.3f}", f"{v4.get('hier_compression', '—')}"
          if isinstance(v4.get('hier_compression'), float) else "—"))

    if results.get("rouge") and v4.get("rouge_hier"):
        r1, rv4 = results["rouge"], v4["rouge_hier"]
        print(fmt.format("ROUGE-1 F1", _r(r1["rouge1_f"]), _r(rv4["rouge1_f"])))
        print(fmt.format("ROUGE-2 F1", _r(r1["rouge2_f"]), _r(rv4["rouge2_f"])))
        print(fmt.format("ROUGE-L F1", _r(r1["rougeL_f"]), _r(rv4["rougeL_f"])))

print(f"\nFull results saved to {out_path}")
print("="*60)
