"""
app.py – Gradio web UI for the AI Study Assistant.
Run: python app.py  then open http://localhost:7860
"""
import json
import os
import re
import time
from typing import List, Dict, Optional

import gradio as gr
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import whisper

from src.text_utils import (
    normalize_whitespace,
    adaptive_chunk_size,
    chunk_text_words,
    filter_topics,
    clean_keywords,
)
from src.eval_metrics import compute_rouge
from src.experiment_log import log_run, build_log_row

CANDIDATE_LABELS = [
    "Artificial Intelligence",
    "Computer Science",
    "Data Science",
    "Software Engineering",
    "Machine Learning",
    "Business",
    "Mathematics",
    "Education",
]
DEFAULT_CLASSIFIER = "facebook/bart-large-mnli"
DEFAULT_SUMMARIZER  = "facebook/bart-large-cnn"


def highlight_keywords(text: str, keywords: List[str]) -> str:
    """Wraps each keyword in a gold highlight span."""
    highlighted = text
    for kw in sorted(keywords, key=len, reverse=True):
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        highlighted = pattern.sub(
            f'<mark style="background:#ffe066;border-radius:3px;padding:0 3px">{kw}</mark>',
            highlighted,
        )
    return (
        '<div style="font-family:Georgia,serif;line-height:1.85;font-size:15px;'
        f'white-space:pre-wrap;padding:12px">{highlighted}</div>'
    )


def build_metrics_md(timings: dict, comp: float, rouge: dict) -> str:
    lines = [
        "### ⏱️ Processing Times",
        "| Stage | Time |",
        "|-------|------|",
        f"| Transcription | {timings.get('transcribe_s', '—')} s |",
        f"| Classification | {timings.get('classify_s', '—')} s |",
        f"| Keywords | {timings.get('keywords_s', '—')} s |",
        f"| Summarisation | {timings.get('summarise_s', '—')} s |",
        f"| **Total** | **{timings.get('total_s', '—')} s** |",
        "",
        f"### 📉 Compression Ratio",
        f"**{comp}** — summary length ÷ transcript length (lower = more compressed)",
    ]
    if any(v is not None for v in rouge.values()):
        lines += [
            "",
            "### 📐 ROUGE Scores (vs. your reference summary)",
            "| Metric | F-Score |",
            "|--------|---------|",
            f"| ROUGE-1 | {rouge.get('rouge1_f', '—')} |",
            f"| ROUGE-2 | {rouge.get('rouge2_f', '—')} |",
            f"| ROUGE-L | {rouge.get('rougeL_f', '—')} |",
        ]
    return "\n".join(lines)


def run_pipeline(
    audio_file: Optional[str],
    whisper_model_size: str,
    language: str,
    min_topic_conf: float,
    top_n_topics: int,
    top_n_keywords: int,
    max_summary_len: int,
    min_summary_len: int,
    ref_summary: str,
    progress=gr.Progress(track_tqdm=True),
):
    if audio_file is None:
        raise gr.Error("Please upload an audio file before clicking Process.")

    timings: dict = {}
    t0 = time.perf_counter()

    # transcribe
    progress(0.05, desc="Loading Whisper model…")
    w_model = whisper.load_model(whisper_model_size)
    progress(0.15, desc="Transcribing audio…")
    t = time.perf_counter()
    result = w_model.transcribe(audio_file, language=language)
    transcript = normalize_whitespace(result["text"])
    timings["transcribe_s"] = round(time.perf_counter() - t, 2)

    # topic classification
    progress(0.35, desc="Classifying topics…")
    t = time.perf_counter()
    clf = pipeline("zero-shot-classification", model=DEFAULT_CLASSIFIER)
    out = clf(sequences=transcript, candidate_labels=CANDIDATE_LABELS, multi_label=True)
    raw_topics = sorted(
        [{"label": l, "confidence": float(s)} for l, s in zip(out["labels"], out["scores"])],
        key=lambda x: x["confidence"],
        reverse=True,
    )
    topics = filter_topics(raw_topics, min_conf=min_topic_conf, max_items=top_n_topics)
    if not topics:
        topics = raw_topics[:top_n_topics]
    timings["classify_s"] = round(time.perf_counter() - t, 2)

    # keyword extraction
    progress(0.55, desc="Extracting keywords…")
    t = time.perf_counter()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
    X = vec.fit_transform([transcript])
    scores_arr = X.toarray()[0]
    feats = np.array(vec.get_feature_names_out())
    top_idx = np.argsort(scores_arr)[::-1][:top_n_keywords]
    raw_kws = [feats[i] for i in top_idx if scores_arr[i] > 0]
    keywords = clean_keywords(raw_kws, min_len=4, max_items=8)
    timings["keywords_s"] = round(time.perf_counter() - t, 2)

    # hierarchical summarisation
    progress(0.65, desc="Loading summarisation model…")
    t = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_SUMMARIZER)
    sum_model  = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_SUMMARIZER)

    def _summarise_one(text: str, max_l: int, min_l: int) -> str:
        inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
        ids = sum_model.generate(
            inputs["input_ids"], max_length=max_l, min_length=min_l,
            do_sample=False, num_beams=2,
        )
        return normalize_whitespace(tokenizer.batch_decode(ids, skip_special_tokens=True)[0])

    words      = transcript.split()
    chunk_size = adaptive_chunk_size(len(words))
    chunks     = chunk_text_words(transcript, chunk_size=chunk_size)

    if len(chunks) == 1:
        progress(0.80, desc="Summarising…")
        summary = _summarise_one(transcript, max_summary_len, min_summary_len)
    else:
        chunk_sums = []
        for i, chunk in enumerate(chunks):
            progress(
                0.70 + 0.20 * (i / len(chunks)),
                desc=f"Summarising chunk {i + 1}/{len(chunks)}…",
            )
            c_min = min(min_summary_len, max(20, len(chunk.split()) // 6))
            c_max = min(max_summary_len, max(60, len(chunk.split()) // 2))
            chunk_sums.append(_summarise_one(chunk, c_max, c_min))
        progress(0.92, desc="Merging chunk summaries…")
        summary = _summarise_one(" ".join(chunk_sums), max_summary_len, min_summary_len)

    timings["summarise_s"] = round(time.perf_counter() - t, 2)
    timings["total_s"]     = round(time.perf_counter() - t0, 2)

    comp  = round(len(summary.split()) / max(1, len(transcript.split())), 3)
    rouge = compute_rouge(ref_summary.strip() or None, summary)

    # save
    os.makedirs("outputs/ui_runs", exist_ok=True)
    payload = {
        "transcript": transcript,
        "topics":     topics,
        "keywords":   keywords,
        "summary":    summary,
        "timings":    timings,
        "compression_ratio": comp,
        "metrics":    rouge,
    }
    json_path = os.path.abspath("outputs/ui_runs/result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log_run(
        "outputs/experiments.csv",
        build_log_row(
            version="v4-ui",
            audio=audio_file,
            timings=timings,
            compression_ratio=comp,
            top_topic=topics[0]["label"] if topics else None,
            top_conf=topics[0]["confidence"] if topics else None,
            rouge1_f=rouge.get("rouge1_f"),
            rouge2_f=rouge.get("rouge2_f"),
            rougeL_f=rouge.get("rougeL_f"),
        ),
    )

    progress(1.0, desc="Done!")

    return (
        transcript,
        highlight_keywords(transcript, keywords),
        {t["label"]: t["confidence"] for t in topics},
        ", ".join(keywords),
        summary,
        build_metrics_md(timings, comp, rouge),
        json_path,
    )


CSS = """
.gradio-container { max-width: 1150px !important; margin: auto !important; }
#header-title   { text-align: center; font-size: 2.2em; font-weight: 700; margin-bottom: 0; }
#header-sub     { text-align: center; color: #666; font-size: 1.05em; margin-top: 4px; }
.tab-nav button { font-size: 14px !important; font-weight: 600 !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="AI Study Assistant") as demo:

    gr.Markdown("# 🎓 AI Study Assistant", elem_id="header-title")
    gr.Markdown(
        "Upload a lecture recording to get a transcript, topic labels, keywords, "
        "and a summary — all processed locally on your machine.",
        elem_id="header-sub",
    )

    with gr.Row(equal_height=False):

        with gr.Column(scale=3):
            audio_in = gr.Audio(
                label="🎙️ Upload Lecture Audio (WAV / MP3 / M4A)",
                type="filepath",
                sources=["upload"],
            )
            ref_sum_in = gr.Textbox(
                label="📋 Reference Summary — optional (enables ROUGE scoring)",
                placeholder="Paste a reference summary here to measure quality against it…",
                lines=4,
            )

        with gr.Column(scale=2):
            with gr.Accordion("⚙️ Settings", open=False):
                whisper_model = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large"],
                    value="base",
                    label="Whisper Model Size",
                    info="Larger = more accurate but slower",
                )
                language = gr.Dropdown(
                    choices=["en", "fr", "de", "es", "it", "zh", "ja"],
                    value="en",
                    label="Audio Language",
                )
                min_conf = gr.Slider(
                    0.0, 1.0, value=0.45, step=0.05,
                    label="Min Topic Confidence",
                    info="Topics below this threshold are excluded",
                )
                top_topics = gr.Slider(
                    1, 8, value=3, step=1,
                    label="Max Topics to Show",
                )
                top_keywords = gr.Slider(
                    4, 24, value=12, step=1,
                    label="Keywords to Evaluate (pre-filter)",
                )
                max_sum = gr.Slider(
                    50, 300, value=130, step=10,
                    label="Max Summary Length (tokens)",
                )
                min_sum = gr.Slider(
                    10, 100, value=40, step=5,
                    label="Min Summary Length (tokens)",
                )

    run_btn = gr.Button("▶  Process Audio", variant="primary", size="lg")
    gr.Markdown("---")

    with gr.Tabs():

        with gr.TabItem("📝 Transcript"):
            transcript_box = gr.Textbox(
                label="Raw Transcript",
                lines=12,
                interactive=False,
                show_copy_button=True,
            )
            highlighted_html = gr.HTML(label="Transcript with Keywords Highlighted")

        with gr.TabItem("🔍 Topics & Keywords"):
            gr.Markdown("**Detected Topics** — confidence scores from zero-shot classification:")
            topics_label = gr.Label(label="Topic Confidence", num_top_classes=8)
            gr.Markdown("**Top Keywords** — extracted via TF-IDF (bigrams included):")
            keywords_box = gr.Textbox(label="Keywords", interactive=False)

        with gr.TabItem("📄 Summary"):
            summary_box = gr.Textbox(
                label="Generated Summary",
                lines=10,
                interactive=False,
                show_copy_button=True,
            )

        with gr.TabItem("📊 Metrics"):
            metrics_md = gr.Markdown()

        with gr.TabItem("💾 Download Results"):
            gr.Markdown("Download the full JSON output (transcript, topics, keywords, summary, timings, metrics).")
            file_out = gr.File(label="result.json")

    run_btn.click(
        fn=run_pipeline,
        inputs=[
            audio_in, whisper_model, language,
            min_conf, top_topics, top_keywords,
            max_sum, min_sum, ref_sum_in,
        ],
        outputs=[
            transcript_box, highlighted_html,
            topics_label, keywords_box,
            summary_box, metrics_md, file_out,
        ],
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
