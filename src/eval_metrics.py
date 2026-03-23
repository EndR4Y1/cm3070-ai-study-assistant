from typing import Optional, Dict
from jiwer import wer, cer
from rouge_score import rouge_scorer


def compute_asr_metrics(reference_text: Optional[str], hypothesis_text: str) -> Dict[str, Optional[float]]:
    """WER and CER against a reference transcript. Returns None values if no reference provided."""
    if not reference_text:
        return {"wer": None, "cer": None}
    return {
        "wer": round(float(wer(reference_text, hypothesis_text)), 4),
        "cer": round(float(cer(reference_text, hypothesis_text)), 4),
    }


def compute_rouge(reference_summary: Optional[str], generated_summary: str) -> Dict[str, Optional[float]]:
    """ROUGE-1/2/L scores against a reference summary (precision, recall, F1).
    Returns None values if no reference is provided."""
    if not reference_summary:
        return {
            "rouge1_p": None, "rouge1_r": None, "rouge1_f": None,
            "rouge2_p": None, "rouge2_r": None, "rouge2_f": None,
            "rougeL_p": None, "rougeL_r": None, "rougeL_f": None,
        }
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return {
        "rouge1_p": round(float(scores["rouge1"].precision), 4),
        "rouge1_r": round(float(scores["rouge1"].recall),    4),
        "rouge1_f": round(float(scores["rouge1"].fmeasure),  4),
        "rouge2_p": round(float(scores["rouge2"].precision), 4),
        "rouge2_r": round(float(scores["rouge2"].recall),    4),
        "rouge2_f": round(float(scores["rouge2"].fmeasure),  4),
        "rougeL_p": round(float(scores["rougeL"].precision), 4),
        "rougeL_r": round(float(scores["rougeL"].recall),    4),
        "rougeL_f": round(float(scores["rougeL"].fmeasure),  4),
    }
