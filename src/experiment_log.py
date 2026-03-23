import csv
import os
from datetime import datetime
from typing import Dict, Optional

def log_run(csv_path: str, row: Dict) -> None:
    dirpath = os.path.dirname(csv_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def build_log_row(
    version: str,
    audio: str,
    timings: Dict[str, float],
    compression_ratio: Optional[float],
    top_topic: Optional[str],
    top_conf: Optional[float],
    wer_score: Optional[float] = None,
    cer_score: Optional[float] = None,
    rouge1_f: Optional[float] = None,
    rouge2_f: Optional[float] = None,
    rougeL_f: Optional[float] = None,
) -> Dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "version": version,
        "audio": audio,
        "transcribe_s": timings.get("transcribe_s"),
        "classify_s": timings.get("classify_s"),
        "keywords_s": timings.get("keywords_s"),
        "summarise_s": timings.get("summarise_s"),
        "total_s": timings.get("total_s"),
        "compression_ratio": compression_ratio,
        "top_topic": top_topic,
        "top_confidence": top_conf,
        "wer": wer_score,
        "cer": cer_score,
        "rouge1_f": rouge1_f,
        "rouge2_f": rouge2_f,
        "rougeL_f": rougeL_f,
    }
