# CM3070 Final Project – AI Study Assistant

Local, privacy-preserving lecture processing pipeline. Takes audio as input and
outputs a transcript, zero-shot topic labels, TF-IDF keywords, and a hierarchical summary.

---

## Setup

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Install FFmpeg (required by Whisper):
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`
- Windows: download from https://ffmpeg.org and add to PATH

---

## Running the pipeline

Place your audio file in the project folder (e.g. `test_lecture.wav`) then run:

```bash
python main.py --audio test_lecture.wav --version_tag v4 --output_dir outputs/run_v4 --log_csv outputs/experiments.csv
```

With reference files for metric scoring:
```bash
python main.py --audio test_lecture.wav \
  --ref_transcript refs/test_lecture_transcript.txt \
  --ref_summary refs/test_lecture_summary.txt \
  --version_tag v4_eval \
  --output_dir outputs/run_v4_eval \
  --log_csv outputs/experiments.csv
```

---

## Web interface

```bash
python app.py
```

Open http://localhost:7860 in your browser.

---

## Generate report tables

```bash
python src/make_tables.py --csv_path outputs/experiments.csv --out_dir outputs
```

Produces:
- `outputs/table2_performance_by_version.csv`
- `outputs/table3_latest_runs.csv`

---

## Evaluation scripts

Full V4 evaluation (hierarchical + one-pass ablation):
```bash
python run_eval.py
```

V1 baseline only:
```bash
python run_v1_baseline.py
```

Both scripts read `test_lecture.wav` and write JSON results to `outputs/`.
Optional reference files in `refs/` enable ROUGE and WER scoring.
