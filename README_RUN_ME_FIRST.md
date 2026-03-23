# AI Study Assistant V3

## 1) Setup

### macOS/Linux
```bash
cd ai_study_assistant_v3
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
cd ai_study_assistant_v3
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Install FFmpeg:
- macOS: brew install ffmpeg
- Ubuntu: sudo apt-get install ffmpeg
- Windows: install FFmpeg and add it to PATH

## 2) Add your audio
Put your file in the project folder, e.g.:
- test_lecture.wav

Optional references (for WER/ROUGE):
- refs/test_lecture_transcript.txt
- refs/test_lecture_summary.txt

## 3) Run
```bash
python main.py --audio test_lecture.wav --version_tag v3 --output_dir outputs/run_v3 --log_csv outputs/experiments.csv
```

With references:
```bash
python main.py --audio test_lecture.wav --ref_transcript refs/test_lecture_transcript.txt --ref_summary refs/test_lecture_summary.txt --version_tag v3_eval --output_dir outputs/run_v3_eval --log_csv outputs/experiments.csv
```

## 4) Generate report tables
```bash
python src/make_tables.py --csv_path outputs/experiments.csv --out_dir outputs
```

Creates:
- outputs/table2_performance_by_version.csv
- outputs/table3_latest_runs.csv

## 5) What to paste into report
- Figure 4: screenshot of terminal output from run
- Figure 5: screenshot of outputs/run_v3/result.json
- Table 2: outputs/table2_performance_by_version.csv
- Table 3: outputs/table3_latest_runs.csv

## 6) V1 vs V3 comparison
Run old code with:
- --version_tag v1_baseline
Then run V3 with:
- --version_tag v3_eval
Use same audio and refs.
Both rows will be in outputs/experiments.csv
