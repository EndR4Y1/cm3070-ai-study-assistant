import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="outputs/experiments.csv")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    perf = (
        df.groupby("version")[["transcribe_s","classify_s","keywords_s","summarise_s","total_s","compression_ratio","top_confidence","wer","cer","rougeL_f"]]
        .mean()
        .round(3)
        .reset_index()
    )
    latest = df.sort_values("timestamp").groupby(["version","audio"]).tail(1)

    perf.to_csv(f"{args.out_dir}/table2_performance_by_version.csv", index=False)
    latest.to_csv(f"{args.out_dir}/table3_latest_runs.csv", index=False)

    print("Saved:")
    print(f" - {args.out_dir}/table2_performance_by_version.csv")
    print(f" - {args.out_dir}/table3_latest_runs.csv")

if __name__ == "__main__":
    main()
