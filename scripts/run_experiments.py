"""Run all experiments and write the result tables to results/.

    python scripts/run_experiments.py            # full run (about 10 minutes on one CPU)
    python scripts/run_experiments.py --quick    # small test run
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from svmtree.data import dataset_summary, load_datasets
from svmtree.experiment import (MODEL_ORDER, NOISE, filter_summary, growth_exponents, mean_table,
                                rank_and_tests, run_cost, run_main, run_study)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="fewer noise levels, one repeat, small sizes")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    datasets = load_datasets()
    dataset_summary(datasets).to_csv(f"{args.out}/datasets.csv")

    noise = [0.0, 0.20] if args.quick else NOISE
    repeats = 1 if args.quick else 2
    res, filt = run_main(datasets, noise=noise, n_repeats=repeats)
    res.to_csv(f"{args.out}/results_all_folds.csv", index=False)
    filt.to_csv(f"{args.out}/filter_analysis.csv", index=False)

    mean_df = mean_table(res)
    mean_df.round(4).to_csv(f"{args.out}/results_mean.csv")
    for metric in ["Accuracy", "Macro F1", "MCC"]:
        t = mean_df[metric].unstack("Model")[MODEL_ORDER]
        print(f"\nAverage {metric} by noise level")
        print(t.groupby("Noise").mean().round(4))
        t.round(4).to_csv(f"{args.out}/table_{metric.replace(' ', '_').lower()}.csv")

    if not args.quick:
        ranks, friedman_p, stats = rank_and_tests(mean_df)
        ranks.groupby("Noise").mean().round(3).to_csv(f"{args.out}/ranks.csv")
        stats.round(4).to_csv(f"{args.out}/stats.csv", index=False)
        print("\nAverage rank:\n", ranks.mean().round(2))
        print("Friedman p =", friedman_p)
        print(stats.round(4).to_string(index=False))
        filter_summary(filt).round(3).to_csv(f"{args.out}/filter_summary.csv")
        res.groupby("Model")[["Tune time", "Fit time"]].mean().loc[MODEL_ORDER].round(4).to_csv(
            f"{args.out}/tuning_cost.csv")

    depth_df, cap_df = run_study(datasets, depths=(1, 2, 3) if args.quick else (1, 2, 3, 4, 5, 6))
    depth_df.round(4).to_csv(f"{args.out}/study_depth.csv", index=False)
    cap_df.round(4).to_csv(f"{args.out}/study_cap.csv", index=False)

    sizes = (500, 1000, 2000) if args.quick else (500, 1000, 2000, 4000, 8000, 16000, 20000)
    cost = run_cost(sizes)
    cost.round(4).to_csv(f"{args.out}/cost.csv", index=False)
    print("growth exponents:", {k: round(v, 2) for k, v in growth_exponents(cost).items()})


if __name__ == "__main__":
    main()
