"""Draw every figure from the files in results/ (and the two method diagrams).

    python scripts/make_figures.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from svmtree import diagrams, plots
from svmtree.data import load_datasets
from svmtree.experiment import growth_exponents, mean_table, rank_and_tests

RES, FIGS = "results", "figs"


def main():
    os.makedirs(FIGS, exist_ok=True)
    datasets = load_datasets()
    res = pd.read_csv(f"{RES}/results_all_folds.csv")
    filt = pd.read_csv(f"{RES}/filter_analysis.csv")
    depth_df = pd.read_csv(f"{RES}/study_depth.csv")
    cap_df = pd.read_csv(f"{RES}/study_cap.csv")
    cost = pd.read_csv(f"{RES}/cost.csv")

    mean_df = mean_table(res)
    ranks, _, _ = rank_and_tests(mean_df)

    diagrams.pipeline_diagram(f"{FIGS}/fig1_pipeline.png")
    diagrams.tree_diagram(f"{FIGS}/fig9_tree_structure.png")
    plots.plot_classes(datasets, f"{FIGS}/fig2_classes.png")
    plots.plot_noise_curves(mean_df, f"{FIGS}/fig3_noise_curves.png")
    plots.plot_accuracy_by_dataset(mean_df, f"{FIGS}/fig4_accuracy_by_dataset.png")
    plots.plot_ranks(ranks, f"{FIGS}/fig5_ranks.png")
    plots.plot_filter(filt, f"{FIGS}/fig6_filter.png")
    plots.plot_study(depth_df, cap_df, datasets, f"{FIGS}/fig7_study.png")
    plots.plot_cost(cost, growth_exponents(cost), f"{FIGS}/fig8_cost.png")
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
