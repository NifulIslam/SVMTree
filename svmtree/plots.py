import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .experiment import MODEL_ORDER

sns.set_theme(style="whitegrid")

COLORS = {"Decision Tree": "tab:gray", "SVM (RBF)": "tab:blue", "Tree + NB filter": "tab:green",
          "SVMTree (no filter)": "tab:purple", "SVMTree": "tab:red"}
COST_COLORS = {"Decision Tree": "tab:gray", "SVM (RBF)": "tab:blue", "SVMTree depth 2": "tab:orange",
               "SVMTree depth 3": "tab:red", "SVMTree depth 4": "tab:brown"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_classes(datasets, path="figs/fig2_classes.png"):
    fig, ax = plt.subplots(1, len(datasets), figsize=(20, 4.2))
    for a, (name, (X, y)) in zip(ax, datasets.items()):
        cnt = np.bincount(np.unique(y, return_inverse=True)[1])
        bars = a.bar([str(i) for i in range(len(cnt))], cnt, color="tab:blue")
        a.bar_label(bars, fontsize=11, fontweight="bold", padding=2)
        a.set_title(f"{name} ({len(y)} rows)", fontsize=14, fontweight="bold")
        a.set_xlabel("Class", fontsize=12)
        a.set_ylim(0, cnt.max() * 1.2)
    ax[0].set_ylabel("Rows", fontsize=12)
    _save(fig, path)


def plot_noise_curves(mean_df, path="figs/fig3_noise_curves.png"):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for a, m in zip(ax, ["Accuracy", "Macro F1", "MCC"]):
        t = mean_df[m].unstack("Model")[MODEL_ORDER].groupby("Noise").mean()
        for name in MODEL_ORDER:
            a.plot(t.index * 100, t[name], marker="o", linewidth=2.8 if name == "SVMTree" else 2,
                   color=COLORS[name], label=name)
        a.set_xlabel("Label noise in the train part (%)", fontsize=13)
        a.set_ylabel(m, fontsize=13)
        a.set_xticks([0, 10, 20, 30])
        a.set_title(f"{m} (mean of 5 datasets)", fontsize=15, fontweight="bold")
    ax[0].legend(fontsize=11)
    _save(fig, path)


def plot_accuracy_by_dataset(mean_df, path="figs/fig4_accuracy_by_dataset.png"):
    fig, ax = plt.subplots(1, 2, figsize=(18, 5.5))
    for a, p in zip(ax, [0.0, 0.20]):
        t = mean_df["Accuracy"].unstack("Model")[MODEL_ORDER].xs(p, level="Noise")
        t.plot(kind="bar", ax=a, width=0.82, color=[COLORS[c] for c in MODEL_ORDER])
        for c in a.containers:
            a.bar_label(c, fmt="%.2f", rotation=90, padding=2, fontsize=9, fontweight="bold")
        a.set_ylim(0, 1.18)
        a.set_xticklabels(t.index, rotation=0, fontsize=12)
        a.set_title(f"Accuracy with {int(p * 100)}% label noise", fontsize=15, fontweight="bold")
        a.set_xlabel("")
        a.legend(fontsize=9, ncol=3, loc="upper center")
    ax[0].set_ylabel("Accuracy", fontsize=13)
    _save(fig, path)


def plot_ranks(ranks, path="figs/fig5_ranks.png"):
    rk = ranks.groupby("Noise").mean()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    rk.plot(kind="bar", ax=ax, width=0.8, color=[COLORS[c] for c in MODEL_ORDER])
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", fontsize=10, fontweight="bold", padding=2)
    ax.set_xticklabels([f"{int(v * 100)}% noise" for v in rk.index], rotation=0, fontsize=12)
    ax.set_ylabel("Average rank (1 is best)", fontsize=13)
    ax.set_xlabel("")
    ax.set_title("Average rank of each model over the 5 datasets", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    _save(fig, path)


def plot_filter(filt, path="figs/fig6_filter.png"):
    noisy = filt[filt["Noise"] > 0]
    f = noisy.groupby("Noise").agg(Removed=("Removed", "mean"), Hit=("Truly noisy removed", "mean"),
                                   Noisy=("Noisy rows", "mean"))
    f["Precision"] = f["Hit"] / f["Removed"]
    f["Recall"] = f["Hit"] / f["Noisy"]

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(f))
    w = 0.38
    b1 = ax[0].bar(x - w / 2, f["Precision"] * 100, w, color="tab:red", label="Precision")
    b2 = ax[0].bar(x + w / 2, f["Recall"] * 100, w, color="tab:blue", label="Recall")
    ax[0].bar_label(b1, fmt="%.0f", fontsize=12, fontweight="bold", padding=2)
    ax[0].bar_label(b2, fmt="%.0f", fontsize=12, fontweight="bold", padding=2)
    ax[0].plot(x, f.index.values * 100, "k--", marker="s", label="Share of noisy rows")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"{int(v * 100)}%" for v in f.index])
    ax[0].set_ylim(0, 105)
    ax[0].set_xlabel("Label noise", fontsize=13)
    ax[0].set_ylabel("Percent", fontsize=13)
    ax[0].set_title("Filter precision and recall", fontsize=15, fontweight="bold")
    ax[0].legend(fontsize=10)

    for a, col, cmap, title in [(ax[1], "Cap", "Reds", "Cap chosen by the search"),
                                (ax[2], "Depth", "Blues", "Tree depth chosen by the search")]:
        c = filt.groupby(["Noise", col]).size().unstack(fill_value=0)
        c = c.div(c.sum(axis=1), axis=0) * 100
        c.plot(kind="bar", stacked=True, ax=a, colormap=cmap, width=0.7)
        a.set_xticklabels([f"{int(v * 100)}%" for v in c.index], rotation=0)
        a.set_xlabel("Label noise", fontsize=13)
        a.set_ylabel("Share of folds (%)", fontsize=13)
        a.set_title(title, fontsize=15, fontweight="bold")
        a.legend(title=col.lower(), fontsize=10)
    _save(fig, path)


def plot_study(depth_df, cap_df, datasets, path="figs/fig7_study.png"):
    fig, ax = plt.subplots(2, 2, figsize=(16, 9.5))
    for j, p in enumerate([0.0, 0.20]):
        for name in datasets:
            g = depth_df[(depth_df.Dataset == name) & (depth_df.Noise == p)]
            ax[0, j].plot(g["Depth"], g["Accuracy"], marker="o", linewidth=2, label=name)
            g = cap_df[(cap_df.Dataset == name) & (cap_df.Noise == p)]
            ax[1, j].plot(g["Cap"] * 100, g["Accuracy"], marker="o", linewidth=2, label=name)
        ax[0, j].set_title(f"Tree depth, {int(p * 100)}% label noise (cap 10%)", fontsize=14, fontweight="bold")
        ax[0, j].set_xlabel("max_depth", fontsize=12)
        ax[0, j].set_ylabel("Accuracy", fontsize=12)
        ax[1, j].set_title(f"Filter cap, {int(p * 100)}% label noise (depth 2)", fontsize=14, fontweight="bold")
        ax[1, j].set_xlabel("Cap on removed rows (%)", fontsize=12)
        ax[1, j].set_ylabel("Accuracy", fontsize=12)
    ax[0, 0].legend(fontsize=10, ncol=2)
    _save(fig, path)


def plot_cost(cost, slopes, path="figs/fig8_cost.png"):
    sizes = sorted(cost["Rows"].unique())
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    for m, g in cost.groupby("Model"):
        ax[0].plot(g["Rows"], g["Fit s"], marker="o", linewidth=2.5, color=COST_COLORS[m],
                   label=f"{m} (slope {slopes[m]:.2f})")
        ax[2].plot(g["Rows"], g["Accuracy"], marker="o", linewidth=2.5, color=COST_COLORS[m], label=m)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=10)
    ax[0].set_xlabel("Training rows", fontsize=13)
    ax[0].set_ylabel("Fit time (s, log scale)", fontsize=13)
    ax[0].set_title("Training time", fontsize=15, fontweight="bold")

    big = cost[cost.Rows == sizes[-1]].set_index("Model")
    bars = ax[1].bar(big.index, big.loc["SVM (RBF)", "Fit s"] / big["Fit s"],
                     color=[COST_COLORS[m] for m in big.index])
    ax[1].bar_label(bars, fmt="%.1fx", fontsize=12, fontweight="bold", padding=3)
    ax[1].set_ylabel("Speed-up over the SVM", fontsize=13)
    ax[1].tick_params(axis="x", labelrotation=20, labelsize=10)
    ax[1].set_title(f"Speed-up at {sizes[-1]} rows", fontsize=15, fontweight="bold")

    ax[2].set_xscale("log")
    ax[2].set_xlabel("Training rows", fontsize=13)
    ax[2].set_ylabel("Accuracy on 4000 test rows", fontsize=13)
    ax[2].set_title("Accuracy", fontsize=15, fontweight="bold")
    _save(fig, path)
