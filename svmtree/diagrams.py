import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _box(ax, x, y, w, h, text, color, fs=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold")


def _arrow(ax, x1, y1, x2, y2, label=None, lx=0, ly=0, color="black", lw=2.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=18))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=11, fontweight="bold")


def pipeline_diagram(path="figs/fig1_pipeline.png"):
    fig, ax = plt.subplots(figsize=(15, 6.2))
    ax.axis("off")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.2)
    _box(ax, 0.2, 4.2, 2.2, 1.2, "1. Data\n5 tabular datasets", "#d9e6f5")
    _box(ax, 3.0, 4.2, 2.4, 1.2, "2. Preprocessing\nimpute, scale", "#d9e6f5")
    _box(ax, 6.0, 4.2, 2.6, 1.2, "3. Label noise\n(train part only)\n0, 10, 20, 30%", "#fde0c5")
    _box(ax, 9.2, 4.2, 2.6, 1.2, "4. 5-fold CV x 2\ntrain / test split", "#d9e6f5")
    _box(ax, 12.4, 4.2, 2.4, 1.2, "5. Inner 3-fold grid\nsearch per model", "#d9e6f5")
    for x in (2.4, 5.4, 8.6, 11.8):
        _arrow(ax, x + 0.05, 4.8, x + 0.55, 4.8)
    _arrow(ax, 7.5, 4.1, 7.5, 3.8)
    ax.text(7.5, 3.55, "SVMTree (fit on the noisy train part)", ha="center", fontsize=14, fontweight="bold")
    _box(ax, 0.8, 1.6, 3.4, 1.5, "A. Naive Bayes filter\nremove up to cap % of rows\nthat NB says have a wrong label", "#f8d7da")
    _box(ax, 5.2, 1.6, 3.4, 1.5, "B. Decision tree\nlimited depth\nsplits data into leaves", "#d4edda")
    _box(ax, 9.6, 1.6, 3.4, 1.5, "C. One SVM per impure leaf\n(pure leaf = its class)", "#fff3cd")
    _arrow(ax, 4.25, 2.35, 5.15, 2.35)
    _arrow(ax, 8.65, 2.35, 9.55, 2.35)
    _box(ax, 3.6, 0.1, 7.8, 0.95, "6. Test part: Accuracy, Macro F1, MCC, training time, filter quality", "#e2d9f3")
    _arrow(ax, 11.3, 1.55, 10.4, 1.1)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def tree_diagram(path="figs/fig9_tree_structure.png"):
    fig, ax = plt.subplots(figsize=(15, 8.6))
    ax.axis("off")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.6)
    ax.text(7.5, 8.3, "Trained SVMTree (illustrative example, depth 2)", ha="center", fontsize=15, fontweight="bold")
    _box(ax, 0.2, 6.6, 2.7, 1.1, "New instance x\n(scaled features)", "#d9e6f5")
    _box(ax, 5.25, 6.6, 4.5, 1.1, "Root split\nGlucose <= 0.2 ?", "#d4edda")
    _box(ax, 1.9, 4.2, 4.2, 1.1, "Node split\nBMI <= -0.1 ?", "#d4edda")
    _box(ax, 8.9, 4.2, 4.2, 1.1, "Node split\nAge <= 0.4 ?", "#d4edda")
    _arrow(ax, 6.6, 6.55, 4.6, 5.4, "yes", -0.6, 0.15)
    _arrow(ax, 8.4, 6.55, 11.0, 5.4, "no", 0.2, 0.15)
    _box(ax, 0.3, 1.5, 3.2, 1.7, "Leaf 1 (pure)\nonly one class\nPredict: No diabetes", "#fff3cd")
    _box(ax, 3.9, 1.5, 3.2, 1.7, "Leaf 2 (impure)\n>= 10 instances\nRBF SVM #1 predicts", "#f8d7da")
    _box(ax, 8.2, 1.5, 3.2, 1.7, "Leaf 3 (impure)\n>= 10 instances\nRBF SVM #2 predicts", "#f8d7da")
    _box(ax, 11.6, 1.5, 3.2, 1.7, "Leaf 4 (pure)\nonly one class\nPredict: Diabetes", "#fff3cd")
    _arrow(ax, 3.4, 4.15, 1.9, 3.3, "yes", -0.6, 0.1)
    _arrow(ax, 4.6, 4.15, 5.5, 3.3, "no", 0.1, 0.1)
    _arrow(ax, 10.0, 4.15, 9.8, 3.3, "yes", -0.6, 0.1)
    _arrow(ax, 12.0, 4.15, 13.2, 3.3, "no", 0.1, 0.1)
    _box(ax, 0.9, 0.05, 6.4, 0.9, "Pure leaf: output its class directly (no SVM, very cheap)", "#fff3cd", fs=10.5)
    _box(ax, 7.7, 0.05, 6.9, 0.9, "Impure leaf: local SVM draws a curved boundary on its own instances", "#f8d7da", fs=10.5)
    red = "#c0392b"
    _arrow(ax, 2.95, 7.15, 5.2, 7.15, color=red, lw=3.6)
    _arrow(ax, 6.6, 6.55, 4.6, 5.4, color=red, lw=3.6)
    _arrow(ax, 4.6, 4.15, 5.5, 3.3, color=red, lw=3.6)
    ax.text(14.9, 7.6, "Example path of x (red):\nroot: yes -> node: no\n-> Leaf 2 -> SVM #1",
            ha="right", va="top", fontsize=10.5, color=red, fontweight="bold")
    ax.text(14.9, 6.5, "Training only: the Naive Bayes filter first removes\nup to cap % of suspected mislabeled instances,\n"
            "then the tree is grown on the cleaned data.", fontsize=10.5, va="top", ha="right", style="italic")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)
