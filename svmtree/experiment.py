import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data import make_large_dataset
from .model import SVMTree

warnings.filterwarnings("ignore")

SEED = 42
NOISE = [0.0, 0.10, 0.20, 0.30]
MODEL_ORDER = ["Decision Tree", "SVM (RBF)", "Tree + NB filter", "SVMTree (no filter)", "SVMTree"]


def make_pipe(clf):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", clf)])


def make_models():
    """Every model gets its own grid search (inner 3-fold CV, macro F1) on the train part."""
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    def gs(clf, grid):
        return GridSearchCV(make_pipe(clf), grid, cv=inner, scoring="f1_macro", n_jobs=1)

    return {
        "Decision Tree": gs(DecisionTreeClassifier(random_state=SEED),
                            {"clf__max_depth": [2, 3, 4, 5, 6, 8, None], "clf__min_samples_leaf": [1, 5]}),
        "SVM (RBF)": gs(SVC(random_state=SEED),
                        {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1]}),
        "Tree + NB filter": gs(SVMTree(local="none"),
                               {"clf__max_depth": [2, 3, 4, 5, 6, 8], "clf__noise_cap": [0.05, 0.10, 0.20]}),
        "SVMTree (no filter)": gs(SVMTree(noise_cap=0.0),
                                  {"clf__max_depth": [1, 2, 3, 4], "clf__C": [1, 10]}),
        "SVMTree": gs(SVMTree(),
                      {"clf__max_depth": [1, 2, 3, 4], "clf__noise_cap": [0.05, 0.10, 0.20], "clf__C": [1, 10]}),
    }


def flip_labels(y, p, rng):
    """Change a share p of the labels to a different class. Returns new labels and a flipped mask."""
    y = y.copy()
    flipped = np.zeros(len(y), dtype=bool)
    if p <= 0:
        return y, flipped
    classes = np.unique(y)
    idx = rng.choice(len(y), size=int(round(p * len(y))), replace=False)
    for i in idx:
        y[i] = rng.choice(classes[classes != y[i]])
        flipped[i] = True
    return y, flipped


def scores(y_true, pred):
    return {"Accuracy": accuracy_score(y_true, pred),
            "Macro F1": f1_score(y_true, pred, average="macro"),
            "MCC": matthews_corrcoef(y_true, pred)}


def run_main(datasets, noise=NOISE, n_splits=5, n_repeats=2):
    """Main comparison. Returns (all fold results, filter analysis)."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    records, filt_records = [], []
    t_start = time.perf_counter()
    for dname, (X, y) in datasets.items():
        for p in noise:
            for fold, (tr, te) in enumerate(rskf.split(X, y)):
                rng = np.random.RandomState(SEED + fold)
                y_tr, flipped = flip_labels(y[tr], p, rng)
                for mname, search in make_models().items():
                    t0 = time.perf_counter()
                    search.fit(X[tr], y_tr)
                    tune_time = time.perf_counter() - t0
                    best = search.best_estimator_
                    t0 = time.perf_counter()
                    clone(best).fit(X[tr], y_tr)
                    fit_time = time.perf_counter() - t0
                    rec = {"Dataset": dname, "Noise": p, "Fold": fold, "Model": mname,
                           "Tune time": tune_time, "Fit time": fit_time,
                           "Best": {k.replace("clf__", ""): v for k, v in search.best_params_.items()}}
                    rec.update(scores(y[te], best.predict(X[te])))
                    records.append(rec)
                    if mname == "SVMTree":
                        clf = best.named_steps["clf"]
                        removed = np.zeros(len(tr), dtype=bool)
                        removed[clf.removed_] = True
                        filt_records.append({
                            "Dataset": dname, "Noise": p, "Fold": fold, "Cap": clf.noise_cap,
                            "Depth": clf.max_depth, "Removed": int(removed.sum()),
                            "Truly noisy removed": int((removed & flipped).sum()),
                            "Noisy rows": int(flipped.sum()),
                            "Leaves": clf.n_leaves_, "SVM leaves": clf.n_svm_})
        print(f"{dname} done, {time.perf_counter() - t_start:.0f}s", flush=True)
    return pd.DataFrame(records), pd.DataFrame(filt_records)


def run_study(datasets, depths=(1, 2, 3, 4, 5, 6), caps=(0.0, 0.05, 0.10, 0.20, 0.30), noise=(0.0, 0.20)):
    """Sensitivity to tree depth and filter cap (no inner search, C=1, gamma=scale)."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED + 1)

    def acc(dname, depth, cap, p):
        X, y = datasets[dname]
        out = []
        for fold, (tr, te) in enumerate(cv.split(X, y)):
            y_tr, _ = flip_labels(y[tr], p, np.random.RandomState(SEED + fold))
            m = make_pipe(SVMTree(max_depth=depth, noise_cap=cap)).fit(X[tr], y_tr)
            out.append(accuracy_score(y[te], m.predict(X[te])))
        return np.mean(out)

    depth_rows, cap_rows = [], []
    for dname in datasets:
        for p in noise:
            for d in depths:
                depth_rows.append({"Dataset": dname, "Noise": p, "Depth": d, "Accuracy": acc(dname, d, 0.10, p)})
            for c in caps:
                cap_rows.append({"Dataset": dname, "Noise": p, "Cap": c, "Accuracy": acc(dname, 2, c, p)})
    return pd.DataFrame(depth_rows), pd.DataFrame(cap_rows)


def run_cost(sizes=(500, 1000, 2000, 4000, 8000, 16000, 20000)):
    """Fit time and accuracy against the number of training rows on a synthetic set."""
    Xs, ys = make_large_dataset(SEED)
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=4000, stratify=ys, random_state=SEED)
    rows = []
    for n in sizes:
        for name, clf in [("Decision Tree", DecisionTreeClassifier(max_depth=6, random_state=SEED)),
                          ("SVM (RBF)", SVC()),
                          ("SVMTree depth 2", SVMTree(max_depth=2)),
                          ("SVMTree depth 3", SVMTree(max_depth=3)),
                          ("SVMTree depth 4", SVMTree(max_depth=4))]:
            pipe = make_pipe(clf)
            t0 = time.perf_counter()
            pipe.fit(Xtr[:n], ytr[:n])
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            pred = pipe.predict(Xte)
            pred_s = time.perf_counter() - t0
            rows.append({"Rows": n, "Model": name, "Fit s": fit_s, "Predict s": pred_s,
                         "Accuracy": accuracy_score(yte, pred)})
        print(n, "done", flush=True)
    return pd.DataFrame(rows)


def growth_exponents(cost):
    """Slope of log(fit time) against log(rows) for every model."""
    return {m: np.polyfit(np.log(g["Rows"]), np.log(g["Fit s"]), 1)[0] for m, g in cost.groupby("Model")}


def mean_table(res):
    cols = ["Accuracy", "Macro F1", "MCC", "Tune time", "Fit time"]
    return res.groupby(["Dataset", "Noise", "Model"])[cols].mean()


def filter_summary(filt):
    f = filt[filt["Noise"] > 0].groupby("Noise").agg(
        Removed=("Removed", "mean"), Hit=("Truly noisy removed", "mean"),
        Noisy=("Noisy rows", "mean"), Cap=("Cap", "mean"))
    f["Precision"] = f["Hit"] / f["Removed"]
    f["Recall"] = f["Hit"] / f["Noisy"]
    return f


def rank_and_tests(mean_df):
    """Ranks over (dataset, noise) cells, Friedman test, and SVMTree vs the others (Wilcoxon)."""
    acc = mean_df["Accuracy"].unstack("Model")[MODEL_ORDER]
    ranks = acc.rank(axis=1, ascending=False)
    friedman_p = friedmanchisquare(*[acc[c] for c in MODEL_ORDER]).pvalue
    rows = []
    for other in ["Decision Tree", "SVM (RBF)", "Tree + NB filter", "SVMTree (no filter)"]:
        subsets = [("all noise levels", acc.index),
                   ("0% noise", acc.xs(0.0, level="Noise", drop_level=False).index),
                   ("20-30% noise", acc[acc.index.get_level_values("Noise") >= 0.2].index)]
        for label, sub in subsets:
            a, b = acc.loc[sub, "SVMTree"], acc.loc[sub, other]
            diff = a - b
            p = wilcoxon(a, b).pvalue if (diff != 0).any() else 1.0
            rows.append({"SVMTree vs": other, "Cells": label, "Mean diff": diff.mean(),
                         "Wins": int((diff > 1e-12).sum()), "Ties": int((diff.abs() <= 1e-12).sum()),
                         "Losses": int((diff < -1e-12).sum()), "Wilcoxon p": p})
    return ranks, friedman_p, pd.DataFrame(rows)
