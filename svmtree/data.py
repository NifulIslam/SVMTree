import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification

GH = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"


def load_datasets():
    """Return {name: (X, y)} for the five UCI datasets."""
    datasets = {}

    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigree", "Age", "Outcome"]
    d = pd.read_csv(GH + "pima-indians-diabetes.data.csv", names=cols)
    d[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = \
        d[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.nan)
    datasets["Pima Diabetes"] = (d.drop(columns="Outcome").values, d["Outcome"].values)

    Xb, yb = load_breast_cancer(return_X_y=True)
    datasets["Breast Cancer"] = (Xb, yb)

    d = pd.read_csv(GH + "wine.csv", header=None)
    datasets["Wine"] = (d.iloc[:, :-1].values, d.iloc[:, -1].values)

    d = pd.read_csv(GH + "glass.csv", header=None)
    datasets["Glass"] = (d.iloc[:, :-1].values, d.iloc[:, -1].values)

    d = pd.read_csv(GH + "iris.csv", header=None)
    datasets["Iris"] = (d.iloc[:, :-1].values, pd.factorize(d.iloc[:, -1])[0])
    return datasets


def dataset_summary(datasets):
    rows = []
    for name, (X, y) in datasets.items():
        cnt = np.bincount(np.unique(y, return_inverse=True)[1])
        rows.append({"Dataset": name, "Samples": len(y), "Features": X.shape[1],
                     "Classes": len(cnt), "Majority class %": round(cnt.max() / len(y) * 100, 1),
                     "Smallest class": int(cnt.min()), "Null cells": int(np.isnan(X).sum())})
    return pd.DataFrame(rows).set_index("Dataset")


def make_large_dataset(seed=42):
    """Synthetic set used only for the training-time test."""
    return make_classification(n_samples=24000, n_features=10, n_informative=6, n_redundant=2,
                               n_clusters_per_class=3, flip_y=0.05, class_sep=1.0, random_state=seed)
