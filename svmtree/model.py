import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_X_y

SEED = 42


def nb_noise_mask(X, y, cap=0.10, n_splits=5, seed=SEED):
    """Return a boolean mask over the training rows. False means the row is removed as noisy.

    Gaussian NB predicts every row with cross-validated predictions. A row is a suspect
    when NB predicts another class than its label. Suspects are sorted by the probability
    NB gives to the given label (lowest first) and at most cap * n of them are removed.
    """
    n = len(y)
    classes, y_idx = np.unique(y, return_inverse=True)
    k = min(n_splits, np.bincount(y_idx).min())
    keep = np.ones(n, dtype=bool)
    if cap <= 0 or k < 2:
        return keep
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    proba = cross_val_predict(GaussianNB(), X, y, method="predict_proba", cv=cv)
    p_true = proba[np.arange(n), y_idx]
    suspects = np.where(proba.argmax(1) != y_idx)[0]
    suspects = suspects[np.argsort(p_true[suspects])]
    keep[suspects[: int(cap * n)]] = False
    return keep


class SVMTree(ClassifierMixin, BaseEstimator):
    """NB noise filter -> depth-limited decision tree -> one SVM in every impure leaf.

    local="none" skips the leaf SVMs (a leaf predicts its majority class). It is used
    for the 'Tree + NB filter' ablation. noise_cap=0 turns the filter off.
    """

    def __init__(self, max_depth=2, noise_cap=0.10, C=1.0, gamma="scale",
                 local="svm", min_leaf=3, min_svm=10, random_state=SEED):
        self.max_depth = max_depth
        self.noise_cap = noise_cap
        self.C = C
        self.gamma = gamma
        self.local = local
        self.min_leaf = min_leaf
        self.min_svm = min_svm
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        keep = nb_noise_mask(X, y, self.noise_cap, seed=self.random_state)
        self.removed_ = np.where(~keep)[0]
        Xc, yc = X[keep], y[keep]

        self.tree_ = DecisionTreeClassifier(
            max_depth=self.max_depth, min_samples_leaf=self.min_leaf,
            random_state=self.random_state).fit(Xc, yc)

        leaf = self.tree_.apply(Xc)
        self.leaf_model_ = {}
        for l in np.unique(leaf):
            idx = leaf == l
            ys = yc[idx]
            vals, cnt = np.unique(ys, return_counts=True)
            if self.local == "svm" and len(vals) > 1 and idx.sum() >= self.min_svm:
                self.leaf_model_[l] = SVC(C=self.C, gamma=self.gamma).fit(Xc[idx], ys)
            else:
                self.leaf_model_[l] = vals[cnt.argmax()]   # pure or small leaf
        self.n_leaves_ = len(self.leaf_model_)
        self.n_svm_ = sum(isinstance(m, SVC) for m in self.leaf_model_.values())
        return self

    def predict(self, X):
        X = check_array(X)
        leaf = self.tree_.apply(X)
        out = np.empty(len(X), dtype=self.classes_.dtype)
        for l in np.unique(leaf):
            idx = leaf == l
            m = self.leaf_model_[l]
            out[idx] = m.predict(X[idx]) if isinstance(m, SVC) else m
        return out
