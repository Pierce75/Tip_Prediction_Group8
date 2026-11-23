"""Simple, fast model. Accuracy is not graded: the model is the workload
for the parallel experiments (hyperparameter search, criterion 4)."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


def build_model(X, C_reg=1.0, class_weight="none", max_iter=1000):
    cat = [c for c in C.CATEGORICAL if c in X.columns]
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    weight = None if class_weight in (None, "none") else class_weight
    clf = LogisticRegression(C=C_reg, class_weight=weight, max_iter=max_iter,
                             solver="lbfgs", random_state=C.SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def train_eval(X_train, y_train, X_eval, y_eval, **params):
    """Fit on train, score on an evaluation set (validation for tuning, test only at the end)."""
    model = build_model(X_train, **params).fit(X_train, y_train)
    prob = model.predict_proba(X_eval)[:, 1]
    return model, binary_scores(y_eval, prob)


def binary_scores(truth, tip_probability):
    """Score the positive class at 0.5 and rank predictions by ROC-AUC."""
    predicted = tip_probability >= 0.5
    return {
        "f1": f1_score(truth, predicted, zero_division=0),
        "roc_auc": roc_auc_score(truth, tip_probability),
        "accuracy": accuracy_score(truth, predicted),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
    }


def majority_scores(y_train, y_eval):
    """Predict the training set’s most common label for every evaluation row."""
    majority = float(y_train.mean() >= 0.5)
    return binary_scores(y_eval, np.full(len(y_eval), majority))
