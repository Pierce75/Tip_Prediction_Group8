"""Task parallelism pattern 1: hyperparameter search (criterion 4),
and the unseeded-randomness demo (criterion 5).

Each trial is an independent job: it reads the shared training/validation data
(loaded once per worker by `init_worker`), fits one model, and scores it on
VALIDATION data only. The test set is never seen during the search.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C, model

_DATA = {}


def select_tuning_rows(labels, limit):
    """Take a repeatable stratified sample within an existing time fold."""
    if len(labels) <= limit:
        return labels.index
    selected, _ = train_test_split(labels.index, train_size=limit, stratify=labels,
                                   random_state=C.SEED)
    return pd.Index(selected).sort_values()


def init_worker(cache_path):
    """Load the feature matrix once per worker process (not once per trial)."""
    df = pd.read_pickle(cache_path)
    y = df.pop("tipped")
    split = df.pop("split")
    for name in ("train", "val"):
        mask = (split == name).to_numpy()
        _DATA[name] = (df[mask], y[mask])


def grid_trial(task):
    trial_id, params = task
    (Xtr, ytr), (Xva, yva) = _DATA["train"], _DATA["val"]
    t0 = time.perf_counter()
    _, scores = model.train_eval(Xtr, ytr, Xva, yva, **params)
    return {"trial_id": trial_id, **params, "val_f1": scores["f1"], "val_roc_auc": scores["roc_auc"],
            "seconds": time.perf_counter() - t0, "pid": os.getpid()}


def make_grid(c_values, class_weights):
    return [(i, {"C_reg": c, "class_weight": weight})
            for i, (c, weight) in enumerate(product(c_values, class_weights))]


def run_search(trials, workers, cache_path, trial_fn=None):
    """workers == 0 -> plain sequential loop in this process (same trial function)."""
    trial_fn = trial_fn or grid_trial
    if workers == 0:
        init_worker(cache_path)
        results = [trial_fn(t) for t in trials]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                                 initargs=(cache_path,)) as pool:
            results = list(pool.map(trial_fn, trials))
    return pd.DataFrame(results).sort_values("trial_id").reset_index(drop=True)


# --- Random search: source of non-determinism 2 --------------------------------------
def random_trial(task):
    """Sample C and class_weight, and a random subsample of training rows.

    task = (trial_id, seed, sample_size). seed=None means unseeded: numpy
    draws fresh OS entropy, so every run samples different values.
    """
    trial_id, seed, sample_size = task
    rng = np.random.default_rng(seed)
    params = {"C_reg": float(10 ** rng.uniform(-3, 1)),
              "class_weight": ["none", "balanced"][int(rng.integers(2))]}
    (Xtr, ytr), (Xva, yva) = _DATA["train"], _DATA["val"]
    idx = rng.choice(len(Xtr), size=sample_size, replace=False)
    _, scores = model.train_eval(Xtr.iloc[idx], ytr.iloc[idx], Xva, yva, **params)
    return {"trial_id": trial_id, **params, "val_roc_auc": scores["roc_auc"]}


def random_tasks(n_trials, sample_size, seed=None):
    """seed=None -> unseeded. Otherwise one independent child seed per trial."""
    if seed is None:
        seeds = [None] * n_trials
    else:
        seeds = np.random.SeedSequence(seed).spawn(n_trials)
    return [(i, s, sample_size) for i, s in enumerate(seeds)]
