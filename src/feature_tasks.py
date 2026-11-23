"""Task parallelism pattern 2: feature extraction (criterion 4).

Each feature group in features.FEATURE_GROUPS only reads columns of the
cleaned data and never another group's output, so the groups are independent
and can run as separate tasks. Results are joined in the fixed FEATURE_GROUPS
order, so the column order never depends on which task finished first.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .features import FEATURE_GROUPS

_DF = {}


def init_worker(clean_cache_path):
    _DF["df"] = pd.read_pickle(clean_cache_path)  # pickle keeps category dtypes exactly


def run_group(name):
    t0 = time.perf_counter()
    out = FEATURE_GROUPS[name](_DF["df"])
    return name, out, time.perf_counter() - t0, os.getpid()


def join_fixed_order(results):
    by_name = {name: frame for name, frame, *_ in results}
    return pd.concat([by_name[name] for name in FEATURE_GROUPS], axis=1)


def run_features_parallel(clean_cache_path, workers):
    """Start a pool, load data in each worker, run each group as one task, join."""
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                             initargs=(clean_cache_path,)) as pool:
        results = list(pool.map(run_group, FEATURE_GROUPS))
    return join_fixed_order(results), results


def run_features_on_pool(pool):
    """Same tasks on an already-running pool whose workers already hold the data."""
    results = list(pool.map(run_group, FEATURE_GROUPS))
    return join_fixed_order(results), results


def group_timings(results):
    return pd.DataFrame([{"group": n, "seconds": s, "columns": f.shape[1], "pid": p}
                         for n, f, s, p in results])
