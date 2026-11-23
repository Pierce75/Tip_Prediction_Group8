"""Data parallelism (rubric criteria 2 and 3) and the completion-order demo (criterion 5).

Workload per shard = read one day of raw trips in batches (iter_batches),
keep card trips, apply the same CLEAN_STEPS as load_clean(), then compute
partial sums per (pickup zone, hour): trips, tipped trips, sum of tip %.
Partial sums/counts are merged once at the end, then divided.
"""
import calendar
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import config as C
from .data import apply_clean_steps, add_target

GROUP_KEYS = ["PULocationID", "hour"]
SUM_COLS = ["trips", "tipped", "tip_pct_sum"]


# --- Sharding -----------------------------------------------------------------
def write_day_shards(src=C.DATA_FILE, out_dir=C.SHARD_DIR):
    """Split the raw file into one Parquet file per pickup day (done once, not timed).

    Rows whose pickup is outside the target month (spill-over) belong to no shard.
    Returns a DataFrame: shard_id, path, rows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    day_count = calendar.monthrange(C.YEAR, C.MONTH)[1]
    counts = [0] * (day_count + 1)
    paths = {day: out_dir / f"day_{day:02d}.parquet" for day in range(1, day_count + 1)}
    with ExitStack() as stack:
        source = stack.enter_context(pq.ParquetFile(src))
        writers = {}
        for batch in source.iter_batches(batch_size=C.SHARD_WRITE_BATCH_SIZE,
                                         columns=C.COLUMNS, use_threads=False):
            table = pa.Table.from_batches([batch])
            pickup = table.column("tpep_pickup_datetime")
            in_month = pc.and_(pc.equal(pc.year(pickup), C.YEAR),
                               pc.equal(pc.month(pickup), C.MONTH))
            table = table.filter(in_month)
            days = pc.day(table.column("tpep_pickup_datetime"))
            for day in pc.unique(days).to_pylist():
                if day not in writers:
                    writers[day] = stack.enter_context(pq.ParquetWriter(paths[day], table.schema))
                part = table.filter(pc.equal(days, day))
                writers[day].write_table(part)
                counts[day] += part.num_rows
    return pd.DataFrame([
        {"shard_id": day, "path": str(paths[day]), "rows": counts[day]}
        for day in paths if counts[day]
    ], columns=["shard_id", "path", "rows"])


# --- Per-shard work (identical for sequential and parallel runs) ---------------
def _peak_rss_mb():
    """Peak resident memory of this process since it started, in MiB.

    On Linux this reads VmHWM from /proc, which (unlike ru_maxrss) is not
    inherited from the process that launched this one.
    """
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024  # kB -> MiB
    except OSError:
        pass
    try:
        import resource
    except ImportError:  # Windows does not provide getrusage.
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KB on Linux


def process_shard(shard_id, path, batch_size=C.BATCH_SIZE):
    """Clean one day shard batch by batch and return its partial aggregates."""
    t0 = time.perf_counter()
    partials, card_rows, clean_rows = [], 0, 0
    with pq.ParquetFile(path) as source:
        for batch in source.iter_batches(batch_size=batch_size, columns=C.COLUMNS, use_threads=False):
            df = pa.Table.from_batches([batch]).to_pandas(use_threads=False)
            df = df[df["payment_type"] == C.CARD_PAYMENT]
            card_rows += len(df)
            df, _ = apply_clean_steps(df)
            clean_rows += len(df)
            df = add_target(df)
            g = df.assign(hour=df["tpep_pickup_datetime"].dt.hour,
                          tip_pct=df["tip_amount"] / df["fare_amount"])
            partials.append(g.groupby(GROUP_KEYS).agg(
                trips=("tipped", "size"), tipped=("tipped", "sum"), tip_pct_sum=("tip_pct", "sum")))
    agg = pd.concat(partials).groupby(level=GROUP_KEYS).sum()
    return {"shard_id": shard_id, "card_rows": card_rows, "clean_rows": clean_rows,
            "agg": agg, "seconds": time.perf_counter() - t0,
            "pid": os.getpid(), "peak_rss_mb": _peak_rss_mb()}


def _process_task(task):
    return process_shard(*task)


# --- Merge ---------------------------------------------------------------------------
def merge_partials(results):
    """Merge partial sums/counts in fixed shard order, THEN compute rates."""
    results = sorted(results, key=lambda r: r["shard_id"])
    total = pd.concat([r["agg"] for r in results]).groupby(level=GROUP_KEYS).sum().sort_index()
    total["tip_rate"] = total["tipped"] / total["trips"]
    total["mean_tip_pct"] = total["tip_pct_sum"] / total["trips"]
    return total


def sequential_reference(clean_df):
    """Independent check: the same statistics computed directly from load_clean()."""
    g = clean_df.assign(PULocationID=clean_df["PULocationID"].astype("int64"),
                        hour=clean_df["tpep_pickup_datetime"].dt.hour,
                        tip_pct=clean_df["tip_amount"] / clean_df["fare_amount"])
    out = g.groupby(GROUP_KEYS).agg(trips=("tipped", "size"), tipped=("tipped", "sum"),
                                    tip_pct_sum=("tip_pct", "sum")).sort_index()
    out["tip_rate"] = out["tipped"] / out["trips"]
    out["mean_tip_pct"] = out["tip_pct_sum"] / out["trips"]
    return out


def assert_same_aggregates(actual, expected):
    """Check keys, exact counts and every floating-point result."""
    pd.testing.assert_index_equal(actual.index, expected.index, exact=False)
    for column in ("trips", "tipped"):
        np.testing.assert_array_equal(actual[column], expected[column], err_msg=column)
    for column in ("tip_pct_sum", "tip_rate", "mean_tip_pct"):
        np.testing.assert_allclose(actual[column], expected[column], rtol=1e-12, atol=0,
                                   err_msg=column)


# --- Runners -----------------------------------------------------------------------
def _tasks(shards, largest_first):
    s = shards.sort_values("rows", ascending=False, kind="stable") if largest_first else shards
    return list(zip(s["shard_id"], s["path"]))


def run_sequential(shards):
    """Plain loop in this process: the T1 baseline (no pool, no pickling)."""
    return [process_shard(sid, path) for sid, path in _tasks(shards, False)]


def run_parallel(shards, workers=None, pool=None, largest_first=True):
    """Same function on every shard with a process pool.

    pool: an already-running ProcessPoolExecutor (start-up cost excluded);
          if None, a new pool with `workers` processes is started (cost included).
    largest_first: hand out the biggest days first so no worker is left
          holding a big shard at the end (reduces load imbalance).
    """
    tasks = _tasks(shards, largest_first)
    if pool is not None:
        return list(pool.map(_process_task, tasks))
    with ProcessPoolExecutor(max_workers=workers) as new_pool:
        return list(new_pool.map(_process_task, tasks))


def run_data_parallel(shards, workers=0, pool=None):
    """Full workload incl. merge, so sequential and parallel timings cover identical work.

    workers == 0 and no pool -> plain sequential loop.
    """
    if pool is None and workers == 0:
        results = run_sequential(shards)
    else:
        results = run_parallel(shards, workers, pool=pool)
    return merge_partials(results), results


def load_balance(results):
    """Busy seconds per worker process: max/mean > 1 means some workers waited on others."""
    busy = pd.DataFrame(results).groupby("pid")["seconds"].sum()
    return {"workers_used": len(busy), "max_busy_s": busy.max(), "mean_busy_s": busy.mean(),
            "imbalance_max_over_mean": busy.max() / busy.mean()}


def _noop(_):
    return os.getpid()


def pool_startup_seconds(workers):
    """Overhead of starting `workers` processes and giving each a trivial task."""
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_noop, range(workers)))
    return time.perf_counter() - t0


# --- Non-determinism source 1: completion order ----------------------------------------
def completion_order_run(shards, workers, fix=False):
    """Collect results with as_completed.

    Without the fix, results are combined in the order workers finish, which
    changes between runs. The float total is accumulated in that order too.
    With the fix, results are sorted by shard_id (their tag) before combining.
    """
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_shard, sid, path) for sid, path in _tasks(shards, False)]
        results = [f.result() for f in as_completed(futures)]
    if fix:
        results.sort(key=lambda r: r["shard_id"])
    order = [r["shard_id"] for r in results]
    total = 0.0
    for r in results:
        for v in r["agg"]["tip_pct_sum"].to_numpy():
            total += v
    return order, total
