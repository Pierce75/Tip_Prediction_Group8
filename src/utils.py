"""Timing and environment helpers shared by every section."""
import multiprocessing
import os
import platform
import subprocess
import sys
import time

import numpy as np

from . import config as C


def time_it(fn, *args, repeats=C.N_REPEATS, warmup=True, check=None, **kwargs):
    """Run fn once as warm-up (not timed), then time `repeats` runs.

    Returns (last_result, mean_seconds, std_seconds, all_times).
    """
    if warmup:
        warm_result = fn(*args, **kwargs)
        if check is not None:
            check(warm_result)
        del warm_result
    times = []
    for repeat in range(repeats):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - start)
        if check is not None:
            check(result)
        if repeat < repeats - 1:
            del result
    return result, float(np.mean(times)), float(np.std(times)), times


def cpu_model():
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def usable_cpus():
    """CPUs this process may use (respects the Slurm --cpus-per-task limit on Rockhill)."""
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    allocation = os.environ.get("SLURM_CPUS_PER_TASK")
    return min(count, int(allocation)) if allocation else count


def environment_info():
    import matplotlib
    import pandas
    import pyarrow
    import scipy
    import sklearn
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "os.cpu_count()": os.cpu_count(),
        "usable_cpus": usable_cpus(),
        "execution_host": "Rockhill via Slurm" if os.environ.get("SLURM_JOB_ID") else "Local execution environment",
        "process_start_method": multiprocessing.get_start_method(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "-"),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }
