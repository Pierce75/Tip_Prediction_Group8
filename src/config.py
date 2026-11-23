"""All configuration, seeds and thresholds in one place (rubric criterion 6).

Change values here, never inside notebook cells or other modules.
"""
import os
from pathlib import Path

# --- Reproducibility --------------------------------------------------------
SEED = 4207

# --- Data -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
DATA_NAME = "yellow_tripdata_2026-01.parquet"
DATA_FILE = Path(os.environ.get("TIP_DATA_FILE", DATA_DIR / DATA_NAME)).expanduser().resolve()
DATA_URL = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{DATA_NAME}"
DATA_SOURCE = "NYC TLC Yellow Taxi Trip Records, January 2026 (Parquet)"
DATA_SHA256 = "8b3933fe6f0d7b6d8826613c0dd724edc680ff7c49e2bd4c7635c05102728637"
YEAR, MONTH = 2026, 1

# Only the columns we need are read (column pruning).
# total_amount is deliberately NOT read: it includes the tip, so it would leak the target.
COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "tolls_amount",
    "Airport_fee",
    "cbd_congestion_fee",
    "tip_amount",  # used ONLY to build the target, then excluded from features
]

CARD_PAYMENT = 1  # tips are only recorded for card payments

# --- Cleaning thresholds (each one is justified in the notebook) -------------
MAX_DISTANCE_MI = 100   # 99.99th percentile of card trips is ~54 mi
MIN_FARE = 0.0          # fare must be strictly positive (removes negative and zero fares)
MAX_FARE = 500.0        # only a handful of trips exceed this; they are not plausible metered fares
MIN_DURATION_MIN = 1.0  # trips shorter than 1 minute are cancellations / meter errors
MAX_DURATION_MIN = 180  # 99.9th percentile is ~139 min; longer trips are meters left running
MAX_SPEED_MPH = 80      # high-speed cutoff used to flag distance or duration errors

# --- Time-based split (no random split) -------------------------------------
# train: 1-17 Jan, validation: 18-24 Jan, test: 25-31 Jan
TRAIN_END = "2026-01-18"  # exclusive
VAL_END = "2026-01-25"    # exclusive

# --- Features ----------------------------------------------------------------
CATEGORICAL = ["PULocationID", "DOLocationID", "RatecodeID"]
AIRPORT_ZONES = {1, 132, 138}  # EWR, JFK, LaGuardia

# --- Parallel experiments ----------------------------------------------------
WORKER_COUNTS = [1, 2, 4, 8]
N_REPEATS = 3   # timed repeats after 1 warm-up run
THREADS_PER_WORKER = "1"  # pin BLAS/OpenMP threads so workers don't oversubscribe cores

WORK_DIR = Path(os.environ.get("TIP_WORK_DIR", ROOT / "data")).expanduser().resolve()
SHARD_DIR = WORK_DIR / "shards"         # one Parquet file per pickup day
CACHE_DIR = WORK_DIR / "cache"          # cleaned data / features shared with workers
BATCH_SIZE = 32_768                    # rows per worker batch
SHARD_WRITE_BATCH_SIZE = 262_144       # rows per batch when building shards

# Hyperparameter search (grid) and random-search demo
GRID_C = [0.01, 0.1, 1.0, 10.0]
GRID_CLASS_WEIGHT = ["none", "balanced"]
SEARCH_TRAIN_ROWS = 200_000
SEARCH_VAL_ROWS = 100_000
RANDOM_TRIALS = 8
RANDOM_SAMPLE_SIZE = 50_000

OUTPUT_DIR = Path(os.environ.get("TIP_OUTPUT_DIR", ROOT / "outputs")).expanduser().resolve()
