"""Loading, cleaning and splitting the taxi data (rubric criterion 1).

Shared entry point for everyone: ``df, steps = load_clean()``.
The cleaning rules live in CLEAN_STEPS so the sequential loader and the
data-parallel shard workers (src/parallel.py) apply exactly the same logic.
"""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import pandas as pd
import pyarrow.parquet as pq

from . import config as C


def ensure_data(path=C.DATA_FILE, url=C.DATA_URL):
    """Download the data file if it is missing, so the notebook runs on a fresh machine."""
    if path.exists():
        return f"Found {path.name} ({path.stat().st_size / 2**20:.0f} MiB)."
    if path != (C.DATA_DIR / C.DATA_NAME).resolve():
        raise FileNotFoundError(f"Could not find the January Parquet file at TIP_DATA_FILE={path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        downloaded = Path(directory) / path.name
        urlretrieve(url, downloaded)
        verify_data(downloaded)
        downloaded.replace(path)
    return f"Downloaded {path.name} from {url}."


def verify_data(path=C.DATA_FILE):
    """Confirm the benchmark is using exactly the documented January file."""
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != C.DATA_SHA256:
        raise ValueError(f"Unexpected data file {path}. Supply the project’s January 2026 Parquet file.")
    return digest


def threshold_evidence(path=C.DATA_FILE):
    """Distribution facts behind each cleaning threshold, computed on card trips in the target month."""
    metadata = pq.ParquetFile(path).metadata
    distance_index = metadata.schema.names.index("trip_distance")
    raw_stats = [metadata.row_group(i).column(distance_index).statistics
                 for i in range(metadata.num_row_groups)]
    raw_max = max(column.max for column in raw_stats if column and column.has_min_max)
    cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "fare_amount"]
    d = pq.read_table(path, columns=cols, filters=[("payment_type", "==", C.CARD_PAYMENT)]).to_pandas()
    t = d["tpep_pickup_datetime"]
    d = d[(t.dt.year == C.YEAR) & (t.dt.month == C.MONTH)]
    dur = duration_min(d)
    dist = d["trip_distance"]
    rows = [
        ("Maximum distance in raw file (mi)", raw_max, "all payment types; Parquet footer"),
        ("Maximum distance in January card trips (mi)", dist.max(), "after card and date filters"),
        ("Card-trip distance 99.99th percentile (mi)", dist.quantile(0.9999), f"threshold {C.MAX_DISTANCE_MI} mi"),
        ("January card trips with distance > threshold", (dist > C.MAX_DISTANCE_MI).sum(), "flagged"),
        ("duration 99.9th percentile (min)", dur.quantile(0.999), f"threshold {C.MAX_DURATION_MIN} min"),
        ("trips with duration <= 0 min", (dur <= 0).sum(), "flagged"),
        ("trips with fare < 0", (d["fare_amount"] < 0).sum(), "flagged"),
        ("trips with fare > threshold", (d["fare_amount"] > C.MAX_FARE).sum(), f"threshold ${C.MAX_FARE:g}"),
    ]
    return pd.DataFrame(rows, columns=["statistic", "value", "note"])


def parquet_summary(path=C.DATA_FILE):
    """Read only the Parquet footer: row count, row groups, schema. No data is loaded."""
    pf = pq.ParquetFile(path)
    md = pf.metadata
    groups = pd.DataFrame(
        {"row_group": range(md.num_row_groups),
         "rows": [md.row_group(i).num_rows for i in range(md.num_row_groups)]}
    )
    return md.num_rows, groups, pf.schema_arrow


def payment_type_counts(path=C.DATA_FILE):
    """Read a single column to show why we keep card trips only."""
    s = pq.read_table(path, columns=["payment_type"]).column(0).to_pandas()
    names = {0: "Flex Fare", 1: "Credit card", 2: "Cash", 3: "No charge",
             4: "Dispute", 5: "Unknown", 6: "Voided"}
    out = s.value_counts().sort_index().rename("trips").to_frame()
    out.index = [f"{i} ({names.get(i, '?')})" for i in out.index]
    return out


def duration_min(df):
    return (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60


# (description, function returning a boolean mask of rows to KEEP), applied in order.
CLEAN_STEPS = [
    (f"Pickup in {C.YEAR}-{C.MONTH:02d} (drops Dec/Feb spill-over)",
     lambda d: (d["tpep_pickup_datetime"].dt.year == C.YEAR)
     & (d["tpep_pickup_datetime"].dt.month == C.MONTH)),
    ("trip_distance > 0",
     lambda d: d["trip_distance"] > 0),
    (f"trip_distance <= {C.MAX_DISTANCE_MI} mi (meter errors)",
     lambda d: d["trip_distance"] <= C.MAX_DISTANCE_MI),
    (f"{C.MIN_FARE:g} < fare_amount <= {C.MAX_FARE:g}",
     lambda d: (d["fare_amount"] > C.MIN_FARE) & (d["fare_amount"] <= C.MAX_FARE)),
    (f"{C.MIN_DURATION_MIN:g} <= duration <= {C.MAX_DURATION_MIN} min",
     lambda d: duration_min(d).between(C.MIN_DURATION_MIN, C.MAX_DURATION_MIN)),
    (f"average speed <= {C.MAX_SPEED_MPH} mph",
     lambda d: d["trip_distance"] / (duration_min(d) / 60) <= C.MAX_SPEED_MPH),
    ("tip_amount >= 0",
     lambda d: d["tip_amount"] >= 0),
]


def apply_clean_steps(df):
    """Apply CLEAN_STEPS in order. Returns (clean_df, [rows_after for each step])."""
    counts = []
    for _, keep in CLEAN_STEPS:
        df = df[keep(df)]
        counts.append(len(df))
    return df, counts


def add_target(df):
    """Duration column and the binary target. No categorical conversion here."""
    return df.assign(duration_min=duration_min(df), tipped=df["tip_amount"] > 0)


def load_clean(path=C.DATA_FILE):
    """Load the card trips for the target month and apply every cleaning step.

    Returns
    -------
    df : cleaned DataFrame with a boolean ``tipped`` target column
    steps : DataFrame with the row count after each cleaning step
    """
    total_rows, _, _ = parquet_summary(path)

    # Column pruning + predicate pushdown: only card trips, only needed columns.
    # This also drops every Flex Fare trip (payment_type 0), which are the rows
    # with null passenger_count / RatecodeID / surcharges.
    table = pq.read_table(path, columns=C.COLUMNS,
                          filters=[("payment_type", "==", C.CARD_PAYMENT)])
    df = table.to_pandas()
    card_rows = len(df)

    df, counts = apply_clean_steps(df)
    names = ["Raw file (from Parquet footer)", "Card payments only (drops cash, Flex Fare, disputes)"]
    names += [name for name, _ in CLEAN_STEPS]
    rows = [total_rows, card_rows] + counts
    steps = pd.DataFrame({"step": names, "rows_after": rows})
    steps["removed"] = (-steps["rows_after"].diff()).fillna(0).astype(int)

    df = add_target(df)
    for col in C.CATEGORICAL:
        df[col] = df[col].astype("int64").astype("category")

    df = df.sort_values("tpep_pickup_datetime", kind="stable").reset_index(drop=True)
    return df, steps


def time_split(df):
    """Time-based train / validation / test split (never random)."""
    t = df["tpep_pickup_datetime"]
    train = df[t < C.TRAIN_END]
    val = df[(t >= C.TRAIN_END) & (t < C.VAL_END)]
    test = df[t >= C.VAL_END]
    return train, val, test
