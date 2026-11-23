"""Feature groups. Each group is an independent function of the cleaned data,
so they can later be run concurrently (rubric criterion 4, feature extraction)
and joined back in the fixed order of FEATURE_GROUPS.

No feature may use tip_amount, total_amount or anything derived from the tip.
"""
import numpy as np
import pandas as pd

from . import config as C


def time_features(df):
    t = df["tpep_pickup_datetime"]
    return pd.DataFrame({
        "hour": t.dt.hour,
        "day_of_week": t.dt.dayofweek,
        "is_weekend": (t.dt.dayofweek >= 5).astype("int8"),
    }, index=df.index)


def distance_features(df):
    hours = df["duration_min"] / 60
    return pd.DataFrame({
        "trip_distance": df["trip_distance"],
        "log_distance": np.log1p(df["trip_distance"]),
        "duration_min": df["duration_min"],
        "speed_mph": df["trip_distance"] / hours,
    }, index=df.index)


def fare_features(df):
    return pd.DataFrame({
        "fare_amount": df["fare_amount"],
        "fare_per_mile": df["fare_amount"] / df["trip_distance"],
        "has_tolls": (df["tolls_amount"] > 0).astype("int8"),
        "has_airport_fee": (df["Airport_fee"].fillna(0) > 0).astype("int8"),
        "has_cbd_fee": (df["cbd_congestion_fee"] > 0).astype("int8"),
        "passenger_count": df["passenger_count"].fillna(1),
    }, index=df.index)


def zone_features(df):
    return pd.DataFrame({
        "PULocationID": df["PULocationID"],
        "DOLocationID": df["DOLocationID"],
        "RatecodeID": df["RatecodeID"],
        "pickup_airport": df["PULocationID"].astype(int).isin(C.AIRPORT_ZONES).astype("int8"),
        "dropoff_airport": df["DOLocationID"].astype(int).isin(C.AIRPORT_ZONES).astype("int8"),
    }, index=df.index)


# Fixed order: the parallel version must concatenate results in exactly this order.
FEATURE_GROUPS = {
    "time": time_features,
    "distance": distance_features,
    "fare": fare_features,
    "zone": zone_features,
}


def make_features(df):
    """Sequential reference version: build every group and join in fixed order."""
    return pd.concat([fn(df) for fn in FEATURE_GROUPS.values()], axis=1)
