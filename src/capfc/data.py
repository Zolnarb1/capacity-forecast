"""Download the AWS CloudWatch part of the Numenta Anomaly Benchmark (NAB, Lavin & Ahmad 2015).

17 real production metrics (EC2 CPU, disk writes, network in, ELB requests, RDS CPU),
one point every 5 minutes, about two weeks each, with expert-labelled anomaly windows.
License: see https://github.com/numenta/NAB (permissive, MIT-style).
"""
import json
import urllib.request

import pandas as pd

from capfc import RAW_DIR

BASE = "https://raw.githubusercontent.com/numenta/NAB/master"
SERIES = [
    "ec2_cpu_utilization_24ae8d", "ec2_cpu_utilization_53ea38", "ec2_cpu_utilization_5f5533",
    "ec2_cpu_utilization_77c1ca", "ec2_cpu_utilization_825cc2", "ec2_cpu_utilization_ac20cd",
    "ec2_cpu_utilization_c6585a", "ec2_cpu_utilization_fe7f93", "ec2_disk_write_bytes_1ef3de",
    "ec2_disk_write_bytes_c0d644", "ec2_network_in_257a54", "ec2_network_in_5abac7",
    "elb_request_count_8c0756", "grok_asg_anomaly", "iio_us-east-1_i-a2eb1cd9_NetworkIn",
    "rds_cpu_utilization_cc0c53", "rds_cpu_utilization_e47b3b",
]


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in SERIES:
        target = RAW_DIR / f"{name}.csv"
        if not target.exists():
            urllib.request.urlretrieve(f"{BASE}/data/realAWSCloudwatch/{name}.csv", target)
    labels = RAW_DIR / "combined_windows.json"
    if not labels.exists():
        urllib.request.urlretrieve(f"{BASE}/labels/combined_windows.json", labels)


def load_series(name: str) -> pd.Series:
    df = pd.read_csv(RAW_DIR / f"{name}.csv", parse_dates=["timestamp"])
    s = df.set_index("timestamp")["value"].astype(float)
    # Some series have duplicate or irregular timestamps: put them on a regular 5-minute grid.
    return s.resample("5min").mean().interpolate(limit_direction="both")


def load_windows(name: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    all_windows = json.loads((RAW_DIR / "combined_windows.json").read_text())
    key = f"realAWSCloudwatch/{name}.csv"
    return [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in all_windows.get(key, [])]


def load_all() -> dict[str, tuple[pd.Series, list]]:
    download()
    return {name: (load_series(name), load_windows(name)) for name in SERIES}
