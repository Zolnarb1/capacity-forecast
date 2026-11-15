"""Server-metric anomaly detection and capacity forecasting on real AWS CloudWatch data."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
