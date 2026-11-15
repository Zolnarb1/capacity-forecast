"""Run both studies on all 17 AWS series.  Usage:  python -m capfc.run"""
import pandas as pd

from capfc import RESULTS_DIR
from capfc.data import load_all
from capfc.detect import DETECTORS, alarms, evaluate
from capfc.forecast import forecast_last_day, mase

MODELS = ["naive", "seasonal naive", "holt-winters", "gradient boosting"]


def main() -> None:
    data = load_all()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    det_rows, fc_rows, examples, sweep_rows = [], [], {}, []
    for name, (s, windows) in data.items():
        for det_name, det in DETECTORS.items():
            score = det(s)
            r = evaluate(alarms(score), windows)
            det_rows.append({"series": name, "detector": det_name, **r})
            for factor in (0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0):
                sweep_rows.append({"detector": det_name, "factor": factor,
                                   **evaluate(alarms(score, factor), windows)})
        fc = forecast_last_day(s)
        for m in MODELS:
            fc_rows.append({"series": name, "model": m, "MASE": mase(fc[m], fc["_actual"], fc["_scale"])})
        examples[name] = fc

    det = pd.DataFrame(det_rows)
    det.to_csv(RESULTS_DIR / "detection_per_series.csv", index=False, float_format="%.2f")
    summary = det.groupby("detector").agg(windows=("windows", "sum"), detected=("detected", "sum"),
                                          false_alarms=("false_alarm_events", "sum"), days=("days", "sum"))
    summary["detection_rate"] = summary["detected"] / summary["windows"]
    # days are summed over all series, so this is false alarms per metric per week
    summary["false_alarms_per_week"] = summary["false_alarms"] / summary["days"] * 7
    summary = summary.sort_values("detection_rate", ascending=False)
    summary.to_csv(RESULTS_DIR / "detection_summary.csv", float_format="%.3f")
    print(f"Anomaly detection on {len(data)} AWS metrics (event level)")
    print(summary.round(3).to_string())

    sw = pd.DataFrame(sweep_rows).groupby(["detector", "factor"]).sum().reset_index()
    sw["detection_rate"] = sw["detected"] / sw["windows"]
    sw["false_alarms_per_week"] = sw["false_alarm_events"] / sw["days"] * 7
    sw.to_csv(RESULTS_DIR / "detection_tradeoff.csv", index=False, float_format="%.3f")

    fc =pd.DataFrame(fc_rows)
    fc.to_csv(RESULTS_DIR / "forecast_per_series.csv", index=False, float_format="%.3f")
    fsum = fc.groupby("model")["MASE"].agg(["median", "mean"]).sort_values("median")
    wins = fc.loc[fc.groupby("series")["MASE"].idxmin(), "model"].value_counts()
    fsum["series_won"] = wins.reindex(fsum.index).fillna(0).astype(int)
    fsum.to_csv(RESULTS_DIR / "forecast_summary.csv", float_format="%.3f")
    print("\nDay-ahead forecast (MASE, lower is better; 1.0 = 'same as yesterday')")
    print(fsum.round(3).to_string())

    pd.to_pickle(examples, RESULTS_DIR.parent / "data" / "forecast_examples.pkl")


if __name__ == "__main__":
    main()
