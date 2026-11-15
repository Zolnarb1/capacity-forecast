"""Charts for the README.  Usage:  python -m capfc.plots"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from capfc import RESULTS_DIR, ROOT
from capfc.data import load_series, load_windows
from capfc.detect import DETECTORS, alarms

COLORS = {"seasonal (vs yesterday)": "#E45756", "rolling z-score": "#4C78A8",
          "isolation forest": "#54A24B"}


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def tradeoff_chart():
    sw = pd.read_csv(RESULTS_DIR / "detection_tradeoff.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for det, g in sw.groupby("detector"):
        g = g.sort_values("factor")
        ax.plot(g["false_alarms_per_week"], g["detection_rate"], "-o", color=COLORS[det], label=det, ms=4)
        d = g[g["factor"] == 1.0].iloc[0]
        ax.annotate("default", (d["false_alarms_per_week"], d["detection_rate"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=8, color=COLORS[det])
    ax.set_xlabel("false alarms per metric per week (pages you'd get for nothing)")
    ax.set_ylabel("share of labelled incidents detected")
    ax.set_ylim(0, 1)
    ax.set_title("Detecting incidents in 17 real AWS metrics: the alert-fatigue trade-off",
                 loc="left", fontsize=11)
    ax.legend(frameon=False)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "detection_tradeoff.png", dpi=160)


def example_chart(name="ec2_cpu_utilization_825cc2"):
    s = load_series(name)
    windows = load_windows(name)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s.index, s, color="#333333", lw=0.7)
    for a, b in windows:
        ax.axvspan(a, b, color="#F2A541", alpha=0.3, label="labelled incident window")
    for det, fn in DETECTORS.items():
        al = alarms(fn(s))
        ax.scatter(al[al].index, s[al], s=12, color=COLORS[det], label=f"alarm: {det}", zorder=3)
    ax.axvline(s.index[int(len(s) * 0.15)], color="#999999", ls="--", lw=0.8)
    ax.text(s.index[int(len(s) * 0.15)], ax.get_ylim()[1], " learning period ends", fontsize=8,
            color="#777777", va="top")
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), frameon=True, fontsize=8, loc="lower right")
    ax.set_title(f"{name}: CPU utilisation (%), 5-minute points", loc="left", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    _clean(ax)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "example_series.png", dpi=160)


def forecast_chart():
    fs = pd.read_csv(RESULTS_DIR / "forecast_per_series.csv")
    order = fs.groupby("model")["MASE"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [fs.loc[fs["model"] == m, "MASE"].clip(upper=4) for m in order]
    ax.boxplot(data, vert=False, tick_labels=list(order), widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#9ecae9"), medianprops=dict(color="#E45756"))
    ax.axvline(1, color="#999999", ls="--", lw=0.8)
    ax.set_xlabel("MASE on the last day (lower is better; values above 4 clipped)")
    ax.set_title("Day-ahead forecasts on 17 AWS metrics: the flat line wins", loc="left", fontsize=11)
    _clean(ax)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "forecast_mase.png", dpi=160)


if __name__ == "__main__":
    tradeoff_chart()
    example_chart()
    forecast_chart()
    print("charts saved to", RESULTS_DIR)
