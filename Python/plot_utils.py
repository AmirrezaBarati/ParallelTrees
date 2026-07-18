import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"

COLORS = [
    "#0072B2",   # Blue
    "#E69F00",   # Orange
    "#009E73",   # Green
    "#D55E00",   # Vermillion
    "#CC79A7",   # Purple
    "#56B4E9",   # Sky Blue
    "#808080",   # Gray
    "#F0E442",   # Yellow
    "#882255",   # Wine
    "#000000",   # Black
]
LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5)), (0, (5, 5, 1, 5)), (0, (1, 1)), (0, (3, 3, 1, 3, 1, 3))]
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]


def add_setup_arg(parser):
    parser.add_argument(
        "-s",
        "--setup",
        choices=["fullmesh", "nvs"],
        default="fullmesh",
        help="Result set: fullmesh uses results/<collective>/ (default); "
        "nvs uses results/NVS/<collective>/.",
    )


def results_dir(collective, setup):
    """Return the results directory for a collective and machine setup."""
    if setup == "nvs":
        return RESULTS_ROOT / "NVS" / collective
    return RESULTS_ROOT / collective


def default_output_name(prefix, gpus, setup):
    """Build default PDF filename; include setup suffix for nvs."""
    suffix = f"_{setup}" if setup != "fullmesh" else ""
    return f"{prefix}_{gpus}_gpus{suffix}.pdf"


def add_chart_type_arg(parser):
    parser.add_argument(
        "-t",
        "--chart-type",
        choices=["bar", "line"],
        default="bar",
        help="Chart type: bar (default) or line.",
    )


def add_yscale_arg(parser):
    parser.add_argument(
        "-y",
        "--yscale",
        choices=["log", "linear"],
        default="log",
        help="Y-axis scale: log (default) or linear (normal latencies).",
    )


def setup_style():
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 10
    plt.rcParams["font.family"] = "Times New Roman"


def plot_algorithms(ax, x, series, chart_type, bar_width=0.14):
    """Plot algorithm series as grouped bars or lines."""
    n = len(series)
    if chart_type == "bar":
        offsets = np.arange(n) - (n - 1) / 2
        for i, (values, label) in enumerate(series):
            ax.bar(
                x + offsets[i] * bar_width,
                values,
                width=bar_width,
                color=COLORS[i],
                label=label,
            )
    else:
        for i, (values, label) in enumerate(series):
            ax.plot(
                x,
                values,
                color=COLORS[i],
                linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                marker=MARKERS[i % len(MARKERS)],
                linewidth=1.2,
                markersize=4,
                label=label,
            )


def apply_yscale(ax, series, yscale):
    """Configure y-axis as log or linear and set an appropriate label."""
    if yscale == "log":
        ax.set_yscale("log")
        ax.set_ylim(bottom=min(values.min() for values, _ in series) * 0.8)
        ax.set_ylabel("Latency (us, log scale)")
    else:
        ax.set_yscale("linear")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Latency (us)")
