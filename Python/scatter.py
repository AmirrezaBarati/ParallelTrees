import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import (
    add_chart_type_arg,
    add_setup_arg,
    add_yscale_arg,
    apply_yscale,
    default_output_name,
    plot_algorithms,
    results_dir,
    setup_style,
)

START_INDEX = 6
XLABELS = ["1MiB", "2MiB", "4MiB", "8MiB", "16MiB", "32MiB"]
ALGORITHMS = (
    ("basiclinear", "Basic Linear"),
    ("binomial", "Binomial"),
    ("linear_nb", "Linear_nb"),
    ("multirail", "Proposed"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot MPI scatter latency results.")
    parser.add_argument(
        "gpus",
        type=int,
        choices=[16, 32, 64],
        help="Number of GPUs to plot (16, 32, or 64).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output PDF path (default: scatter_<gpus>_gpus.pdf next to this script).",
    )
    add_chart_type_arg(parser)
    add_setup_arg(parser)
    add_yscale_arg(parser)
    return parser.parse_args()


def read_latencies_us(filename):
    """Read avg latency (2nd column) in microseconds."""
    values = []
    with open(filename, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{filename}:{line_no}: expected size and latency, got {line!r}")
            values.append(float(parts[1]))
    return np.array(values)


def load_scatter_latencies(gpu_count, setup):
    """Load sliced scatter latencies (us) for all algorithms at a given GPU count."""
    base = results_dir("scatter", setup)
    results_path = base / f"gpu_{gpu_count}"
    return {
        name: read_latencies_us(results_path / f"{name}.txt")[START_INDEX:]
        for name, _ in ALGORITHMS
    }


def plot_scatter(latencies, output_path, chart_type, yscale):
    setup_style()

    fig_width = 7.16  # IEEE two-column width (inches)
    series = [(latencies[name], label) for name, label in ALGORITHMS]
    x = np.arange(len(series[0][0]))

    fig, ax = plt.subplots(figsize=(0.55 * fig_width, 2.0))
    plot_algorithms(ax, x, series, chart_type)

    apply_yscale(ax, series, yscale)

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_xlabel("Message Size")

    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS)
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")
    ax.legend(loc="upper left", fontsize=8, ncol=2, bbox_to_anchor=(0.01, 1))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    args = parse_args()
    latencies = load_scatter_latencies(args.gpus, args.setup)

    for name, _ in ALGORITHMS:
        print(f"{name} ({args.setup}, {args.gpus} GPUs, us):", latencies[name].tolist())

    output_path = args.output or Path(__file__).resolve().parent / default_output_name(
        "scatter", args.gpus, args.setup
    )
    plot_scatter(latencies, output_path, args.chart_type, args.yscale)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
