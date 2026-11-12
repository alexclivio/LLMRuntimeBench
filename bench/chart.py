import csv
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Colors: one per series, kept the same across every chart
BLUE = "#2a78d6"    # vLLM / tuned setting
ORANGE = "#eb6834"  # TensorRT-LLM
GRAY = "#8a8a85"    # default setting (before the fix)
INK = "#1f1f1e"
MUTED = "#6b6b66"


def load_rows():
    with open(RESULTS_DIR / "results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["concurrency_target"] = int(r["concurrency_target"])
        r["users"] = int(r["users"])
        for key in ["ttft_median_ms", "ttft_worst_ms", "tps_per_user", "total_tps"]:
            r[key] = float(r[key])
    return rows


def series(rows, engine, target):
    picked = [r for r in rows if r["engine"] == engine and r["concurrency_target"] == target]
    return sorted(picked, key=lambda r: r["users"])


def style_axis(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel("Users sending requests at the same time", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_xticks([1, 8, 32])
    ax.grid(axis="y", color="#e6e6e2", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#c9c9c4")
    ax.tick_params(colors=MUTED)


def plot_line(ax, data, key, color, label, dashed=False):
    xs = [r["users"] for r in data]
    ys = [r[key] for r in data]
    ax.plot(xs, ys, color=color, linewidth=2, linestyle="--" if dashed else "-",
            marker="o", markersize=7, label=label)
    # Label only the last point, not every point
    ax.annotate(f"{ys[-1]:,.0f}", (xs[-1], ys[-1]), textcoords="offset points",
                xytext=(8, 0), va="center", fontsize=9, color=INK)


def chart_concurrency_fix(rows):
    before = series(rows, "vllm", 1)
    after = series(rows, "vllm", 32)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    plot_line(left, before, "total_tps", GRAY, "Default (concurrency target 1)", dashed=True)
    plot_line(left, after, "total_tps", BLUE, "Tuned (concurrency target 32)")
    style_axis(left, "Total throughput (higher is better)", "Tokens per second, all users")
    left.set_ylim(bottom=0)

    plot_line(right, before, "ttft_median_ms", GRAY, "Default (concurrency target 1)", dashed=True)
    plot_line(right, after, "ttft_median_ms", BLUE, "Tuned (concurrency target 32)")
    style_axis(right, "Median wait for first token (lower is better)", "Milliseconds (log scale)")
    right.set_yscale("log")

    left.legend(loc="upper left", frameon=False, fontsize=10)
    fig.suptitle("vLLM on one L4: effect of Baseten's concurrency target",
                 x=0.01, ha="left", fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "concurrency_fix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def chart_engine_comparison(rows):
    vllm = series(rows, "vllm", 32)
    trtllm = series(rows, "trtllm", 32)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    plot_line(left, vllm, "total_tps", BLUE, "vLLM")
    plot_line(left, trtllm, "total_tps", ORANGE, "TensorRT-LLM")
    style_axis(left, "Total throughput (higher is better)", "Tokens per second, all users")
    left.set_ylim(bottom=0)

    plot_line(right, vllm, "ttft_median_ms", BLUE, "vLLM")
    plot_line(right, trtllm, "ttft_median_ms", ORANGE, "TensorRT-LLM")
    style_axis(right, "Median wait for first token (lower is better)", "Milliseconds")
    right.set_ylim(bottom=0)

    left.legend(loc="upper left", frameon=False, fontsize=10)
    fig.suptitle("vLLM vs TensorRT-LLM: Qwen 2.5 3B on one L4 (concurrency target 32)",
                 x=0.01, ha="left", fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "engine_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rows = load_rows()
    chart_concurrency_fix(rows)
    chart_engine_comparison(rows)
    print(f"Saved charts to {RESULTS_DIR}")
