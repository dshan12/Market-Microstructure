"""RQ7 visualization: market making strategy comparison."""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_rq7_comparison(results_file: str, output_path: str = "plots/rq7_comparison.png"):
    with open(results_file) as f:
        data = json.load(f)

    strategies = [k for k in data if k not in ("skew_grid", "onesided_grid")]
    names = []
    pnls = []
    sharpes = []
    colors = []
    for s in strategies:
        names.append(s)
        pnls.append(data[s]["stats"]["total_pnl"])
        sharpes.append(data[s]["stats"]["sharpe_ratio"])
        colors.append("#2ecc71" if pnls[-1] > 0 else "#e74c3c")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PnL bar chart
    axes[0].bar(names, pnls, color=colors, edgecolor="white", width=0.6)
    for i, (n, p) in enumerate(zip(names, pnls)):
        axes[0].annotate(f"${p:+,.0f}", (i, p), textcoords="offset points",
                         xytext=(0, 8 if p > 0 else -12), ha="center", fontsize=9)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("Total PnL ($)", fontsize=12)
    axes[0].set_title("Strategy PnL Comparison", fontsize=14, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=30)

    # Sharpe bar chart
    colors_sharpe = ["#2ecc71" if s > 0 else "#e74c3c" for s in sharpes]
    axes[1].bar(names, sharpes, color=colors_sharpe, edgecolor="white", width=0.6)
    for i, (n, s) in enumerate(zip(names, sharpes)):
        axes[1].annotate(f"{s:+.1f}", (i, s), textcoords="offset points",
                         xytext=(0, 8 if s > 0 else -12), ha="center", fontsize=9)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Sharpe Ratio", fontsize=12)
    axes[1].set_title("Risk-Adjusted Performance", fontsize=14, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")

    # Grid search plots
    for grid_key, x_label, title_prefix in [
        ("skew_grid", "Skew Strength", "Adaptive Skew"),
        ("onesided_grid", "Threshold", "One-Sided Threshold"),
    ]:
        if grid_key not in data:
            continue
        grid = pd.DataFrame(data[grid_key])
        fig, ax = plt.subplots(figsize=(8, 5))
        x = grid[grid.columns[0]] if grid_key == "onesided_grid" else grid["skew"]
        ax.plot(x, grid["total_pnl"], "o-", color="#3498db", linewidth=2, markersize=8)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        for xi, pnl in zip(x, grid["total_pnl"]):
            ax.annotate(f"${pnl:+,.0f}", (xi, pnl), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Total PnL ($)", fontsize=12)
        ax.set_title(f"{title_prefix} Grid Search", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_path.replace(".png", f"_{grid_key}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path.replace('.png', f'_{grid_key}.png')}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/rq7_btc.json")
    parser.add_argument("--output", default="plots/rq7_comparison.png")
    args = parser.parse_args()
    plot_rq7_comparison(args.input, args.output)


if __name__ == "__main__":
    main()
