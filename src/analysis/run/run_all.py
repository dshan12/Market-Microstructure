"""
Master script: run all RQ1-RQ6 analyses and generate all plots on processed data.
"""

import pandas as pd
import numpy as np
import json
import logging
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from src.analysis.run.market_making import MarketMakingSimulator, BasicMarketMaker, AdaptiveMarketMaker

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def load_and_enrich(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["depth_imbalance"] = (df["bid_volume"] - df["ask_volume"]) / (df["bid_volume"] + df["ask_volume"])
    df["spread"] = df["best_ask"] - df["best_bid"]
    df["spread_bps"] = (df["spread"] / df["best_bid"]) * 10000
    return df


# ── RQ1 ─────────────────────────────────────────────────────────────
def rq1_correlations(df: pd.DataFrame) -> dict:
    result = {}
    for h in [1, 5, 10, 30, 60, 300]:
        col = f"return_{h}s"
        if col in df.columns:
            result[f"{h}s"] = round(float(df["depth_imbalance"].corr(df[col])), 4)
    return result


# ── RQ2 ─────────────────────────────────────────────────────────────
def rq2_decay(corrs: dict) -> dict:
    horizons = np.array([int(k.replace("s", "")) for k in corrs])
    values = np.array(list(corrs.values()))
    def exp_decay(t, a, b):
        return a * np.exp(-b * t)
    try:
        popt, _ = curve_fit(exp_decay, horizons, values, p0=[max(values), 0.01], maxfev=10000)
        half_life = float(np.log(2) / popt[1])
        return {"half_life_s": round(half_life, 1), "decay_rate_b": round(float(popt[1]), 4), "amplitude_a": round(float(popt[0]), 4)}
    except Exception as e:
        return {"error": str(e)}


# ── RQ3 ─────────────────────────────────────────────────────────────
def rq3_regime(df: pd.DataFrame) -> dict:
    result = {}
    for label in ["spread_bps", "volatility"]:
        if label == "spread_bps":
            vals = df["spread_bps"]
        else:
            df["_vol"] = df["return_1s"].rolling(20).std()
            vals = df["_vol"].fillna(df["_vol"].median())
        quintile = pd.qcut(vals.rank(method="first"), 5, labels=False)
        for q in range(5):
            sub = df[quintile == q]
            c = sub["depth_imbalance"].corr(sub["return_10s"])
            result[f"{label}_q{q}"] = {"corr": round(float(c), 4), "n": len(sub)}
    if "_vol" in df.columns:
        del df["_vol"]
    return result


# ── RQ4 ─────────────────────────────────────────────────────────────
def rq4_market_making(df: pd.DataFrame) -> dict:
    sim = MarketMakingSimulator(df)
    strategies = [BasicMarketMaker(), AdaptiveMarketMaker(skew_strength=0.5, vol_mult=2.0)]
    results = sim.run(strategies)
    return {name: {"stats": r["stats"], "trades": r["trades"]} for name, r in results.items()}


# ── RQ5 ─────────────────────────────────────────────────────────────
def rq5_granger(df: pd.DataFrame) -> dict:
    data = df[["return_10s", "depth_imbalance"]].dropna()
    gc = grangercausalitytests(data, maxlag=5, verbose=False)
    return {f"lag_{lag}": {"ssr_p": round(float(gc[lag][0]["ssr_chi2test"][1]), 6)} for lag in range(1, 6)}


# ── RQ6 ─────────────────────────────────────────────────────────────
def rq6_velocity(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["imbalance_velocity"] = df["depth_imbalance"].diff(5) / 5
    df_clean = df.dropna(subset=["return_10s", "depth_imbalance", "imbalance_velocity"])

    y = df_clean["return_10s"]
    X1 = add_constant(df_clean["depth_imbalance"])
    X2 = add_constant(df_clean[["depth_imbalance", "imbalance_velocity"]])
    m1 = OLS(y, X1).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    m2 = OLS(y, X2).fit(cov_type="HAC", cov_kwds={"maxlags": 10})

    med = df_clean["depth_imbalance"].median()
    high = df_clean[df_clean["depth_imbalance"] > med]
    low = df_clean[df_clean["depth_imbalance"] <= med]

    return {
        "level_corr": round(float(df_clean["depth_imbalance"].corr(df_clean["return_10s"])), 4),
        "velocity_corr": round(float(df_clean["imbalance_velocity"].corr(df_clean["return_10s"])), 4),
        "level_r2": round(float(m1.rsquared), 6),
        "level_vel_r2": round(float(m2.rsquared), 6),
        "r2_pct_change": round(float((m2.rsquared / m1.rsquared - 1) * 100), 2),
        "level_coef": round(float(m1.params.iloc[1]), 8),
        "level_tstat": round(float(m1.tvalues.iloc[1]), 2),
        "level_pval": round(float(m1.pvalues.iloc[1]), 6),
        "vel_coef": round(float(m2.params.iloc[2]), 8),
        "vel_tstat": round(float(m2.tvalues.iloc[2]), 2),
        "vel_pval": round(float(m2.pvalues.iloc[2]), 6),
        "high_state_vel_corr": round(float(high["imbalance_velocity"].corr(high["return_10s"])), 4),
        "low_state_vel_corr": round(float(low["imbalance_velocity"].corr(low["return_10s"])), 4),
        "n_obs": len(df_clean),
    }


# ── PLOTS ────────────────────────────────────────────────────────────
def plot_signal_decay(corrs, output_dir):
    horizons = sorted([int(k.replace("s", "")) for k in corrs])
    values = [corrs[f"{h}s"] for h in horizons]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(horizons, values, "o-", color="#2c3e50", linewidth=2, markersize=8)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    for h, v in zip(horizons, values):
        ax.annotate(f"{v:.3f}", (h, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel("Forecast Horizon (seconds)", fontsize=12)
    ax.set_ylabel("Correlation (Imbalance, Return)", fontsize=12)
    ax.set_title("Signal Decay: Imbalance-Return Correlation by Horizon", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/signal_decay_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_signal_by_regime(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, (label, ax) in enumerate(zip(["spread_bps", "volatility"], axes)):
        if label == "spread_bps":
            vals = df["spread_bps"]
        else:
            df["_vol"] = df["return_1s"].rolling(20).std()
            vals = df["_vol"].fillna(df["_vol"].median())
        quintile = pd.qcut(vals.rank(method="first"), 5, labels=False)
        corrs = [df[quintile == q]["depth_imbalance"].corr(df[quintile == q]["return_10s"]) for q in range(5)]
        colors = ["#3498db", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
        ax.bar(range(5), corrs, color=colors, edgecolor="white")
        for q, c in enumerate(corrs):
            ax.annotate(f"{c:.3f}", (q, c), textcoords="offset points", xytext=(0, 5), ha="center")
        ax.set_xlabel("Quintile", fontsize=11)
        ax.set_ylabel("Correlation", fontsize=11)
        title = "Spread" if label == "spread_bps" else "Volatility"
        ax.set_title(f"Signal by {title} Quintile", fontsize=12, fontweight="bold")
        ax.set_xticks(range(5))
    if "_vol" in df.columns:
        del df["_vol"]
    fig.tight_layout()
    fig.savefig(f"{output_dir}/signal_by_regime.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_market_making(mm_results, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    colors = {"Basic": "#e74c3c", "Adaptive": "#3498db"}
    for name, res in mm_results.items():
        trades = res["trades"]
        axes[0].plot(trades["pnl"].cumsum(), label=name, color=colors.get(name, "gray"), alpha=0.8)
        axes[1].plot(trades["inventory"], label=name, color=colors.get(name, "gray"), alpha=0.5)
        cum = trades["pnl"].cumsum()
        peak = cum.expanding().max()
        dd = cum - peak
        axes[2].fill_between(range(len(dd)), dd, 0, alpha=0.3, color=colors.get(name, "gray"))
    axes[0].set_ylabel("Cumulative PnL", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].set_title("Market Maker Performance", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Inventory", fontsize=11)
    axes[2].set_ylabel("Drawdown", fontsize=11)
    axes[2].set_xlabel("Time Step", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/market_making_pnl.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_causality(df, output_dir):
    data = df[["return_10s", "depth_imbalance"]].dropna()
    gc = grangercausalitytests(data, maxlag=5, verbose=False)
    lags = list(range(1, 6))
    pvals = [gc[lag][0]["ssr_chi2test"][1] for lag in lags]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(lags, pvals, color=["#27ae60" if p < 0.05 else "#e74c3c" for p in pvals], edgecolor="white")
    ax.axhline(0.05, color="black", linestyle="--", alpha=0.7, label="p = 0.05")
    ax.axhline(0.01, color="gray", linestyle=":", alpha=0.5, label="p = 0.01")
    for lag, p in zip(lags, pvals):
        ax.annotate(f"{p:.4f}", (lag, p), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9)
    ax.set_xlabel("Lag", fontsize=12)
    ax.set_ylabel("p-value", fontsize=12)
    ax.set_title("Granger Causality: Imbalance → Returns", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/causality_analysis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_velocity(df, output_dir):
    df = df.copy()
    df["imbalance_velocity"] = df["depth_imbalance"].diff(5) / 5
    df_clean = df.dropna(subset=["return_10s", "depth_imbalance", "imbalance_velocity"])
    y = df_clean["return_10s"]
    X1 = add_constant(df_clean["depth_imbalance"])
    X2 = add_constant(df_clean[["depth_imbalance", "imbalance_velocity"]])
    m1 = OLS(y, X1).fit()
    m2 = OLS(y, X2).fit()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Level Only", "+ Velocity"]
    r2s = [m1.rsquared, m2.rsquared]
    colors = ["#3498db", "#e74c3c"]
    axes[0].bar(labels, r2s, color=colors, edgecolor="white", width=0.5)
    for i, r in enumerate(r2s):
        axes[0].annotate(f"{r:.5f}", (i, r), textcoords="offset points", xytext=(0, 5), ha="center")
    axes[0].set_ylabel("R²", fontsize=12)
    axes[0].set_title("Incremental R²: Velocity Effect", fontsize=12, fontweight="bold")
    med = df_clean["depth_imbalance"].median()
    high = df_clean[df_clean["depth_imbalance"] > med]
    low = df_clean[df_clean["depth_imbalance"] <= med]
    corrs = [low["imbalance_velocity"].corr(low["return_10s"]), high["imbalance_velocity"].corr(high["return_10s"])]
    axes[1].bar(["Low Imbalance", "High Imbalance"], corrs, color=["#2ecc71", "#e74c3c"], edgecolor="white", width=0.5)
    for i, c in enumerate(corrs):
        axes[1].annotate(f"{c:.3f}", (i, c), textcoords="offset points", xytext=(0, 5), ha="center")
    axes[1].set_ylabel("Velocity-Return Correlation", fontsize=12)
    axes[1].set_title("State-Dependent Velocity Effect", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/velocity_analysis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_single(df, col, title, output_dir, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[col].dropna(), bins=80, kde=True, ax=ax, color="#3498db")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/{filename}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_time_series_simple(df, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    sample = df.iloc[:min(len(df), 2000)]
    axes[0].plot(sample["depth_imbalance"], color="#3498db", alpha=0.7, linewidth=0.5)
    axes[0].set_ylabel("Imbalance", fontsize=11)
    axes[0].set_title("Time Series of Liquidity Metrics", fontsize=14, fontweight="bold")
    axes[1].plot(sample["spread"], color="#e74c3c", alpha=0.7, linewidth=0.5)
    axes[1].set_ylabel("Spread", fontsize=11)
    axes[2].plot(sample["midprice"], color="#2c3e50", alpha=0.7, linewidth=0.5)
    axes[2].set_ylabel("Midprice", fontsize=11)
    axes[2].set_xlabel("Observation Index", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/liquidity_time_series.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df, output_dir):
    sample = df.iloc[::5].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(sample["depth_imbalance"], sample["return_10s"], gridsize=30, cmap="Blues", mincnt=1, alpha=0.8)
    cb = fig.colorbar(hb, ax=ax, label="Count")
    ax.set_xlabel("Depth Imbalance", fontsize=12)
    ax.set_ylabel("10s Return", fontsize=12)
    ax.set_title("Imbalance vs 10s Return Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/imbalance_return_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_decile(df, output_dir):
    df = df.copy()
    df["imb_decile"] = pd.qcut(df["depth_imbalance"].rank(method="first"), 10, labels=False)
    means = df.groupby("imb_decile")["return_10s"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if m < 0 else "#3498db" for m in means]
    ax.bar(range(10), means, color=colors, edgecolor="white")
    for i, m in enumerate(means):
        ax.annotate(f"{m:.6f}", (i, m), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8, rotation=45)
    ax.set_xlabel("Imbalance Decile", fontsize=12)
    ax.set_ylabel("Mean 10s Return", fontsize=12)
    ax.set_title("Mean 10s Return by Imbalance Decile", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/decile_analysis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── MAIN ─────────────────────────────────────────────────────────────
def run_all(input_file: str, output_dir: str = "plots", results_file: str = "results/rq_analysis.json"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_file) or ".", exist_ok=True)

    df = load_and_enrich(input_file)
    logger.info(f"Loaded {len(df)} rows from {input_file}")

    # Summary stats
    summary = {
        "n_rows": len(df),
        "imbalance_mean": round(float(df["depth_imbalance"].mean()), 4),
        "imbalance_std": round(float(df["depth_imbalance"].std()), 4),
        "imbalance_ar1": round(float(df["depth_imbalance"].corr(df["depth_imbalance"].shift(1))), 4),
        "spread_mean_bps": round(float(df["spread_bps"].mean()), 4),
        "return_1s_mean": round(float(df["return_1s"].mean()), 8),
        "return_1s_std": round(float(df["return_1s"].std()), 8),
    }

    # Run all RQs
    corrs = rq1_correlations(df)
    decay = rq2_decay(corrs)
    regime = rq3_regime(df)
    mm = rq4_market_making(df)
    gc = rq5_granger(df)
    vel = rq6_velocity(df)

    results = {
        "data_summary": summary,
        "rq1_correlations": corrs,
        "rq2_decay": decay,
        "rq3_regime": regime,
        "rq4_market_making": {name: {"stats": r["stats"]} for name, r in mm.items()},
        "rq5_granger_causality": gc,
        "rq6_velocity_analysis": vel,
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    # Generate plots
    logger.info("Generating plots...")
    plot_signal_decay(corrs, output_dir)
    plot_signal_by_regime(df, output_dir)
    plot_market_making(mm, output_dir)
    plot_causality(df, output_dir)
    plot_velocity(df, output_dir)
    plot_distribution_single(df, "depth_imbalance", "Distribution of Depth Imbalance", output_dir, "imbalance_distribution.png")
    plot_distribution_single(df, "spread", "Distribution of Spread", output_dir, "spread_distribution.png")
    plot_time_series_simple(df, output_dir)
    plot_heatmap(df, output_dir)
    plot_decile(df, output_dir)
    logger.info(f"All plots saved to {output_dir}/")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Analysis Complete: {len(df)} rows")
    print(f"{'='*60}")
    print(f"\nRQ1 — Correlations:")
    for k, v in corrs.items():
        print(f"  {k}: {v}")
    print(f"\nRQ2 — Decay:")
    for k, v in decay.items():
        print(f"  {k}: {v}")
    print(f"\nRQ3 — Regime (key):")
    for k in list(regime.keys())[:2]:
        print(f"  {k}: r={regime[k]['corr']}, n={regime[k]['n']}")
    print(f"\nRQ4 — Market Making:")
    for name, r in mm.items():
        s = r["stats"]
        print(f"  {name}: PnL={s['total_pnl']}, Sharpe={s['sharpe_ratio']}")
    print(f"\nRQ5 — Granger:")
    for k, v in gc.items():
        print(f"  {k}: p={v['ssr_p']}")
    print(f"\nRQ6 — Velocity:")
    print(f"  R² increase: {vel['r2_pct_change']}%")
    print(f"  Vel t-stat: {vel['vel_tstat']} (p={vel['vel_pval']})")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run all RQ1-RQ6 analysis and generate plots")
    parser.add_argument("--input", default="data/processed/kraken_combined.csv")
    parser.add_argument("--output-dir", default="plots")
    parser.add_argument("--results", default="results/rq_analysis.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_all(args.input, args.output_dir, args.results)


if __name__ == "__main__":
    main()
