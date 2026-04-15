"""
Exploratory data analysis visualizations.
Phase 4 implementation for quantitative market microstructure research.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """Generate EDA visualizations for limit order book analysis."""
    
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = output_dir
        plt.style.use('seaborn-v0_8')
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data for visualization."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} data points for visualization")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def plot_distribution(self, df: pd.DataFrame, column: str, title: str, bins: int = 50):
        """Create histogram of distribution for a given column."""
        if df.empty or column not in df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[column].dropna(), bins=bins, kde=True, ax=ax)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(column.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        
        return fig
    
    def plot_time_series(self, df: pd.DataFrame, value_cols: list, timestamp_col: str = 'timestamp', title: str = "Time Series"):
        """Create time series plot for specified columns."""
        if df.empty:
            return None
        
        # Convert timestamp to datetime if needed
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df.set_index(timestamp_col, inplace=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for col in value_cols:
            if col in df.columns:
                ax.plot(df.index, df[col], label=col, linewidth=1)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_heatmap(self, df: pd.DataFrame, x_col: str, y_col: str, title: str = "Heatmap"):
        """Create heatmap of relationship between two columns."""
        if df.empty:
            return None
        
        # Create a pivot table for heatmap
        pivot_df = df.pivot_table(values=y_col, columns=x_col, aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap='RdBu_r', center=0, ax=ax)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        return fig
    
    def plot_decile_analysis(self, df: pd.DataFrame, imbalance_col: str = 'depth_imbalance', return_col: str = 'return_1s'):
        """Create decile analysis plot showing imbalance vs future returns."""
        if df.empty:
            return None
        
        # Create deciles based on imbalance
        df_sorted = df.sort_values(by=imbalance_col)
        df_sorted['imbalance_decile'] = pd.qcut(df_sorted[imbalance_col], 10)
        
        # Calculate average returns per decile
        decile_stats = df_sorted.groupby('imbalance_decile').agg(
            mean_return=(return_col, 'mean'),
            std_return=(return_col, 'std'),
            mean_imbalance=(imbalance_col, 'mean'),
            count=(imbalance_col, 'count')
        ).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Distribution of imbalance by decile
        ax1.bar(range(1, 11), decile_stats['mean_imbalance'], color='lightblue')
        ax1.set_xlabel('Imbalance Decile')
        ax1.set_ylabel('Mean Imbalance')
        ax1.set_title('Mean Imbalance by Decile', fontweight='bold')
        
        # Plot 2: Average returns by decile
        colors = ['red' if x < 0 else 'green' for x in decile_stats['mean_return']]
        bars = ax2.bar(range(1, 11), decile_stats['mean_return'], color=colors)
        ax2.set_xlabel('Imbalance Decile')
        ax2.set_ylabel('Average Future Return')
        ax2.set_title('Average Future Returns by Imbalance Decile', fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add error bars
        ax2.errorbar(range(1, 11), decile_stats['mean_return'], 
                    yerr=decile_stats['std_return'], fmt='none', color='black', capsize=5)
        
        plt.tight_layout()
        return fig
    
    def plot_signal_decay_curve(
        self,
        horizons: list,
        correlations: list,
        exp_fit: Optional[dict] = None,
        power_fit: Optional[dict] = None,
    ):
        """Create dual-panel signal decay curve visualization.

        Left panel: horizons vs correlations with exponential fit (semilog x).
        Right panel: horizons vs absolute correlations with power-law fit (log-log).
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left panel: exponential fit on semilog x
        ax1.scatter(horizons, correlations, color='steelblue', label='Observed')
        if exp_fit is not None:
            x_smooth = np.linspace(min(horizons), max(horizons), 200)
            y_fit = exp_fit['a'] * np.exp(-exp_fit['b'] * x_smooth)
            ax1.plot(x_smooth, y_fit, 'r-', label=f"Exp fit (R²={exp_fit.get('r_squared', 0):.3f})")
        ax1.set_xscale('log')
        ax1.set_xlabel('Forecast Horizon (s, log scale)')
        ax1.set_ylabel('Correlation')
        ax1.set_title('Signal Decay Curve (Exponential Fit)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right panel: power-law fit on log-log
        abs_corr = [abs(c) for c in correlations]
        ax2.scatter(horizons, abs_corr, color='steelblue', label='Observed')
        if power_fit is not None:
            x_smooth = np.linspace(min(horizons), max(horizons), 200)
            y_fit = power_fit['a'] * x_smooth ** (-power_fit['b'])
            ax2.plot(x_smooth, y_fit, 'r-', label=f"Power fit (R²={power_fit.get('r_squared', 0):.3f})")
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Forecast Horizon (s, log scale)')
        ax2.set_ylabel('|Correlation| (log scale)')
        ax2.set_title('Absolute Decay (Power-Law Fit)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_interaction_analysis(self, df: pd.DataFrame, results: dict):
        """Three-panel plot showing how imbalance-return correlation varies across
        spread, volatility, and volume quintiles (RQ3: When does imbalance matter most?)."""
        if df.empty or 'depth_imbalance' not in df.columns:
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        factors = ['spread', 'volatility', 'volume']
        titles = ['Spread (Liquidity)', 'Volatility', 'Volume (Activity)']
        ylabels = ['Spread Size', 'Volatility', 'Volume']
        
        for idx, (factor, title, ylabel) in enumerate(zip(factors, titles, ylabels)):
            ax = axes[idx]
            regime = results.get('regime_analysis', {}).get(factor, [])
            if not regime:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            q_labels = [str(r['quintile']) for r in regime]
            corrs = [r['correlation'] for r in regime]
            colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in corrs]
            
            bars = ax.bar(q_labels, corrs, color=colors, edgecolor='gray', linewidth=0.5)
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_xlabel(f'{ylabel} Quintile\n(Low → High)')
            ax.set_ylabel('Imbalance-Return Correlation')
            ax.set_title(f'Signal Strength by {title}', fontweight='bold', fontsize=12)
            
            for bar, corr in zip(bars, corrs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{corr:.4f}', ha='center', va='bottom' if corr > 0 else 'top',
                        fontsize=8)
            
            ax.tick_params(axis='x', labelsize=9)
            ax.grid(axis='y', alpha=0.3)
        
        fig.suptitle('RQ3: When Does Order Book Imbalance Matter Most?',
                     fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        return fig

    def plot_market_making(self, sim_results: dict):
        """Plot market making results: cumulative PnL, inventory, and drawdown."""
        if not sim_results:
            return None

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        colors = {'Basic': '#e74c3c', 'Adaptive': '#2ecc71'}

        for name, result in sim_results.items():
            trades = result['trades']
            c = colors.get(name, '#3498db')
            label = f"{name} MM"
            axes[0].plot(trades['step'], trades['cumulative_pnl'], color=c, lw=1.5, label=label)
            axes[1].plot(trades['step'], trades['inventory'], color=c, lw=1, alpha=0.8, label=label)
            cum_pnl = trades['pnl'].fillna(0).cumsum()
            peak = cum_pnl.expanding().max()
            dd = cum_pnl - peak
            axes[2].fill_between(trades['step'], dd, 0, color=c, alpha=0.3, label=label)

        axes[0].set_ylabel('Cumulative PnL')
        axes[0].set_title('Market Making Simulation: PnL Comparison', fontweight='bold')
        axes[0].legend()
        axes[0].axhline(y=0, color='gray', ls='--', alpha=0.5)
        axes[0].grid(alpha=0.3)

        axes[1].set_ylabel('Inventory Position')
        axes[1].set_title('Inventory Over Time', fontweight='bold')
        axes[1].legend()
        axes[1].axhline(y=0, color='gray', ls='--', alpha=0.5)
        axes[1].grid(alpha=0.3)

        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Drawdown')
        axes[2].set_title('Drawdown', fontweight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_causality_analysis(self, results: dict):
        """RQ5: Two-panel plot of lead-lag cross-correlation and variance decomposition."""
        if not results:
            return None

        ca = results.get('causality_analysis', {})
        ll = ca.get('lead_lag', {})
        vd = ca.get('variance_decomposition', {})

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left panel: Lead-lag cross-correlation
        ax = axes[0]
        if ll:
            fwd = ll.get('imbalance_leads_returns', {})
            bwd = ll.get('returns_lead_imbalance', {})
            lags = list(range(1, len(fwd) + 1))
            fwd_vals = [fwd[k] for k in fwd]
            bwd_vals = [bwd[k] for k in bwd]
            ax.plot(lags, fwd_vals, 'o-', color='#2ecc71', lw=2, label='Imbalance → Future Return')
            ax.plot(lags, bwd_vals, 's-', color='#e74c3c', lw=2, label='Return → Future Imbalance')
            ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
            ax.set_xlabel('Lag (steps)')
            ax.set_ylabel('Cross-Correlation')
            ax.set_title('Lead-Lag: Does Imbalance Lead or Follow Price?', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

            summary = ca.get('lead_lag_summary', {})
            direction = summary.get('direction', 'unknown')
            ratio = summary.get('asymmetry_ratio', 0)
            ax.text(0.5, -0.25,
                    f'Direction: {direction}  |  Asymmetry ratio: {ratio:.2f}x',
                    ha='center', va='top', transform=ax.transAxes,
                    fontsize=11, style='italic')

        # Right panel: Variance decomposition
        ax = axes[1]
        if vd:
            labels = ['Base\n(Spread+Vol)', 'Full\n(+Imbalance)']
            values = [vd.get('r2_base', 0) * 100, vd.get('r2_full', 0) * 100]
            colors_bar = ['#3498db', '#2ecc71']
            bars = ax.bar(labels, values, color=colors_bar, width=0.5, edgecolor='gray')
            inc = vd.get('r2_increment', 0) * 100
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                        f'{val:.4f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
            ax.set_ylabel('R² (%)')
            ax.set_title(f'Variance Decomposition\nImbalance adds {inc:.4f}pp', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

        fig.suptitle('RQ5: Why Does the Imbalance Signal Exist?',
                     fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        return fig

    def plot_velocity_analysis(self, results: dict):
        """RQ6: Bar chart comparing R² of level-only vs level+velocity+acceleration models."""
        va = results.get('velocity_analysis', {})
        if not va:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left panel: correlation comparison
        ax = axes[0]
        labels = ['Level', 'Velocity', 'Acceleration']
        corrs = [
            va.get('level_correlation', 0),
            va.get('velocity_correlation', 0),
            va.get('acceleration_correlation', 0),
        ]
        colors_c = ['#3498db', '#2ecc71', '#e67e22']
        bars = ax.bar(labels, corrs, color=colors_c, width=0.5, edgecolor='gray')
        ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
        ax.set_ylabel('Correlation with 1s Return')
        ax.set_title('Imbalance Level vs Dynamics\nPredictive Power', fontweight='bold')
        for bar, val in zip(bars, corrs):
            y_pos = bar.get_height() + max(corrs) * 0.15 if val >= 0 else bar.get_height() - max(corrs) * 0.15
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top',
                    fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Right panel: incremental R²
        ax = axes[1]
        inc = va.get('incremental_r2', {})
        if inc:
            labels_r = list(inc.keys())
            vals_r = list(inc.values())
            colors_r = ['#3498db', '#2ecc71', '#e67e22'][:len(labels_r)]
            bars = ax.bar(labels_r, vals_r, color=colors_r, width=0.5, edgecolor='gray')
            for bar, val in zip(bars, vals_r):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals_r)*0.02,
                        f'{val:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.set_ylabel('R²')
            g = va.get('r2_gain', 0)
            p = va.get('r2_gain_pct', 0)
            ax.set_title(f'Incremental Predictive Power (Gain: +{g:.6f}, {p:.1f}%)', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

        # Combined signal annotation
        cs = va.get('combined_signal', {})
        if cs:
            spread = cs.get('long_short_spread', 0)
            text = (
                f'RQ6: Imbalance Dynamics - Velocity & Acceleration | '
                f'Long-short spread={spread:.8f} | '
                f'Long n={cs.get("n_long",0)} mean={cs.get("mean_return_long",0):.8f} | '
                f'Short n={cs.get("n_short",0)} mean={cs.get("mean_return_short",0):.8f}'
            )
            fig.suptitle(text, fontweight='bold', fontsize=12, y=1.02)
        else:
            fig.suptitle('RQ6: Imbalance Dynamics — Velocity & Acceleration',
                         fontweight='bold', fontsize=14, y=1.02)

        plt.tight_layout()
        return fig

    def generate_all_visualizations(self, input_file: str, output_dir: str = "plots", results: dict = None):
        """Generate all EDA visualizations."""
        logger.info("Starting visualization generation pipeline")
        
        # Load data
        df = self.load_data(input_file)
        if df.empty:
            return
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        visualizations = {}
        
        # 0. RQ3: Interaction / regime analysis plot
        if results is not None:
            fig = self.plot_interaction_analysis(df, results)
            if fig:
                fig.savefig(f"{output_dir}/signal_by_regime.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
                visualizations['signal_by_regime'] = f"{output_dir}/signal_by_regime.png"
                logger.info("Saved signal-by-regime interaction plot")
        
        # 1. Distribution of imbalance
        if 'depth_imbalance' in df.columns:
            fig = self.plot_distribution(
                df, 'depth_imbalance', 
                "Distribution of Depth Imbalance", 
                bins=50
            )
            if fig:
                fig.savefig(f"{output_dir}/imbalance_distribution.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
                visualizations['imbalance_distribution'] = f"{output_dir}/imbalance_distribution.png"
                logger.info("Saved imbalance distribution plot")
        
        # 2. Distribution of spread
        if 'spread' in df.columns:
            fig = self.plot_distribution(
                df, 'spread', 
                "Distribution of Spread", 
                bins=50
            )
            if fig:
                fig.savefig(f"{output_dir}/spread_distribution.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
                visualizations['spread_distribution'] = f"{output_dir}/spread_distribution.png"
                logger.info("Saved spread distribution plot")
        
        # 3. Time series of liquidity
        time_series_cols = ['depth_imbalance', 'spread', 'midprice']
        fig = self.plot_time_series(
            df, time_series_cols, 
            "Time Series of Liquidity Metrics",
            "Liquidity Metrics Time Series"
        )
        if fig:
            fig.savefig(f"{output_dir}/liquidity_time_series.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            visualizations['liquidity_time_series'] = f"{output_dir}/liquidity_time_series.png"
            logger.info("Saved liquidity time series plot")
        
        # 4. Heatmap of imbalance vs returns
        if 'depth_imbalance' in df.columns:
            # Sample data for heatmap (take every 100th row)
            sample_df = df.iloc[::100].copy()
            fig = self.plot_heatmap(
                sample_df, 'depth_imbalance', 'return_1s',
                "Depth Imbalance vs 1-Second Future Return"
            )
            if fig:
                fig.savefig(f"{output_dir}/imbalance_return_heatmap.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
                visualizations['imbalance_return_heatmap'] = f"{output_dir}/imbalance_return_heatmap.png"
                logger.info("Saved imbalance-return heatmap")
        
        # 5. Decile analysis
        fig = self.plot_decile_analysis(df)
        if fig:
            fig.savefig(f"{output_dir}/decile_analysis.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            visualizations['decile_analysis'] = f"{output_dir}/decile_analysis.png"
            logger.info("Saved decile analysis plot")
        
        # 6. Market making simulation (RQ4)
        if results is not None and 'market_making' in results:
            fig = self.plot_market_making(results['market_making'])
            if fig:
                fig.savefig(f"{output_dir}/market_making_pnl.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
                visualizations['market_making'] = f"{output_dir}/market_making_pnl.png"
                logger.info("Saved market making PnL plot")
        
        # 7. Causality analysis (RQ5)
        if results is not None and 'causality_analysis' in results:
            fig = self.plot_causality_analysis(results)
            if fig:
                fig.savefig(f"{output_dir}/causality_analysis.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
                visualizations['causality_analysis'] = f"{output_dir}/causality_analysis.png"
                logger.info("Saved causality analysis plot")
        
        # 8. Velocity / acceleration analysis (RQ6)
        if results is not None and 'velocity_analysis' in results:
            fig = self.plot_velocity_analysis(results)
            if fig:
                fig.savefig(f"{output_dir}/velocity_analysis.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
                visualizations['velocity_analysis'] = f"{output_dir}/velocity_analysis.png"
                logger.info("Saved velocity analysis plot")
        
        logger.info(f"Generated {len(visualizations)} visualizations")
        return visualizations

def generate_all(output_dir: str = "plots", feature_file: str = "data/features/btcusdt_features.csv"):
    """Full pipeline: run analysis + MM simulation then generate all visualizations."""
    from .statistical import StatisticalAnalyzer
    from .market_making import MarketMakingSimulator, BasicMarketMaker, AdaptiveMarketMaker

    sa = StatisticalAnalyzer()
    results = sa.run_statistical_analysis(feature_file)

    df = pd.read_csv(feature_file)
    sim = MarketMakingSimulator(df)
    mm_strategies = [BasicMarketMaker(), AdaptiveMarketMaker(skew_strength=0.5, vol_mult=2.0)]
    mm_results = sim.run(mm_strategies)
    results['market_making'] = mm_results

    vg = VisualizationGenerator(output_dir=output_dir)
    vg.generate_all_visualizations(input_file=feature_file, output_dir=output_dir, results=results)
    return results

def main():
    """Main entry point for visualization generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualization Generation Pipeline")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", default="plots", help="Output directory for plots (default: plots)")
    parser.add_argument("--results", default=None, help="Optional results JSON for interaction plots")
    
    args = parser.parse_args()
    
    visualizer = VisualizationGenerator(output_dir=args.output)
    results = None
    if args.results:
        import json
        with open(args.results) as f:
            results = json.load(f)
    visualizer.generate_all_visualizations(input_file=args.input, output_dir=args.output, results=results)

if __name__ == "__main__":
    main()