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
        
        logger.info(f"Generated {len(visualizations)} visualizations")
        return visualizations

def generate_all(output_dir: str = "plots", feature_file: str = "data/features/btcusdt_features.csv"):
    """Full pipeline: run analysis then generate all visualizations."""
    from .statistical import StatisticalAnalyzer
    sa = StatisticalAnalyzer()
    results = sa.run_statistical_analysis(feature_file)
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