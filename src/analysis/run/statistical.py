"""
Statistical analysis for limit order book data.
Phase 5 implementation for quantitative market microstructure research.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """Perform statistical analysis on order book imbalance and price discovery."""
    
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        
    def load_feature_data(self, file_path: str) -> pd.DataFrame:
        """Load feature-engineered data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} feature data points")
            return df
        except Exception as e:
            logger.error(f"Error loading feature data: {e}")
            return pd.DataFrame()
    
    def run_correlation_analysis(self, df: pd.DataFrame) -> dict:
        """Run baseline correlation analysis between imbalance and future returns."""
        if df.empty:
            return {}
        
        correlations = {}
        
        # Find imbalance column (could be depth_imbalance or order_flow_imbalance)
        imbalance_cols = ['depth_imbalance', 'order_flow_imbalance']
        imbalance_col = None
        for col in imbalance_cols:
            if col in df.columns:
                imbalance_col = col
                break
        
        if not imbalance_col:
            logger.warning("No imbalance column found")
            return correlations
        
        # Calculate correlations with future returns
        for horizon in [1, 5, 10, 30, 60, 300]:
            return_col = f'return_{horizon}s'
            if return_col in df.columns:
                corr = df[imbalance_col].corr(df[return_col])
                correlations[return_col] = corr
                logger.info(f"Correlation({imbalance_col}, {return_col}) = {corr:.4f}")
        
        return correlations
    
    def run_ols_regressions(self, df: pd.DataFrame) -> dict:
        """Run OLS regressions with HAC/Newey-West standard errors."""
        if df.empty:
            return {}
        
        results = {}
        
        # Ensure we have required columns
        required_cols = ['depth_imbalance', 'spread', 'midprice']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in data")
                return results
        
        # Create future return target variable (1-second return as example)
        df['future_return'] = df['return_1s'].fillna(0)
        
        # Prepare data
        X = df[['depth_imbalance', 'spread', 'volatility']].dropna()
        y = df.loc[X.index, 'future_return']
        
        if len(X) == 0:
            logger.warning("No valid data for regression analysis")
            return results
        
        # Add constant for intercept
        X = sm.add_constant(X)
        
        try:
            # Run OLS regression with HAC standard errors
            model = sm.OLS(y, X).fit()
            
            # Get HAC standard errors
            hac_cov = cov_hac(model.cov_params(), nlags=1)
            model_hac = type('HACModel', (object,), {
                'params': model.params,
                'bse': np.sqrt(np.diag(hac_cov)),
                'tvalues': model.params / np.sqrt(np.diag(hac_cov)),
                'pvalues': 2 * (1 - abs(model.tvalues) / model.tvalues.max()),
                'rsquared': model.rsquared
            })()
            
            results['model_summary'] = model_hac
            results['rsquared'] = model.rsquared
            
            logger.info(f"OLS regression completed. R-squared: {model.rsquared:.4f}")
            
        except Exception as e:
            logger.error(f"Error running OLS regression: {e}")
        
        return results
    
    def run_interaction_models(self, df: pd.DataFrame) -> dict:
        """Run interaction models to test liquidity's effect on signal strength."""
        if df.empty:
            return {}
        
        results = {}
        
        # Create interaction term: Imbalance × Spread
        if 'depth_imbalance' in df.columns and 'spread' in df.columns:
            df['imbalance_spread_interaction'] = df['depth_imbalance'] * df['spread']
            
            # Prepare data
            X = df[['depth_imbalance', 'spread', 'imbalance_spread_interaction']].dropna()
            y = df.loc[X.index, 'return_1s']
            
            if len(X) > 0:
                X = sm.add_constant(X)
                
                try:
                    model = sm.OLS(y, X).fit()
                    
                    # Get HAC standard errors
                    hac_cov = cov_hac(model.cov_params(), nlags=1)
                    model_hac = type('HACModel', (object,), {
                        'params': model.params,
                        'bse': np.sqrt(np.diag(hac_cov)),
                        'tvalues': model.params / np.sqrt(np.diag(hac_cov)),
                        'pvalues': 2 * (1 - abs(model.tvalues) / model.tvalues.max()),
                        'rsquared': model.rsquared
                    })()
                    
                    results['interaction_model'] = model_hac
                    results['interaction_rsquared'] = model.rsquared
                    
                    logger.info(f"Interaction model completed. R-squared: {model.rsquared:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error running interaction model: {e}")
        
        return results
    
    def run_statistical_analysis(self, input_file: str, output_file: str = None) -> dict:
        """Run complete statistical analysis pipeline."""
        logger.info("Starting statistical analysis pipeline")
        
        # Load feature data
        df = self.load_feature_data(input_file)
        if df.empty:
            return {}
        
        # Run analyses
        correlations = self.run_correlation_analysis(df)
        ols_results = self.run_ols_regressions(df)
        interaction_results = self.run_interaction_models(df)
        
        # Compile results
        all_results = {
            'correlations': correlations,
            'ols_regression': ols_results,
            'interaction_model': interaction_results,
            'sample_size': len(df)
        }
        
        # Save results
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"Statistical results saved to {output_file}")
        
        logger.info(f"Statistical analysis complete. Analyzed {len(df)} data points")
        return all_results

def main():
    """Main entry point for statistical analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Statistical Analysis Pipeline")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    
    args = parser.parse_args()
    
    analyzer = StatisticalAnalyzer()
    analyzer.run_statistical_analysis(input_file=args.input, output_file=args.output)

if __name__ == "__main__":
    main()