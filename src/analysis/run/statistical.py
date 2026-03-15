"""
Statistical analysis for limit order book data.
Phase 5 implementation for quantitative market microstructure research.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy import stats
from scipy.optimize import curve_fit
import logging
from typing import Optional, List, Dict

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
            hac_cov = cov_hac(model, nlags=1)
            model_hac = type('HACModel', (object,), {
                'params': model.params,
                'bse': np.sqrt(np.diag(hac_cov)),
                'tvalues': model.params / np.sqrt(np.diag(hac_cov)),
                'pvalues': 2 * stats.t.sf(abs(model.tvalues), df=model.df_resid),
                'rsquared': model.rsquared
            })()
            
            results['model_summary'] = model_hac
            results['rsquared'] = model.rsquared
            
            logger.info(f"OLS regression completed. R-squared: {model.rsquared:.4f}")
            
        except Exception as e:
            logger.error(f"Error running OLS regression: {e}")
        
        return results
    
    def _fit_hac_model(self, y: pd.Series, X: pd.DataFrame) -> Optional[Dict]:
        """Fit OLS with HAC standard errors and return structured results."""
        try:
            model = sm.OLS(y, X).fit()
            hac_cov = cov_hac(model, nlags=1)
            return {
                'params': model.params.to_dict(),
                'bse': {k: float(v) for k, v in zip(model.params.index, np.sqrt(np.diag(hac_cov)))},
                'tvalues': (model.params / np.sqrt(np.diag(hac_cov))).to_dict(),
                'pvalues': {k: float(2 * stats.t.sf(abs(t), df=model.df_resid))
                           for k, t in zip(model.params.index, model.params / np.sqrt(np.diag(hac_cov)))},
                'rsquared': float(model.rsquared),
                'nobs': int(model.nobs)
            }
        except Exception as e:
            logger.warning(f"HAC model fit failed: {e}")
            return None

    def run_interaction_models(self, df: pd.DataFrame, horizon: str = 'return_1s') -> dict:
        """Test whether market conditions moderate the imbalance-return relationship.
        
        For each moderator (spread, volatility, volume), fits:
            return = b0 + b1*imbalance + b2*moderator + b3*(imbalance × moderator)
        
        A significant b3 means the signal strength changes with that condition.
        """
        if df.empty or 'depth_imbalance' not in df.columns or horizon not in df.columns:
            return {}
        
        results = {}
        imbalance = 'depth_imbalance'
        moderators = ['spread', 'volatility', 'volume']
        available = [m for m in moderators if m in df.columns and df[m].nunique() > 1]
        
        for mod in available:
            df_mod = df[[imbalance, mod, horizon]].dropna().copy()
            if len(df_mod) < 100:
                continue
            df_mod['interaction'] = df_mod[imbalance] * df_mod[mod]
            X = sm.add_constant(df_mod[[imbalance, mod, 'interaction']])
            y = df_mod[horizon]
            fit = self._fit_hac_model(y, X)
            if fit is not None:
                results[mod] = fit
                logger.info(
                    f"Interaction({imbalance} × {mod}): β_interaction={fit['params'].get('interaction', 0):.6f}, "
                    f"p={fit['pvalues'].get('interaction', 1):.4f}, R²={fit['rsquared']:.4f}"
                )
        
        return results

    def run_regime_analysis(self, df: pd.DataFrame, horizon: str = 'return_1s') -> dict:
        """Split data into quintiles by spread, volatility, volume and compute
        the imbalance-return correlation within each regime.
        
        Answers: does the signal strengthen or weaken in certain market conditions?
        """
        if df.empty or 'depth_imbalance' not in df.columns or horizon not in df.columns:
            return {}
        
        results = {}
        factors = ['spread', 'volatility', 'volume']
        available = [f for f in factors if f in df.columns and df[f].nunique() > 1]
        
        for factor in available:
            valid = df[['depth_imbalance', horizon, factor]].dropna()
            if len(valid) < 100:
                continue
            valid['quintile'] = pd.qcut(valid[factor], 5, labels=[1, 2, 3, 4, 5])
            quintile_results = []
            for q in range(1, 6):
                subset = valid[valid['quintile'] == q]
                if len(subset) < 10:
                    continue
                corr = subset['depth_imbalance'].corr(subset[horizon])
                q_mean = subset[factor].mean()
                quintile_results.append({
                    'quintile': q,
                    f'mean_{factor}': float(q_mean),
                    'correlation': float(corr),
                    'n': int(len(subset))
                })
            results[factor] = quintile_results
            corrs = [q['correlation'] for q in quintile_results]
            logger.info(f"Regime({factor}): correlations by quintile = {[f'{c:.4f}' for c in corrs]}")
        
        return results
    
    def fit_exponential_decay(self, horizons: np.ndarray, correlations: np.ndarray) -> Optional[Dict]:
        """Fit exponential decay model: corr = a * exp(-b * h) + c."""
        if len(horizons) < 4 or len(correlations) < 4:
            return None
        sort_idx = np.argsort(horizons)
        h = np.asarray(horizons)[sort_idx]
        c = np.asarray(correlations)[sort_idx]
        try:
            a_guess = c[0] - c[-1]
            b_guess = 0.01
            c_guess = c[-1]
            popt, pcov = curve_fit(
                lambda h, a, b, c: a * np.exp(-b * h) + c,
                h, c,
                p0=[a_guess, b_guess, c_guess],
                bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf])
            )
            residuals = c - (popt[0] * np.exp(-popt[1] * h) + popt[2])
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((c - np.mean(c)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            half_life = np.log(2) / popt[1] if popt[1] > 0 else np.inf
            return {
                'model': 'exponential_decay',
                'params': {'a': popt[0], 'b': popt[1], 'c': popt[2]},
                'param_errors': {'a': np.sqrt(pcov[0, 0]), 'b': np.sqrt(pcov[1, 1]), 'c': np.sqrt(pcov[2, 2])},
                'r_squared': r_squared,
                'half_life': half_life
            }
        except Exception as e:
            logger.warning(f"Exponential decay fit failed: {e}")
            return None
    
    def fit_power_law(self, horizons: np.ndarray, correlations: np.ndarray) -> Optional[Dict]:
        """Fit power law decay: corr = a * h^(-b) + c."""
        if len(horizons) < 4 or len(correlations) < 4:
            return None
        sort_idx = np.argsort(horizons)
        h = np.asarray(horizons)[sort_idx]
        c = np.asarray(correlations)[sort_idx]
        try:
            a_guess = c[0] - c[-1]
            b_guess = 0.1
            c_guess = c[-1]
            popt, pcov = curve_fit(
                lambda h, a, b, c: a * h ** (-b) + c,
                h, c,
                p0=[a_guess, b_guess, c_guess],
                maxfev=10000
            )
            residuals = c - (popt[0] * h ** (-popt[1]) + popt[2])
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((c - np.mean(c)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            return {
                'model': 'power_law',
                'params': {'a': popt[0], 'b': popt[1], 'c': popt[2]},
                'param_errors': {'a': np.sqrt(pcov[0, 0]), 'b': np.sqrt(pcov[1, 1]), 'c': np.sqrt(pcov[2, 2])},
                'r_squared': r_squared
            }
        except Exception as e:
            logger.warning(f"Power law fit failed: {e}")
            return None
    
    def compute_half_life(self, horizons: np.ndarray, correlations: np.ndarray) -> Optional[float]:
        """Compute signal half-life via linear interpolation."""
        if len(horizons) < 2 or len(correlations) < 2:
            return None
        sort_idx = np.argsort(horizons)
        h = np.asarray(horizons)[sort_idx]
        c = np.asarray(correlations)[sort_idx]
        half_max = np.max(np.abs(c)) / 2
        for i in range(len(c) - 1):
            if c[i] >= half_max >= c[i + 1] or c[i] <= half_max <= c[i + 1]:
                frac = (half_max - c[i]) / (c[i + 1] - c[i]) if c[i + 1] != c[i] else 0
                return float(h[i] + frac * (h[i + 1] - h[i]))
        return None
    
    def run_signal_decay_analysis(self, df: pd.DataFrame) -> Dict:
        """Run signal decay analysis fitting exponential and power law models."""
        if df.empty:
            return {}
        return_cols = [col for col in df.columns if col.startswith('return_')]
        if not return_cols:
            return {}
        imbalance_cols = ['depth_imbalance', 'order_flow_imbalance']
        imbalance_col = next((c for c in imbalance_cols if c in df.columns), None)
        if imbalance_col is None:
            return {}
        horizons = np.array([int(col.split('_')[1].rstrip('s')) for col in return_cols])
        correlations = np.array([df[imbalance_col].corr(df[col]) for col in return_cols])
        sort_idx = np.argsort(horizons)
        horizons = horizons[sort_idx]
        correlations = correlations[sort_idx]
        half_life = self.compute_half_life(horizons, correlations)
        exp_fit = self.fit_exponential_decay(horizons, correlations)
        pow_fit = self.fit_power_law(horizons, correlations)
        preferred_model = 'exponential'
        if pow_fit is not None and exp_fit is not None:
            preferred_model = 'exponential' if exp_fit['r_squared'] >= pow_fit['r_squared'] else 'power_law'
        elif pow_fit is not None:
            preferred_model = 'power_law'
        result = {
            'horizons': horizons.tolist(),
            'correlations': correlations.tolist(),
            'half_life_seconds': half_life,
            'preferred_model': preferred_model
        }
        if exp_fit is not None:
            result['exponential_fit'] = {
                'a': exp_fit['params']['a'],
                'b': exp_fit['params']['b'],
                'c': exp_fit['params']['c'],
                'half_life_seconds': exp_fit['half_life'],
                'r_squared': exp_fit['r_squared']
            }
        if pow_fit is not None:
            result['power_law_fit'] = {
                'a': pow_fit['params']['a'],
                'b': pow_fit['params']['b'],
                'c': pow_fit['params']['c'],
                'r_squared': pow_fit['r_squared']
            }
        return result
    
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
        regime_results = self.run_regime_analysis(df)
        signal_decay = self.run_signal_decay_analysis(df)
        
        # Compile results
        all_results = {
            'correlations': correlations,
            'ols_regression': ols_results,
            'interaction_model': interaction_results,
            'regime_analysis': regime_results,
            'signal_decay': signal_decay,
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