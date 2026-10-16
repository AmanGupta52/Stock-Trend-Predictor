import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PortfolioAnalyzer:
    """
    Portfolio analysis and risk management class
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate  # Annual risk-free rate
        self.portfolio_data = None
        self.weights = None
        
    def create_portfolio(self, stock_data: Dict[str, pd.DataFrame], 
                        weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Create portfolio from multiple stocks
        """
        try:
            # Align all dataframes by date
            common_dates = None
            returns_data = {}
            
            for ticker, data in stock_data.items():
                if 'Close' in data.columns:
                    returns = data['Close'].pct_change().dropna()
                    returns_data[ticker] = returns
                    
                    if common_dates is None:
                        common_dates = returns.index
                    else:
                        common_dates = common_dates.intersection(returns.index)
            
            if not common_dates or len(common_dates) == 0:
                raise ValueError("No common dates found in stock data")
            
            # Create aligned returns dataframe
            aligned_returns = pd.DataFrame()
            for ticker, returns in returns_data.items():
                aligned_returns[ticker] = returns.loc[common_dates]
            
            # Calculate weights if not provided
            if weights is None:
                # Equal weighting
                n_stocks = len(aligned_returns.columns)
                weights = {ticker: 1/n_stocks for ticker in aligned_returns.columns}
            
            self.weights = weights
            
            # Calculate portfolio returns
            weighted_returns = aligned_returns * pd.Series(weights)
            portfolio_returns = weighted_returns.sum(axis=1)
            
            # Create portfolio dataframe
            portfolio_df = pd.DataFrame({
                'Portfolio_Return': portfolio_returns,
                'Cumulative_Return': (1 + portfolio_returns).cumprod() - 1
            })
            
            # Add individual stock returns
            for ticker in aligned_returns.columns:
                portfolio_df[f'{ticker}_Return'] = aligned_returns[ticker]
                portfolio_df[f'{ticker}_Cumulative'] = (1 + aligned_returns[ticker]).cumprod() - 1
            
            self.portfolio_data = portfolio_df
            logger.info(f"Portfolio created with {len(aligned_returns.columns)} stocks")
            
            return portfolio_df
            
        except Exception as e:
            logger.error(f"Error creating portfolio: {str(e)}")
            raise
    
    def calculate_portfolio_metrics(self, portfolio_df: pd.DataFrame) -> Dict:
        """Calculate portfolio performance metrics"""
        try:
            returns = portfolio_df['Portfolio_Return'].dropna()
            
            # Basic metrics
            total_return = portfolio_df['Cumulative_Return'].iloc[-1] if len(portfolio_df) > 0 else 0
            annual_return = self._calculate_annual_return(returns)
            volatility = returns.std() * np.sqrt(252)  # Annualized
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(portfolio_df['Cumulative_Return'])
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(annual_return, max_drawdown)
            
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'var_95': self._calculate_var(returns, 0.95),
                'cvar_95': self._calculate_cvar(returns, 0.95),
                'win_rate': (returns > 0).mean(),
                'profit_factor': self._calculate_profit_factor(returns)
            }
            
            logger.info("Portfolio metrics calculated")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {}
    
    def _calculate_annual_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        if len(returns) == 0:
            return 0
        cumulative_return = (1 + returns).prod() - 1
        trading_days = len(returns)
        if trading_days == 0:
            return 0
        return (1 + cumulative_return) ** (252 / trading_days) - 1
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        excess_returns = returns - (self.risk_free_rate / 252)
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf')
        downside_std = negative_returns.std()
        if downside_std == 0:
            return float('inf')
        excess_returns = returns.mean() - (self.risk_free_rate / 252)
        return excess_returns / downside_std * np.sqrt(252)
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(cumulative_returns) == 0:
            return 0
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / (1 + running_max)
        return drawdown.min()
    
    def _calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0:
            return float('inf')
        return annual_return / abs(max_drawdown)
    
    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk"""
        if len(returns) == 0:
            return 0
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor (gross profits / gross losses)"""
        if len(returns) == 0:
            return 0
        gross_profits = returns[returns > 0].sum()
        gross_losses = abs(returns[returns < 0].sum())
        if gross_losses == 0:
            return float('inf')
        return gross_profits / gross_losses
    
    def calculate_correlation_matrix(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation matrix for stocks in portfolio"""
        try:
            returns_data = {}
            
            for ticker, data in stock_data.items():
                if 'Close' in data.columns:
                    returns = data['Close'].pct_change().dropna()
                    returns_data[ticker] = returns
            
            # Create returns dataframe
            returns_df = pd.DataFrame(returns_data)
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            return pd.DataFrame()
    
    def optimize_portfolio_weights(self, stock_data: Dict[str, pd.DataFrame], 
                                 method: str = 'sharpe') -> Dict[str, float]:
        """Optimize portfolio weights using different methods"""
        try:
            # This is a simplified version - in practice, you'd use scipy.optimize
            returns_data = {}
            
            for ticker, data in stock_data.items():
                if 'Close' in data.columns:
                    returns = data['Close'].pct_change().dropna()
                    returns_data[ticker] = returns
            
            returns_df = pd.DataFrame(returns_data)
            
            if method == 'equal':
                # Equal weighting
                n_stocks = len(returns_df.columns)
                return {ticker: 1/n_stocks for ticker in returns_df.columns}
            elif method == 'sharpe':
                # Simple Sharpe-based weighting (simplified)
                sharpe_ratios = {}
                for ticker in returns_df.columns:
                    returns = returns_df[ticker].dropna()
                    if len(returns) > 0 and returns.std() > 0:
                        sharpe = returns.mean() / returns.std()
                        sharpe_ratios[ticker] = max(sharpe, 0)  # Ensure non-negative
                
                total_sharpe = sum(sharpe_ratios.values())
                if total_sharpe > 0:
                    return {ticker: sharpe/total_sharpe for ticker, sharpe in sharpe_ratios.items()}
                else:
                    # Fallback to equal weighting
                    n_stocks = len(returns_df.columns)
                    return {ticker: 1/n_stocks for ticker in returns_df.columns}
            else:
                raise ValueError(f"Unknown optimization method: {method}")
                
        except Exception as e:
            logger.error(f"Error optimizing portfolio weights: {str(e)}")
            # Fallback to equal weighting
            n_stocks = len(stock_data)
            return {ticker: 1/n_stocks for ticker in stock_data.keys()}
    
    def generate_portfolio_report(self, portfolio_df: pd.DataFrame, 
                                metrics: Dict) -> str:
        """Generate a text report of portfolio performance"""
        report = []
        report.append("=" * 50)
        report.append("PORTFOLIO PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Period: {portfolio_df.index.min().strftime('%Y-%m-%d')} to {portfolio_df.index.max().strftime('%Y-%m-%d')}")
        report.append(f"Total Trading Days: {len(portfolio_df)}")
        report.append("")
        report.append("PERFORMANCE METRICS:")
        report.append("-" * 30)
        report.append(f"Total Return: {metrics.get('total_return', 0):.2%}")
        report.append(f"Annual Return: {metrics.get('annual_return', 0):.2%}")
        report.append(f"Volatility: {metrics.get('volatility', 0):.2%}")
        report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        report.append(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        report.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
        report.append(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
        report.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        report.append(f"VaR (95%): {metrics.get('var_95', 0):.2%}")
        report.append(f"CVaR (95%): {metrics.get('cvar_95', 0):.2%}")
        
        if self.weights:
            report.append("")
            report.append("PORTFOLIO WEIGHTS:")
            report.append("-" * 30)
            for ticker, weight in self.weights.items():
                report.append(f"{ticker}: {weight:.2%}")
        
        return "\n".join(report)