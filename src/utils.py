import yaml
import json
import logging
import logging.config
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import os
import sys

class ConfigLoader:
    """
    Configuration loader for YAML and JSON files
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.configs = {}
    
    def load_yaml_config(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            filepath = os.path.join(self.config_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            self.configs[filename] = config
            logging.info(f"Loaded YAML config: {filename}")
            return config
        except Exception as e:
            logging.error(f"Error loading YAML config {filename}: {str(e)}")
            return {}
    
    def load_json_config(self, filename: str) -> Dict[str, Any]:
        """Load JSON configuration file"""
        try:
            filepath = os.path.join(self.config_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                config = json.load(file)
            self.configs[filename] = config
            logging.info(f"Loaded JSON config: {filename}")
            return config
        except Exception as e:
            logging.error(f"Error loading JSON config {filename}: {str(e)}")
            return {}
    
    def get_config(self, filename: str) -> Dict[str, Any]:
        """Get loaded configuration"""
        return self.configs.get(filename, {})
    
    def save_config(self, config: Dict[str, Any], filename: str):
        """Save configuration to file"""
        try:
            filepath = os.path.join(self.config_dir, filename)
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                with open(filepath, 'w', encoding='utf-8') as file:
                    yaml.dump(config, file, default_flow_style=False)
            elif filename.endswith('.json'):
                with open(filepath, 'w', encoding='utf-8') as file:
                    json.dump(config, file, indent=4)
            logging.info(f"Saved config: {filename}")
        except Exception as e:
            logging.error(f"Error saving config {filename}: {str(e)}")

class LoggerSetup:
    """
    Setup logging configuration
    """
    
    @staticmethod
    def setup_logging(log_file: str = "stock_prediction.log", 
                     level: str = "INFO",
                     console: bool = True):
        """Setup logging configuration"""
        
        log_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                },
                'simple': {
                    'format': '%(levelname)s - %(message)s'
                }
            },
            'handlers': {
                'file': {
                    'class': 'logging.FileHandler',
                    'filename': log_file,
                    'formatter': 'detailed',
                    'level': level
                }
            },
            'root': {
                'level': level,
                'handlers': ['file']
            }
        }
        
        if console:
            log_config['handlers']['console'] = {
                'class': 'logging.StreamHandler',
                'formatter': 'simple',
                'level': 'INFO'
            }
            log_config['root']['handlers'].append('console')
        
        logging.config.dictConfig(log_config)
        logging.info("Logging setup completed")

class DateUtils:
    """
    Date utility functions
    """
    
    @staticmethod
    def get_trading_days(start_date: str, end_date: str) -> List[str]:
        """Get list of trading days between two dates (simplified)"""
        try:
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            return [date.strftime('%Y-%m-%d') for date in dates]
        except Exception as e:
            logging.error(f"Error getting trading days: {str(e)}")
            return []
    
    @staticmethod
    def is_market_hours() -> bool:
        """Check if current time is during market hours (simplified)"""
        now = datetime.now()
        # Indian market hours: 9:15 AM to 3:30 PM
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_open <= now <= market_close
    
    @staticmethod
    def get_next_trading_day() -> str:
        """Get the next trading day"""
        today = datetime.now()
        next_day = today + timedelta(days=1)
        
        # Skip weekends (simplified)
        while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
            next_day += timedelta(days=1)
        
        return next_day.strftime('%Y-%m-%d')

class DataUtils:
    """
    Data utility functions
    """
    
    @staticmethod
    def normalize_data(df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """Normalize data using different methods"""
        df_normalized = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in numeric_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            for col in numeric_cols:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df_normalized[col] = (df[col] - mean_val) / std_val
        
        elif method == 'robust':
            for col in numeric_cols:
                median_val = df[col].median()
                iqr_val = df[col].quantile(0.75) - df[col].quantile(0.25)
                if iqr_val > 0:
                    df_normalized[col] = (df[col] - median_val) / iqr_val
        
        return df_normalized
    
    @staticmethod
    def detect_anomalies(df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Detect anomalies in data"""
        anomalies = pd.DataFrame()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_anomalies = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if not col_anomalies.empty:
                    anomalies = pd.concat([anomalies, col_anomalies])
        
        return anomalies.drop_duplicates()
    
    @staticmethod
    def calculate_rolling_stats(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate rolling statistics"""
        df_rolling = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            df_rolling[f'{col}_rolling_mean'] = df[col].rolling(window=window).mean()
            df_rolling[f'{col}_rolling_std'] = df[col].rolling(window=window).std()
            df_rolling[f'{col}_rolling_min'] = df[col].rolling(window=window).min()
            df_rolling[f'{col}_rolling_max'] = df[col].rolling(window=window).max()
        
        return df_rolling

class FileUtils:
    """
    File utility functions
    """
    
    @staticmethod
    def ensure_directory(directory: str):
        """Ensure directory exists"""
        os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, filepath: str, format: str = 'csv'):
        """Save DataFrame to file"""
        try:
            if format == 'csv':
                df.to_csv(filepath, index=True)
            elif format == 'parquet':
                df.to_parquet(filepath, index=True)
            elif format == 'excel':
                df.to_excel(filepath, index=True)
            logging.info(f"DataFrame saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving DataFrame to {filepath}: {str(e)}")
    
    @staticmethod
    def load_dataframe(filepath: str, format: str = 'csv') -> pd.DataFrame:
        """Load DataFrame from file"""
        try:
            if format == 'csv':
                return pd.read_csv(filepath, index_col=0, parse_dates=True)
            elif format == 'parquet':
                return pd.read_parquet(filepath)
            elif format == 'excel':
                return pd.read_excel(filepath, index_col=0, parse_dates=True)
        except Exception as e:
            logging.error(f"Error loading DataFrame from {filepath}: {str(e)}")
            return pd.DataFrame()

class ValidationUtils:
    """
    Validation utility functions
    """
    
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """Validate stock ticker format"""
        # Basic validation - can be enhanced
        if not ticker or not isinstance(ticker, str):
            return False
        if len(ticker) < 1 or len(ticker) > 20:
            return False
        return True
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> bool:
        """Validate date range"""
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            return start <= end
        except:
            return False
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: List[str] = None) -> bool:
        """Validate DataFrame structure"""
        if df.empty:
            return False
        
        if required_cols:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logging.warning(f"Missing required columns: {missing_cols}")
                return False
        
        return True