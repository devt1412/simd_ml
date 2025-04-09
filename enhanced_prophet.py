from prophet import Prophet
import numpy as np
import pandas as pd

class EnhancedProphet(Prophet):
    """Custom forecasting model extending Prophet with additional capabilities"""
    
    def __init__(self, *args, **kwargs):
        self.custom_features = kwargs.pop('custom_features', True)
        super().__init__(*args, **kwargs)
        
    def add_custom_seasonality(self, df):
        """Add domain-specific seasonal patterns"""
        df['special_season'] = np.sin(2 * np.pi * df['ds'].dt.dayofyear / 365.25)
        return df
    
    def add_custom_features(self, df):
        """Add custom regression features"""
        df['rolling_mean'] = df['y'].rolling(window=7, min_periods=1).mean()
        df['rolling_std'] = df['y'].rolling(window=7, min_periods=1).std()
        
        # Add more sophisticated features
        df['rolling_median'] = df['y'].rolling(window=7, min_periods=1).median()
        df['rolling_max'] = df['y'].rolling(window=14, min_periods=1).max()
        df['trend_direction'] = df['y'].diff().rolling(window=7).mean().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        return df
    
    def prepare_dataframe(self, df):
        """Prepare DataFrame with all custom features"""
        if self.custom_features:
            df = self.add_custom_seasonality(df)
            df = self.add_custom_features(df)
        return df
    
    def fit(self, df, **kwargs):
        """Enhanced fitting process with custom features"""
        df = df.copy()  # Create a copy to avoid modifying original data
        
        if self.custom_features:
            df = self.prepare_dataframe(df)
            
            # Add all custom regressors
            for column in ['special_season', 'rolling_mean', 'rolling_std', 
                         'rolling_median', 'rolling_max', 'trend_direction']:
                self.add_regressor(column)
            
            # Dynamic changepoint detection
            recent_volatility = df['y'].tail(30).std()
            self.changepoint_prior_scale = max(0.05, recent_volatility / df['y'].mean())
        
        return super().fit(df, **kwargs)
    
    def predict(self, df):
        """Enhanced prediction with custom features"""
        df = df.copy()  # Create a copy to avoid modifying original data
        
        if self.custom_features:
            # Add seasonal features for prediction
            df = self.add_custom_seasonality(df)
            
            # For prediction, we need to handle the rolling features carefully
            # We'll use the last known values from training data
            df['rolling_mean'] = self.history['rolling_mean'].iloc[-1]
            df['rolling_std'] = self.history['rolling_std'].iloc[-1]
            df['rolling_median'] = self.history['rolling_median'].iloc[-1]
            df['rolling_max'] = self.history['rolling_max'].iloc[-1]
            df['trend_direction'] = self.history['trend_direction'].iloc[-1]
        
        return super().predict(df)