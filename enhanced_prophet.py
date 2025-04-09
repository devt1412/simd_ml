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
    
    def interpolate_sparse_data(self, df):
        """Handle sparse data by interpolating missing values"""
        # Create a complete date range
        date_range = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='D')
        complete_df = pd.DataFrame({'ds': date_range})
        
        # Merge with existing data
        df = pd.merge(complete_df, df, on='ds', how='left')
        
        # Interpolate missing values
        df['y'] = df['y'].interpolate(method='linear', limit_direction='both')
        
        return df
    
    def add_custom_features(self, df):
        """Add custom regression features with handling for sparse data"""
        # First interpolate the data to handle gaps
        df = self.interpolate_sparse_data(df)
        
        # Calculate rolling statistics on interpolated data
        df['rolling_mean'] = df['y'].rolling(window=7, min_periods=2).mean()
        df['rolling_std'] = df['y'].rolling(window=7, min_periods=2).std()
        df['rolling_median'] = df['y'].rolling(window=7, min_periods=2).median()
        df['rolling_max'] = df['y'].rolling(window=14, min_periods=3).max()
        
        # Calculate trend direction with more robustness to missing data
        df['trend_direction'] = df['y'].diff().rolling(window=7, min_periods=2).mean()
        df['trend_direction'] = df['trend_direction'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        # Forward fill any remaining NaN values in the features
        feature_columns = ['rolling_mean', 'rolling_std', 'rolling_median', 'rolling_max', 'trend_direction']
        df[feature_columns] = df[feature_columns].ffill().bfill()
        
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
            
            # Adjust changepoint detection based on data sparsity
            data_density = len(df) / (df['ds'].max() - df['ds'].min()).days
            recent_volatility = df['y'].tail(min(30, len(df))).std()
            
            # Increase changepoint_prior_scale for sparse data
            self.changepoint_prior_scale = max(0.05, 
                recent_volatility / df['y'].mean() * (1 + (1 - data_density)))
        
        return super().fit(df, **kwargs)
    
    def predict(self, df):
        """Enhanced prediction with custom features"""
        df = df.copy()  # Create a copy to avoid modifying original data
        
        if self.custom_features:
            # Add seasonal features for prediction
            df = self.add_custom_seasonality(df)
            
            # For prediction, we'll use the last known values from training data
            # and interpolate as needed
            history_end = self.history['ds'].max()
            
            for column in ['rolling_mean', 'rolling_std', 'rolling_median', 
                          'rolling_max', 'trend_direction']:
                # Get the last few known values
                last_values = self.history[column].tail(30)
                # Calculate a smoothed final value
                final_value = last_values.mean()
                df[column] = final_value
        
        return super().predict(df)