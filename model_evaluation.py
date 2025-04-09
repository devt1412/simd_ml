import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """Calculate various accuracy metrics for the model"""
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'r2': r2_score(y_true, y_pred)
    }
    return metrics

def cross_validate_prophet(model, df, initial_days=30, horizon_days=7):
    """Perform rolling cross-validation for Prophet model"""
    results = []
    cutoffs = []
    
    # Generate cutoff dates for rolling window
    min_date = df['ds'].min()
    max_date = df['ds'].max()
    
    current = min_date + pd.Timedelta(days=initial_days)
    while current < max_date - pd.Timedelta(days=horizon_days):
        cutoffs.append(current)
        current += pd.Timedelta(days=horizon_days)
    
    # Perform cross validation
    for cutoff in cutoffs:
        # Train on data before cutoff
        train = df[df['ds'] <= cutoff].copy()
        model.fit(train)
        
        # Predict for horizon_days after cutoff
        future = model.make_future_dataframe(periods=horizon_days)
        forecast = model.predict(future)
        
        # Get actual values
        actual = df[
            (df['ds'] > cutoff) & 
            (df['ds'] <= cutoff + pd.Timedelta(days=horizon_days))
        ]
        
        # Calculate metrics
        merged = forecast.merge(actual, on='ds', how='inner')
        metrics = calculate_metrics(merged['y'], merged['yhat'])
        metrics['cutoff'] = cutoff
        results.append(metrics)
    
    return pd.DataFrame(results)