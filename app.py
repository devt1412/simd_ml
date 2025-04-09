import sys
import numpy as np

try:
    if np.__version__ != '1.23.5':
        raise ImportError("Incorrect numpy version detected")
except Exception as e:
    print(f"NUMPY VERSION CHECK FAILED: {str(e)}", file=sys.stderr)
    sys.exit(1)

from flask import Flask, request, jsonify
from enhanced_prophet import EnhancedProphet
import pandas as pd
import logging
from datetime import datetime
from functools import lru_cache
from werkzeug.exceptions import HTTPException

# Flask web application
app = Flask(__name__)

# Logging to print messages to the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Helper functions for metrics calculation
def calculate_metrics(actual, predicted):
    """Calculate accuracy metrics."""
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}

def cross_validate_prophet(model, df):
    """Perform cross-validation for Prophet model."""
    # Placeholder for cross-validation logic
    return {"cv_mae": 0.0, "cv_rmse": 0.0, "cv_mape": 0.0, "cv_r2": 0.0}

# Training and reusing EnhancedProphet models based on unique input data
@lru_cache(maxsize=10)
def train_prophet_model(data_hash, product_name, sales_data):
    """Trains an EnhancedProphet model for a given product using provided sales data."""
    try:
        # Converting tuple data to list of dictionaries for DataFrame creation
        dict_rows = [dict(row) for row in sales_data]
        df = pd.DataFrame(dict_rows)

        # Preparing DataFrame with timestamp and quantity sold
        df['ds'] = pd.to_datetime(df['timestamp'])
        df['y'] = df['quantitySold'].astype(float)

        # Grouping sales by day and sum up quantities (handles multiple entries per day)
        df = df.resample('D', on='ds')['y'].sum().reset_index()

        logger.info(f"Resampled DataFrame for {product_name}:\n{df}")
        logger.info(f"Unique days for {product_name}: {df['ds'].nunique()}")

        # When not enough data to make accurate forecasts
        if df['ds'].nunique() < 5:
            raise ValueError(f"Not enough unique days to train model for {product_name}")

        # Split data for validation (80% train, 20% test)
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size]
        test_df = df[train_size:]

        # Initialize and train the enhanced model
        model = EnhancedProphet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
            custom_features=True
        )
        
        model.fit(train_df)

        # Evaluate model performance
        future = model.make_future_dataframe(periods=len(test_df))
        forecast = model.predict(future)
        
        # Calculate accuracy metrics
        merged = forecast[-len(test_df):].merge(test_df, on='ds', how='inner')
        metrics = calculate_metrics(merged['y'], merged['yhat'])
        
        # Perform cross-validation
        cv_metrics = cross_validate_prophet(model, df)
        
        # Log performance metrics
        logger.info(f"Model metrics for {product_name}:")
        logger.info(f"MAE: {metrics['mae']:.2f}")
        logger.info(f"RMSE: {metrics['rmse']:.2f}")
        logger.info(f"MAPE: {metrics['mape']:.2f}%")
        logger.info(f"R2 Score: {metrics['r2']:.2f}")
        
        # Store metrics with the model
        model.metrics = metrics
        model.cv_metrics = cv_metrics
        
        return model

    except Exception as e:
        logger.error(f"Model training failed for {product_name}: {str(e)}")
        raise

def make_predictions(model, periods, freq='D'):
    """Generates future sales predictions from the trained model."""
    # Creating a future date range to predict over
    future = model.make_future_dataframe(
        periods=periods,
        freq=freq,
        include_history=False
    )

    # Generating the forecast and format it
    forecast = model.predict(future)
    result = forecast[['ds', 'yhat']].rename(columns={'yhat': 'predictedSales'})
    result['ds'] = result['ds'].dt.strftime('%Y-%m-%d')
    return result.to_dict('records')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint that accepts sales data and returns predictions."""
    try:
        # Checking if request is in JSON format
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()

        # Validating that sales history is provided and in correct format
        if 'sales_history' not in data or not isinstance(data['sales_history'], list):
            return jsonify({"error": "Missing or invalid sales_history"}), 400

        predictions = {}
        sales_data = data['sales_history']

        # Geting unique product names from the data
        products = {item['name'] for item in sales_data if 'name' in item}

        logger.info(f"Received data for products: {products}")

        for product in products:
            logger.info(f"Processing product: {product}")

            # Filtering out any incomplete or invalid entries for the current product
            product_data = [
                item for item in sales_data
                if item.get('name') == product
                and isinstance(item.get('quantitySold'), (int, float))
                and isinstance(item.get('timestamp'), str)
            ]

            logger.info(f"Filtered data for {product}: {product_data}")

            if not product_data:
                logger.warning(f"No valid data for product '{product}' after filtering.")
                continue

            try:
                # Creating a hash from the data to use for model caching
                data_hash = hash(tuple(sorted(
                    (item['timestamp'], item['quantitySold'])
                    for item in product_data
                )))

                # Training or retrieve the cached model
                model = train_prophet_model(
                    data_hash,
                    product,
                    tuple(frozenset(d.items()) for d in product_data)  # Needs to be hashable
                )

                # Generating predictions for next week, month, and year
                predictions[product] = {
                    "next_week": make_predictions(model, 7),
                    "next_month": make_predictions(model, 30),
                    "next_year": make_predictions(model, 12, 'M')
                }

            except Exception as e:
                logger.error(f"Prediction failed for {product}: {str(e)}")
                continue

        # If no valid predictions could be made
        if not predictions:
            return jsonify({"error": "No valid products found"}), 400

        return jsonify({
            "success": True,
            "predictions": predictions,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        })

    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# Handling any HTTP errors in a friendly JSON format
@app.errorhandler(HTTPException)
def handle_exception(e):
    return jsonify({
        "error": e.name,
        "message": e.description
    }), e.code

# Starting the Flask development server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
