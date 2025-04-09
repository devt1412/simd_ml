import sys
import numpy as np

try:
    if np.__version__ != '1.23.5':
        raise ImportError("Incorrect numpy version detected")
except Exception as e:
    print(f"NUMPY VERSION CHECK FAILED: {str(e)}", file=sys.stderr)
    sys.exit(1)

from flask import Flask, request, jsonify
from prophet import Prophet
import pandas as pd
import logging
from datetime import datetime
from functools import lru_cache
from werkzeug.exceptions import HTTPException

#Flask web application
app = Flask(__name__)

#Logging to print messages to the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Training and reusing Prophet models based on unique input data
@lru_cache(maxsize=10)
def train_prophet_model(data_hash, product_name, sales_data):
    """Trains a Prophet model for a given product using provided sales data."""
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

        # Initializing and training the forecasting model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(df)
        return model

    except Exception as e:
        logger.error(f"Model training failed for {product_name}: {str(e)}")
        raise

def calculate_accuracy_metrics(model, df):
    """Calculates accuracy metrics for the model predictions."""
    # Get the predictions for historical data
    historical_forecast = model.predict(model.history)
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    y_true = model.history['y'].values
    y_pred = historical_forecast['yhat'].values
    
    # Calculate MAPE avoiding division by zero
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {
        'mape': round(mape, 2),  # MAPE as percentage
        'mae': round(mae, 2),    # Mean Absolute Error
        'accuracy': round(100 - mape, 2)  # Convert MAPE to accuracy percentage
    }

def make_predictions(model, periods, freq='D'):
    """Generates future sales predictions from the trained model."""
    # Calculate accuracy metrics
    accuracy_metrics = calculate_accuracy_metrics(model, model.history)
    
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
    
    return {
        'predictions': result.to_dict('records'),
        'accuracy_metrics': accuracy_metrics
    }

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