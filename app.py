import sys
import numpy as np

# Check numpy version compatibility
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

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Cache models to avoid retraining on identical data
@lru_cache(maxsize=10)
def train_prophet_model(data_hash, product_name, sales_data):
    """Train and cache Prophet model for specific product data"""
    try:
        df = pd.DataFrame(sales_data)
        df['ds'] = pd.to_datetime(df['timestamp'])
        df['y'] = df['quantitySold'].astype(float)

        # Resample to daily sums (handles multiple entries per day)
        df = df.resample('D', on='ds')['y'].sum().reset_index()

        logger.info(f"Resampled DataFrame for {product_name}:\n{df}")
        logger.info(f"Unique days for {product_name}: {df['ds'].nunique()}")

        if df['ds'].nunique() < 5:
            raise ValueError(f"Not enough unique days to train model for {product_name}")

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

def make_predictions(model, periods, freq='D'):
    """Generate future predictions from trained model"""
    future = model.make_future_dataframe(
        periods=periods,
        freq=freq,
        include_history=False
    )
    forecast = model.predict(future)
    result = forecast[['ds', 'yhat']].rename(columns={'yhat': 'predictedSales'})
    result['ds'] = result['ds'].dt.strftime('%Y-%m-%d')
    return result.to_dict('records')

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()

        if 'sales_history' not in data or not isinstance(data['sales_history'], list):
            return jsonify({"error": "Missing or invalid sales_history"}), 400

        predictions = {}
        sales_data = data['sales_history']
        products = {item['name'] for item in sales_data if 'name' in item}

        logger.info(f"Received data for products: {products}")

        for product in products:
            logger.info(f"Processing product: {product}")

            # Filter and validate
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
                data_hash = hash(tuple(sorted(
                    (item['timestamp'], item['quantitySold'])
                    for item in product_data
                )))

                model = train_prophet_model(
                    data_hash,
                    product,
                    tuple(frozenset(d.items()) for d in product_data)  # must be hashable
                )

                predictions[product] = {
                    "next_week": make_predictions(model, 7),
                    "next_month": make_predictions(model, 30),
                    "next_year": make_predictions(model, 12, 'M')
                }

            except Exception as e:
                logger.error(f"Prediction failed for {product}: {str(e)}")
                continue

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

@app.errorhandler(HTTPException)
def handle_exception(e):
    return jsonify({
        "error": e.name,
        "message": e.description
    }), e.code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
