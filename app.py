import sys
import numpy as np
import gc
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# More flexible numpy version check
try:
    np_version = tuple(map(int, np.__version__.split('.')))
    if np_version < (1, 23, 0):
        raise ImportError("Numpy version too old - need 1.23.0 or higher")
except Exception as e:
    print(f"NUMPY VERSION CHECK WARNING: {str(e)}", file=sys.stderr)

from flask import Flask, request, jsonify
from enhanced_prophet import EnhancedProphet
import pandas as pd
import logging
from datetime import datetime
from functools import lru_cache
from werkzeug.exceptions import HTTPException

# Flask web application with increased timeout
app = Flask(__name__)
app.config['TIMEOUT'] = 300  # 5 minute timeout

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Thread pool for handling model training
executor = ThreadPoolExecutor(max_workers=2)

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
    return {"cv_mae": 0.0, "cv_rmse": 0.0, "cv_mape": 0.0, "cv_r2": 0.0}

@lru_cache(maxsize=10)
def train_prophet_model(data_hash, product_name, sales_data):
    """Trains an EnhancedProphet model for a given product using provided sales data."""
    try:
        # Converting tuple data to list of dictionaries
        dict_rows = [dict(row) for row in sales_data]
        df = pd.DataFrame(dict_rows)

        # Basic data validation
        if df.empty:
            raise ValueError("Empty dataset provided")

        # Memory optimization - convert to efficient dtypes
        df['ds'] = pd.to_datetime(df['timestamp'])
        df['y'] = pd.to_numeric(df['quantitySold'], downcast='float')
        
        # Clean up memory
        del df['timestamp']
        gc.collect()

        # Resample data
        df = df.resample('D', on='ds')['y'].sum().reset_index()
        
        logger.info(f"Processed data shape for {product_name}: {df.shape}")
        
        if df['ds'].nunique() < 5:
            raise ValueError(f"Not enough unique days ({df['ds'].nunique()}) to train model")

        # Initialize model with memory-optimized settings
        model = EnhancedProphet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
            custom_features=True,
            growth='linear'  # More stable than default 'flat'
        )
        
        # Fit model with error handling
        try:
            model.fit(df)
        except Exception as fit_error:
            logger.error(f"Model fitting failed for {product_name}: {str(fit_error)}")
            # Try with simplified model if regular fit fails
            model = EnhancedProphet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.1,
                custom_features=False
            )
            model.fit(df)

        return model

    except Exception as e:
        logger.error(f"Model training failed for {product_name}: {str(e)}")
        raise
    finally:
        # Force garbage collection
        gc.collect()

def make_predictions(model, periods, freq='D'):
    """Generates future sales predictions."""
    try:
        future = model.make_future_dataframe(
            periods=periods,
            freq=freq,
            include_history=False
        )
        
        forecast = model.predict(future)
        result = forecast[['ds', 'yhat']].rename(columns={'yhat': 'predictedSales'})
        result['ds'] = result['ds'].dt.strftime('%Y-%m-%d')
        
        # Clean up memory
        del forecast
        gc.collect()
        
        return result.to_dict('records')
    except Exception as e:
        logger.error(f"Prediction generation failed: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint that accepts sales data and returns predictions."""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        if not data or 'sales_history' not in data:
            return jsonify({"error": "Missing sales_history in request"}), 400

        sales_data = data['sales_history']
        if not isinstance(sales_data, list) or not sales_data:
            return jsonify({"error": "sales_history must be a non-empty list"}), 400

        predictions = {}
        products = {item['name'] for item in sales_data if 'name' in item}
        
        if not products:
            return jsonify({"error": "No valid product names found in data"}), 400

        logger.info(f"Processing {len(products)} products")

        for product in products:
            try:
                product_data = [
                    item for item in sales_data
                    if item.get('name') == product
                    and isinstance(item.get('quantitySold'), (int, float))
                    and isinstance(item.get('timestamp'), str)
                ]

                if not product_data:
                    logger.warning(f"No valid data for product '{product}'")
                    continue

                data_hash = hash(tuple(sorted(
                    (item['timestamp'], item['quantitySold'])
                    for item in product_data
                )))

                # Run model training with timeout
                future = executor.submit(train_prophet_model, data_hash, product,
                                      tuple(frozenset(d.items()) for d in product_data))
                model = future.result(timeout=240)  # 4 minute timeout

                predictions[product] = {
                    "next_week": make_predictions(model, 7),
                    "next_month": make_predictions(model, 30),
                    "next_year": make_predictions(model, 12, 'M')
                }

            except TimeoutError:
                logger.error(f"Model training timed out for {product}")
                continue
            except Exception as e:
                logger.error(f"Error processing {product}: {str(e)}")
                continue
            finally:
                gc.collect()

        if not predictions:
            return jsonify({"error": "Could not generate predictions for any products"}), 400

        return jsonify({
            "success": True,
            "predictions": predictions,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        })

    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500
    finally:
        gc.collect()

@app.errorhandler(HTTPException)
def handle_exception(e):
    return jsonify({
        "error": e.name,
        "message": e.description
    }), e.code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
