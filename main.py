import requests
import json

URL = "https://simd-ml.onrender.com"

sales_history = [
    # Laptop sales (10 days)
    {"name": "Laptop", "quantitySold": 5, "timestamp": "2025-03-16T10:00:00"},
    {"name": "Laptop", "quantitySold": 8, "timestamp": "2025-03-17T10:00:00"},
    {"name": "Laptop", "quantitySold": 4, "timestamp": "2025-03-18T10:00:00"},
    {"name": "Laptop", "quantitySold": 6, "timestamp": "2025-03-19T10:00:00"},
    {"name": "Laptop", "quantitySold": 10, "timestamp": "2025-03-20T10:00:00"},
    {"name": "Laptop", "quantitySold": 9, "timestamp": "2025-03-21T10:00:00"},
    {"name": "Laptop", "quantitySold": 3, "timestamp": "2025-03-22T10:00:00"},
    {"name": "Laptop", "quantitySold": 7, "timestamp": "2025-03-23T10:00:00"},
    {"name": "Laptop", "quantitySold": 5, "timestamp": "2025-03-24T10:00:00"},
    {"name": "Laptop", "quantitySold": 6, "timestamp": "2025-03-25T10:00:00"},

    # Mouse sales (10 days)
    {"name": "Mouse", "quantitySold": 2, "timestamp": "2025-03-16T10:00:00"},
    {"name": "Mouse", "quantitySold": 3, "timestamp": "2025-03-17T10:00:00"},
    {"name": "Mouse", "quantitySold": 1, "timestamp": "2025-03-18T10:00:00"},
    {"name": "Mouse", "quantitySold": 2, "timestamp": "2025-03-19T10:00:00"},
    {"name": "Mouse", "quantitySold": 4, "timestamp": "2025-03-20T10:00:00"},
    {"name": "Mouse", "quantitySold": 3, "timestamp": "2025-03-21T10:00:00"},
    {"name": "Mouse", "quantitySold": 2, "timestamp": "2025-03-22T10:00:00"},
    {"name": "Mouse", "quantitySold": 1, "timestamp": "2025-03-23T10:00:00"},
    {"name": "Mouse", "quantitySold": 2, "timestamp": "2025-03-24T10:00:00"},
    {"name": "Mouse", "quantitySold": 3, "timestamp": "2025-03-25T10:00:00"},
]

# Validate and format sales history
def validate_and_format_sales_history(raw_sales_history):
    formatted_sales_history = []
    for item in raw_sales_history:
        if not isinstance(item, dict):
            raise ValueError("Each item must be a dictionary")
        
        if not all(key in item for key in ["name", "quantitySold", "timestamp"]):
            raise ValueError("Missing required fields (name, quantitySold, or timestamp)")
        
        quantity = item["quantitySold"]
        if isinstance(quantity, str):
            try:
                quantity = float(quantity)
            except ValueError:
                raise ValueError("quantitySold must be a number or numeric string")
        elif not isinstance(quantity, (int, float)):
            raise ValueError("quantitySold must be a number or numeric string")
        
        timestamp = item["timestamp"]
        try:
            # Validate timestamp format
            timestamp = str(timestamp)
            _ = json.dumps({"timestamp": timestamp})
        except Exception:
            raise ValueError("Invalid timestamp format. Use ISO format (YYYY-MM-DD HH:MM:SS)")
        
        formatted_sales_history.append({
            "name": str(item["name"]),
            "quantitySold": quantity,
            "timestamp": timestamp,
        })
    return formatted_sales_history

try:
    formatted_sales_history = validate_and_format_sales_history(sales_history)
    payload = {
        "sales_history": formatted_sales_history
    }

    response = requests.post(URL + "/predict", json=payload)

    print("Status Code:", response.status_code)

    try:
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
    except json.decoder.JSONDecodeError:
        print("Failed to decode JSON. Raw response:")
        print(response.text)
except ValueError as e:
    print(f"Data validation error: {e}")
