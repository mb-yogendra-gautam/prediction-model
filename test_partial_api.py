"""
Test script for partial prediction endpoint
"""
import requests
import json

# API endpoint
url = "http://localhost:8000/api/predict/partial"

# Test request
request_data = {
    "studio_id": "STU001",
    "input_levers": {
        "total_members": 150,
        "retention_rate": 0.75,
        "avg_ticket_price": 120
    },
    "output_levers": ["total_revenue", "class_attendance_rate"],
    "projection_months": 3
}

print("Testing /predict/partial endpoint...")
print(f"Request: {json.dumps(request_data, indent=2)}")
print("\nSending request...")

try:
    response = requests.post(url, json=request_data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"\nError: {e}")
    if hasattr(e, 'response') and e.response:
        print(f"Response text: {e.response.text}")

