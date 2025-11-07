"""
Unit Tests for Studio Revenue Simulator API
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Studio Revenue Simulator" in data["service"]


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_version" in data
    assert "model_type" in data
    assert data["model_version"] == "2.2.0"


def test_forward_prediction_basic():
    """Test forward prediction with valid input"""
    request_data = {
        "studio_id": "STU001",
        "levers": {
            "retention_rate": 0.75,
            "avg_ticket_price": 150.0,
            "class_attendance_rate": 0.70,
            "new_members": 25,
            "staff_utilization_rate": 0.85,
            "upsell_rate": 0.25,
            "total_classes_held": 120,
            "total_members": 200
        },
        "projection_months": 3
    }
    
    response = client.post("/api/v1/predict/forward", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "scenario_id" in data
    assert "studio_id" in data
    assert "predictions" in data
    assert "total_projected_revenue" in data
    assert "model_version" in data
    
    # Check predictions structure
    assert len(data["predictions"]) == 3
    for pred in data["predictions"]:
        assert "month" in pred
        assert "revenue" in pred
        assert "member_count" in pred
        assert "confidence_score" in pred
        assert pred["revenue"] > 0
        assert pred["member_count"] > 0
        assert 0 <= pred["confidence_score"] <= 1


def test_forward_prediction_invalid_levers():
    """Test forward prediction with invalid lever values"""
    request_data = {
        "studio_id": "STU001",
        "levers": {
            "retention_rate": 1.5,  # Invalid: > 1.0
            "avg_ticket_price": 150.0,
            "class_attendance_rate": 0.70,
            "new_members": 25,
            "staff_utilization_rate": 0.85,
            "upsell_rate": 0.25,
            "total_classes_held": 120,
            "total_members": 200
        },
        "projection_months": 3
    }
    
    response = client.post("/api/v1/predict/forward", json=request_data)
    assert response.status_code == 422  # Validation error


def test_inverse_prediction_basic():
    """Test inverse prediction with valid input"""
    request_data = {
        "studio_id": "STU001",
        "target_revenue": 35000.0,
        "current_state": {
            "retention_rate": 0.70,
            "avg_ticket_price": 140.0,
            "class_attendance_rate": 0.65,
            "new_members": 20,
            "staff_utilization_rate": 0.80,
            "upsell_rate": 0.20,
            "total_classes_held": 100,
            "total_members": 180
        },
        "constraints": {
            "max_retention_increase": 0.05,
            "max_ticket_increase": 20.0,
            "max_new_members_increase": 10
        },
        "target_months": 3
    }
    
    response = client.post("/api/v1/predict/inverse", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "optimization_id" in data
    assert "studio_id" in data
    assert "target_revenue" in data
    assert "achievable_revenue" in data
    assert "recommended_levers" in data
    assert "lever_changes" in data
    assert "action_plan" in data
    
    # Check achievable revenue is positive
    assert data["achievable_revenue"] > 0
    
    # Check recommended levers are within bounds
    levers = data["recommended_levers"]
    assert 0.5 <= levers["retention_rate"] <= 1.0
    assert 50.0 <= levers["avg_ticket_price"] <= 500.0


def test_partial_prediction_basic():
    """Test partial lever prediction with valid input"""
    request_data = {
        "studio_id": "STU001",
        "input_levers": {
            "retention_rate": 0.75,
            "avg_ticket_price": 150.0,
            "total_members": 200
        },
        "output_levers": [
            "class_attendance_rate",
            "new_members",
            "total_revenue"
        ]
    }
    
    response = client.post("/api/v1/predict/partial", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction_id" in data
    assert "studio_id" in data
    assert "input_levers" in data
    assert "predicted_levers" in data
    assert "overall_confidence" in data
    
    # Check predicted levers
    assert len(data["predicted_levers"]) == 3
    for pred in data["predicted_levers"]:
        assert "lever_name" in pred
        assert "predicted_value" in pred
        assert "confidence_score" in pred
        assert pred["predicted_value"] > 0
        assert 0 <= pred["confidence_score"] <= 1


def test_partial_prediction_revenue_only():
    """Test partial prediction for revenue only"""
    request_data = {
        "studio_id": "STU001",
        "input_levers": {
            "retention_rate": 0.80,
            "avg_ticket_price": 160.0,
            "class_attendance_rate": 0.75,
            "new_members": 30,
            "staff_utilization_rate": 0.85,
            "upsell_rate": 0.30,
            "total_classes_held": 130,
            "total_members": 220
        },
        "output_levers": ["total_revenue"]
    }
    
    response = client.post("/api/v1/predict/partial", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert len(data["predicted_levers"]) == 1
    assert data["predicted_levers"][0]["lever_name"] == "total_revenue"
    assert data["predicted_levers"][0]["predicted_value"] > 20000  # Reasonable revenue


def test_list_studios():
    """Test list studios endpoint"""
    response = client.get("/api/v1/studios")
    assert response.status_code == 200
    data = response.json()
    assert "studios" in data
    assert "count" in data


def test_get_studio_details():
    """Test get studio details endpoint"""
    response = client.get("/api/v1/studios/STU001")
    assert response.status_code == 200
    data = response.json()
    # Should return studio summary or error
    assert isinstance(data, dict)


def test_forward_prediction_extended_months():
    """Test forward prediction with more than 3 months"""
    request_data = {
        "studio_id": "STU001",
        "levers": {
            "retention_rate": 0.75,
            "avg_ticket_price": 150.0,
            "class_attendance_rate": 0.70,
            "new_members": 25,
            "staff_utilization_rate": 0.85,
            "upsell_rate": 0.25,
            "total_classes_held": 120,
            "total_members": 200
        },
        "projection_months": 6
    }
    
    response = client.post("/api/v1/predict/forward", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert len(data["predictions"]) == 6
    
    # Check confidence decreases over time
    confidences = [p["confidence_score"] for p in data["predictions"]]
    assert confidences[0] > confidences[-1]


def test_prediction_metrics_loaded():
    """Test that prediction_metrics are loaded from actual model performance"""
    request_data = {
        "studio_id": "STU001",
        "levers": {
            "retention_rate": 0.75,
            "avg_ticket_price": 150.0,
            "class_attendance_rate": 0.70,
            "new_members": 25,
            "staff_utilization_rate": 0.85,
            "upsell_rate": 0.25,
            "total_classes_held": 120,
            "total_members": 200
        },
        "projection_months": 3
    }
    
    response = client.post("/api/v1/predict/forward", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    
    # Verify prediction_metrics exist
    assert "prediction_metrics" in data, "prediction_metrics should be present in response"
    
    metrics = data["prediction_metrics"]
    
    # Verify all expected metrics are present
    expected_metrics = [
        'rmse', 'mae', 'r2_score', 'mape',
        'accuracy_within_5pct', 'accuracy_within_10pct',
        'forecast_accuracy', 'directional_accuracy',
        'confidence_level'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics, f"{metric} should be present in prediction_metrics"
    
    # Verify metrics are loaded from actual model results (not hardcoded defaults)
    # For v2.2.0 ridge model, overall_rmse should be ~535.1 (not default 2500.0)
    # We allow for some variation due to degradation/confidence adjustments
    assert metrics['rmse'] < 2000, \
        f"RMSE should be based on actual model performance (~535), got {metrics['rmse']}"
    
    # R² should be very high for v2.2.0 (0.999), not the default 0.82
    assert metrics['r2_score'] > 0.90, \
        f"R² should be based on actual model performance (>0.90), got {metrics['r2_score']}"
    
    # MAPE should be low (~0.018 or 1.8%), not default 0.08 (8%)
    assert metrics['mape'] < 0.05, \
        f"MAPE should be based on actual model performance (<5%), got {metrics['mape']}"
    
    print(f"\n✓ Metrics loaded successfully from actual model performance:")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  MAE: {metrics['mae']:.2f}")
    print(f"  R²: {metrics['r2_score']:.3f}")
    print(f"  MAPE: {metrics['mape']:.4f}")


def test_inverse_prediction_metrics():
    """Test that inverse prediction also includes real metrics"""
    request_data = {
        "studio_id": "STU001",
        "target_revenue": 35000.0,
        "current_state": {
            "retention_rate": 0.70,
            "avg_ticket_price": 140.0,
            "class_attendance_rate": 0.65,
            "new_members": 20,
            "staff_utilization_rate": 0.80,
            "upsell_rate": 0.20,
            "total_classes_held": 100,
            "total_members": 180
        },
        "target_months": 3
    }
    
    response = client.post("/api/v1/predict/inverse", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    
    # Verify prediction_metrics exist
    assert "prediction_metrics" in data, "prediction_metrics should be present in inverse response"
    
    metrics = data["prediction_metrics"]
    
    # Verify metrics are reasonable (based on actual model performance, not defaults)
    assert metrics['rmse'] < 2000, "RMSE should be based on actual model performance"
    assert metrics['r2_score'] > 0.85, "R² should be based on actual model performance"
    
    print(f"\n✓ Inverse prediction metrics loaded successfully:")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  R²: {metrics['r2_score']:.3f}")


def test_partial_prediction_metrics():
    """Test that partial prediction also includes real metrics"""
    request_data = {
        "studio_id": "STU001",
        "input_levers": {
            "retention_rate": 0.75,
            "avg_ticket_price": 150.0,
            "total_members": 200
        },
        "output_levers": ["total_revenue"]
    }
    
    response = client.post("/api/v1/predict/partial", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    
    # Verify prediction_metrics exist
    assert "prediction_metrics" in data, "prediction_metrics should be present in partial response"
    
    metrics = data["prediction_metrics"]
    
    # Verify metrics are reasonable
    assert metrics['rmse'] < 2000, "RMSE should be based on actual model performance"
    assert metrics['r2_score'] > 0.85, "R² should be based on actual model performance"
    
    print(f"\n✓ Partial prediction metrics loaded successfully:")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  R²: {metrics['r2_score']:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

