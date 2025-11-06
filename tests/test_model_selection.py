"""
Test Model Selection Feature

Tests various model selection scenarios including:
- Default model behavior
- Explicit model selection
- Invalid model/version handling
- Cache performance
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app

client = TestClient(app)


class TestModelSelection:
    """Test model selection functionality"""
    
    def test_default_model_selection(self):
        """Test that default model (Ridge v2.2.0) is used when not specified"""
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
        assert "model_info" in data
        assert data["model_info"]["model_type"] == "ridge"
        assert data["model_info"]["version"] == "2.2.0"
    
    def test_explicit_ridge_selection(self):
        """Test explicitly selecting Ridge v2.2.0"""
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
            "projection_months": 3,
            "model_type": "ridge",
            "model_version": "2.2.0"
        }
        
        response = client.post("/api/v1/predict/forward", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_info"]["model_type"] == "ridge"
        assert data["model_info"]["version"] == "2.2.0"
    
    def test_xgboost_selection(self):
        """Test selecting XGBoost v2.3.0"""
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
            "projection_months": 3,
            "model_type": "xgboost",
            "model_version": "2.3.0"
        }
        
        response = client.post("/api/v1/predict/forward", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_info"]["model_type"] == "xgboost"
        assert data["model_info"]["version"] == "2.3.0"
    
    def test_lightgbm_selection(self):
        """Test selecting LightGBM v2.3.0"""
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
            "projection_months": 3,
            "model_type": "lightgbm",
            "model_version": "2.3.0"
        }
        
        response = client.post("/api/v1/predict/forward", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_info"]["model_type"] == "lightgbm"
        assert data["model_info"]["version"] == "2.3.0"
    
    def test_invalid_model_type(self):
        """Test that invalid model type returns 400 error"""
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
            "projection_months": 3,
            "model_type": "invalid_model",
            "model_version": "2.2.0"
        }
        
        response = client.post("/api/v1/predict/forward", json=request_data)
        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()
    
    def test_invalid_version(self):
        """Test that invalid version returns 400 error"""
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
            "projection_months": 3,
            "model_type": "ridge",
            "model_version": "9.9.9"
        }
        
        response = client.post("/api/v1/predict/forward", json=request_data)
        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()
    
    def test_invalid_model_version_combination(self):
        """Test that invalid model/version combination returns 400 error"""
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
            "projection_months": 3,
            "model_type": "xgboost",
            "model_version": "2.2.0"  # XGBoost not available in v2.2.0
        }
        
        response = client.post("/api/v1/predict/forward", json=request_data)
        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()
    
    def test_inverse_prediction_with_model_selection(self):
        """Test inverse prediction with explicit model selection"""
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
            "target_months": 3,
            "model_type": "xgboost",
            "model_version": "2.3.0"
        }
        
        response = client.post("/api/v1/predict/inverse", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_info"]["model_type"] == "xgboost"
        assert data["model_info"]["version"] == "2.3.0"
    
    def test_partial_prediction_with_model_selection(self):
        """Test partial lever prediction with explicit model selection"""
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
            ],
            "model_type": "ridge",
            "model_version": "2.2.0"
        }
        
        response = client.post("/api/v1/predict/partial", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_info"]["model_type"] == "ridge"
        assert data["model_info"]["version"] == "2.2.0"


class TestModelsEndpoint:
    """Test the /models endpoint"""
    
    def test_get_available_models(self):
        """Test GET /api/v1/predict/models endpoint"""
        response = client.get("/api/v1/predict/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert "count" in data
        assert "default_model" in data
        assert "cache_stats" in data
        
        # Check default model
        assert data["default_model"]["model_type"] == "ridge"
        assert data["default_model"]["version"] == "2.2.0"
        
        # Check models list
        assert len(data["models"]) > 0
        
        # Verify at least Ridge v2.2.0 is available
        model_keys = [(m["model_type"], m["version"]) for m in data["models"]]
        assert ("ridge", "2.2.0") in model_keys
        
        # Check model structure
        first_model = data["models"][0]
        assert "model_type" in first_model
        assert "version" in first_model
        assert "is_default" in first_model
        assert "targets" in first_model
        assert "n_features" in first_model
        assert "n_targets" in first_model
        assert "prediction_horizons" in first_model
    
    def test_models_include_performance_metrics(self):
        """Test that v2.3.0 models include performance metrics"""
        response = client.get("/api/v1/predict/models")
        assert response.status_code == 200
        
        data = response.json()
        
        # Find XGBoost v2.3.0 model
        xgboost_model = None
        for model in data["models"]:
            if model["model_type"] == "xgboost" and model["version"] == "2.3.0":
                xgboost_model = model
                break
        
        if xgboost_model:
            # Check if performance metrics are included
            assert "performance" in xgboost_model or "training_date" in xgboost_model
            if "performance" in xgboost_model:
                assert "r2" in xgboost_model["performance"]
                assert "rmse" in xgboost_model["performance"]


class TestCachePerformance:
    """Test cache performance and behavior"""
    
    def test_cache_hit_on_repeat_requests(self):
        """Test that repeated requests use cached model"""
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
            "projection_months": 3,
            "model_type": "xgboost",
            "model_version": "2.3.0"
        }
        
        # First request (cache miss)
        response1 = client.post("/api/v1/predict/forward", json=request_data)
        assert response1.status_code == 200
        
        # Get cache stats before second request
        models_response = client.get("/api/v1/predict/models")
        cache_stats_before = models_response.json()["cache_stats"]
        initial_hits = cache_stats_before.get("cache_hits", 0)
        
        # Second request (should be cache hit)
        response2 = client.post("/api/v1/predict/forward", json=request_data)
        assert response2.status_code == 200
        
        # Get cache stats after second request
        models_response = client.get("/api/v1/predict/models")
        cache_stats_after = models_response.json()["cache_stats"]
        final_hits = cache_stats_after.get("cache_hits", 0)
        
        # Cache hits should have increased
        assert final_hits > initial_hits
    
    def test_multiple_models_in_cache(self):
        """Test that multiple models can be cached simultaneously"""
        # Request with Ridge
        ridge_request = {
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
            "projection_months": 3,
            "model_type": "ridge",
            "model_version": "2.2.0"
        }
        
        # Request with XGBoost
        xgboost_request = ridge_request.copy()
        xgboost_request["model_type"] = "xgboost"
        xgboost_request["model_version"] = "2.3.0"
        
        # Make both requests
        response1 = client.post("/api/v1/predict/forward", json=ridge_request)
        assert response1.status_code == 200
        
        response2 = client.post("/api/v1/predict/forward", json=xgboost_request)
        assert response2.status_code == 200
        
        # Check cache stats
        models_response = client.get("/api/v1/predict/models")
        cache_stats = models_response.json()["cache_stats"]
        
        # Should have at least 2 models cached
        cached_models = cache_stats.get("cached_models", [])
        assert len(cached_models) >= 2


class TestHealthCheck:
    """Test health check with model info"""
    
    def test_health_check_includes_model_info(self):
        """Test that health check includes model information"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_version" in data
        assert "model_type" in data
        assert "available_models" in data
        assert "cached_models" in data
        assert "cache_hit_rate" in data
        
        assert data["status"] == "healthy"
        assert data["model_type"] == "ridge"
        assert data["model_version"] == "2.2.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

