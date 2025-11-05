# Studio Revenue Simulator API Documentation

## Overview

The Studio Revenue Simulator API provides machine learning-powered predictions for fitness studio revenue optimization. It offers three main capabilities:

1. **Forward Prediction**: Predict future revenue based on business lever adjustments
2. **Inverse Optimization**: Find optimal lever values to achieve target revenue
3. **Partial Lever Prediction**: Predict unknown levers based on known values

---

## Base URL

```
http://localhost:8000
```

---

## Authentication

Currently, no authentication is required (MVP version).

---

## Endpoints

### Health Check

#### GET `/`
Basic health check

**Response:**
```json
{
  "service": "Studio Revenue Simulator API",
  "status": "healthy",
  "version": "1.0.0",
  "message": "Welcome! Visit /docs for API documentation"
}
```

#### GET `/api/v1/health`
Detailed health check with model information

**Response:**
```json
{
  "status": "healthy",
  "model_version": "2.2.0",
  "model_type": "ridge",
  "n_features": 15,
  "available_versions": ["2.2.0", "2.1.0", "2.0.0"],
  "timestamp": "2025-11-05T23:45:00"
}
```

---

### Forward Prediction

#### POST `/api/v1/predict/forward`

Predict future revenue based on business lever inputs.

**Request Body:**
```json
{
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
```

**Lever Constraints:**
- `retention_rate`: 0.5 - 1.0 (member retention rate)
- `avg_ticket_price`: $50 - $500 (average monthly price)
- `class_attendance_rate`: 0.4 - 1.0 (attendance rate)
- `new_members`: 0 - 100 (new members per month)
- `staff_utilization_rate`: 0.6 - 1.0 (staff utilization)
- `upsell_rate`: 0.0 - 0.5 (upsell rate)
- `total_classes_held`: 50 - 500 (classes per month)
- `total_members`: 50+ (current member count)

**Response:**
```json
{
  "scenario_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "studio_id": "STU001",
  "predictions": [
    {
      "month": 1,
      "revenue": 32450.75,
      "member_count": 205,
      "retention_rate": 0.75,
      "confidence_score": 0.85
    },
    {
      "month": 2,
      "revenue": 33120.50,
      "member_count": 208,
      "retention_rate": 0.75,
      "confidence_score": 0.80
    },
    {
      "month": 3,
      "revenue": 33890.25,
      "member_count": 210,
      "retention_rate": 0.76,
      "confidence_score": 0.75
    }
  ],
  "total_projected_revenue": 99461.50,
  "average_confidence": 0.80,
  "model_version": "2.2.0",
  "timestamp": "2025-11-05T23:45:00"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/forward" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

---

### Inverse Optimization

#### POST `/api/v1/predict/inverse`

Find optimal lever values to achieve a target revenue.

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "optimization_id": "opt-123abc",
  "studio_id": "STU001",
  "target_revenue": 35000.0,
  "achievable_revenue": 34250.50,
  "achievement_rate": 0.98,
  "recommended_levers": {
    "retention_rate": 0.75,
    "avg_ticket_price": 155.0,
    "class_attendance_rate": 0.72,
    "new_members": 28,
    "staff_utilization_rate": 0.85,
    "upsell_rate": 0.27,
    "total_classes_held": 115,
    "total_members": 195
  },
  "lever_changes": [
    {
      "lever_name": "retention_rate",
      "current_value": 0.70,
      "recommended_value": 0.75,
      "change_absolute": 0.05,
      "change_percentage": 7.14,
      "priority": 1
    },
    {
      "lever_name": "avg_ticket_price",
      "current_value": 140.0,
      "recommended_value": 155.0,
      "change_absolute": 15.0,
      "change_percentage": 10.71,
      "priority": 2
    }
  ],
  "action_plan": [
    {
      "priority": 1,
      "lever": "retention_rate",
      "action": "Improve member retention through enhanced engagement programs and personalized services",
      "expected_impact": 2500.0,
      "timeline_weeks": 8
    },
    {
      "priority": 2,
      "lever": "avg_ticket_price",
      "action": "Implement strategic pricing increase with value-added services",
      "expected_impact": 2000.0,
      "timeline_weeks": 4
    }
  ],
  "confidence_score": 0.75,
  "model_version": "2.2.0",
  "timestamp": "2025-11-05T23:45:00"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/inverse" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

---

### Partial Lever Prediction (NEW)

#### POST `/api/v1/predict/partial`

Predict unknown lever values based on known levers.

**Use Cases:**
- "If I know retention and price, what attendance rate should I expect?"
- "Given my current members and classes, what revenue can I achieve?"
- "What new member count do I need for my target?"

**Request Body:**
```json
{
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
```

**Supported Output Levers:**
- All 8 primary levers
- `total_revenue` (special case)

**Response:**
```json
{
  "prediction_id": "pred-xyz789",
  "studio_id": "STU001",
  "input_levers": {
    "retention_rate": 0.75,
    "avg_ticket_price": 150.0,
    "total_members": 200
  },
  "predicted_levers": [
    {
      "lever_name": "class_attendance_rate",
      "predicted_value": 0.72,
      "confidence_score": 0.65,
      "value_range": [0.68, 0.76]
    },
    {
      "lever_name": "new_members",
      "predicted_value": 23.5,
      "confidence_score": 0.60,
      "value_range": [22, 25]
    },
    {
      "lever_name": "total_revenue",
      "predicted_value": 32150.75,
      "confidence_score": 0.85,
      "value_range": [28935.68, 35365.83]
    }
  ],
  "overall_confidence": 0.70,
  "model_version": "2.2.0",
  "timestamp": "2025-11-05T23:45:00",
  "notes": "Predictions based on historical patterns and model relationships"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/partial" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

---

### Studio Management

#### GET `/api/v1/studios`

List all available studios in the system.

**Response:**
```json
{
  "studios": [
    {
      "studio_id": "STU001",
      "months_of_data": 36,
      "avg_revenue": 31500.50,
      "avg_members": 195,
      "avg_retention": 0.76,
      "latest_revenue": 33200.75,
      "latest_members": 205
    }
  ],
  "count": 14,
  "showing": 14
}
```

#### GET `/api/v1/studios/{studio_id}`

Get detailed information for a specific studio.

**Response:**
```json
{
  "studio_id": "STU001",
  "months_of_data": 36,
  "avg_revenue": 31500.50,
  "avg_members": 195,
  "avg_retention": 0.76,
  "latest_revenue": 33200.75,
  "latest_members": 205
}
```

---

## Error Responses

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "levers", "retention_rate"],
      "msg": "ensure this value is less than or equal to 1.0",
      "type": "value_error.number.not_le"
    }
  ]
}
```

### 500 Server Error
```json
{
  "detail": "Prediction failed: <error message>"
}
```

---

## Interactive Documentation

Visit these URLs when the server is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Running the API

### Development Mode
```bash
python run_api.py
# or
python run_api.py development
```

### Production Mode
```bash
python run_api.py production
```

### With uvicorn directly
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Testing

Run the test suite:
```bash
pytest tests/test_api.py -v
```

---

## Model Information

- **Model Version**: 2.2.0
- **Model Type**: Ridge Regression (Multi-output)
- **Number of Features**: 15 selected features
- **Training Data**: Multi-studio data (~840 studio-months)
- **Expected Performance**: R² 0.40-0.55

**Selected Features:**
1. total_members
2. new_members
3. churned_members
4. total_revenue
5. membership_revenue
6. class_pack_revenue
7. retail_revenue
8. staff_count
9. estimated_ltv
10. prev_month_revenue
11. prev_month_members
12. 3m_avg_revenue
13. revenue_momentum
14. retention_x_ticket
15. staff_util_x_members

---

## Support

For issues or questions, please refer to the project documentation or contact the development team.

