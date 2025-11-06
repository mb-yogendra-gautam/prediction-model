# Studio Revenue Simulator API Documentation

## Overview

The Studio Revenue Simulator API provides machine learning-powered predictions for fitness studio revenue optimization. It offers the following capabilities:

1. **Forward Prediction**: Predict future revenue based on business lever adjustments
2. **Inverse Optimization**: Find optimal lever values to achieve target revenue
3. **Partial Lever Prediction**: Predict unknown levers based on known values
4. **Scenario Comparison**: Compare multiple optimization strategies side-by-side
5. **Lever Management**: Get metadata for all business levers with constraints and implementation guidance
6. **AI-Powered Insights**: Get strategic business recommendations using GPT-4 (optional)
7. **Model Explainability**: Understand prediction drivers using SHAP analysis

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

**Query Parameters:**
- `include_ai_insights` (boolean, optional): Set to `true` to generate AI-powered business insights using GPT-4. Requires OpenAI API key configured. Default: `false`

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
  "timestamp": "2025-11-05T23:45:00",
  "explanation": {
    "top_features": [
      {
        "feature": "total_members",
        "impact": 0.45,
        "description": "Current member base has the strongest positive impact"
      },
      {
        "feature": "avg_ticket_price",
        "impact": 0.32,
        "description": "Pricing strategy significantly influences revenue"
      },
      {
        "feature": "retention_rate",
        "impact": 0.28,
        "description": "Member retention is a key revenue driver"
      }
    ],
    "method": "SHAP (SHapley Additive exPlanations)"
  },
  "quick_wins": [
    {
      "lever": "class_attendance_rate",
      "current_value": 0.70,
      "recommended_value": 0.75,
      "expected_impact": 1250.50,
      "difficulty": "low",
      "action": "Implement automated class reminders and incentivize regular attendance"
    },
    {
      "lever": "upsell_rate",
      "current_value": 0.25,
      "recommended_value": 0.30,
      "expected_impact": 980.25,
      "difficulty": "medium",
      "action": "Train staff on upselling techniques and create attractive package bundles"
    }
  ],
  "ai_insights": {
    "executive_summary": "Based on your current levers, the studio is projected to generate $99.5K over the next 3 months with steady growth. Your strong retention rate of 75% provides a solid foundation, but there's opportunity to optimize class attendance and upselling.",
    "key_drivers": [
      "Strong member base of 200 provides consistent revenue foundation",
      "Above-average ticket price of $150 positions studio in premium segment",
      "Retention rate at 75% is healthy but has room for improvement"
    ],
    "recommendations": [
      "Increase class attendance from 70% to 75% through better scheduling and member engagement",
      "Implement upselling training program to capture additional revenue per member",
      "Consider modest price increases given strong retention metrics"
    ],
    "risks": [
      "Confidence decreases over time - monitor actual results closely",
      "Class attendance below optimal level may indicate scheduling or engagement issues",
      "New member acquisition rate should be maintained to offset natural churn"
    ],
    "confidence_explanation": "High confidence (85%) for month 1 based on strong historical data. Confidence gradually decreases to 75% by month 3 due to increased uncertainty in future projections."
  }
}
```

**cURL Example (Basic):**
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

**cURL Example (With AI Insights):**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/forward?include_ai_insights=true" \
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

**Query Parameters:**
- `include_ai_insights` (boolean, optional): Set to `true` to generate AI-powered strategic insights using GPT-4. Requires OpenAI API key configured. Default: `false`

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
  "timestamp": "2025-11-05T23:45:00",
  "ai_insights": {
    "executive_summary": "To reach your $35K revenue target, focus on retention and pricing. The recommended strategy can achieve 98% of your goal ($34,250) through realistic adjustments that balance member satisfaction with revenue growth.",
    "key_drivers": [
      "Retention improvement from 70% to 75% is your highest-impact lever",
      "Strategic price increase of $15 (10.7%) aligns with premium positioning",
      "Member growth to 195 provides sustainable revenue foundation"
    ],
    "recommendations": [
      "Launch member engagement initiatives to improve retention - highest priority",
      "Implement value-added services to justify price increase",
      "Focus on quality over quantity in new member acquisition",
      "Monitor member satisfaction closely during transition period"
    ],
    "risks": [
      "Price increase may impact retention if not communicated properly",
      "Simultaneous changes to multiple levers require careful change management",
      "Achievement rate of 98% means target may not be fully met within constraints"
    ],
    "confidence_explanation": "Moderate confidence (75%) reflects the complexity of simultaneous lever adjustments. Historical data supports these changes individually, but combined impact has higher uncertainty."
  }
}
```

**cURL Example (Basic):**
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

**cURL Example (With AI Insights):**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/inverse?include_ai_insights=true" \
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

**Query Parameters:**
- `include_ai_insights` (boolean, optional): Set to `true` to generate AI-powered insights about predicted levers using GPT-4. Requires OpenAI API key configured. Default: `false`

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
  "notes": "Predictions based on historical patterns and model relationships",
  "ai_insights": {
    "executive_summary": "Given your retention rate of 75% and average ticket price of $150 for 200 members, the model predicts healthy operational metrics with expected revenue around $32K monthly.",
    "key_drivers": [
      "Your strong retention rate of 75% supports stable member base",
      "Premium pricing at $150 positions studio well for revenue growth",
      "Predicted attendance rate of 72% is above industry average"
    ],
    "recommendations": [
      "With predicted 23-25 new members monthly, focus on consistent acquisition channels",
      "Attendance rate of 72% suggests good member engagement - maintain current scheduling",
      "Revenue prediction of $32K provides solid foundation for growth initiatives"
    ],
    "risks": [
      "Lower confidence on new member predictions (60%) - monitor acquisition closely",
      "Revenue confidence is strong (85%), but always validate with actual data",
      "Predicted values assume current market conditions remain stable"
    ],
    "confidence_explanation": "Revenue prediction has high confidence (85%) due to strong correlation with member count and pricing. New member predictions have moderate confidence (60%) as acquisition is more variable."
  }
}
```

**cURL Example (Basic):**
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

**cURL Example (With AI Insights):**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/partial?include_ai_insights=true" \
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

### Scenario Comparison

#### POST `/api/v1/predict/inverse/compare-scenarios`

Compare multiple optimization strategies side-by-side to achieve a target revenue goal.

**Use Case:**
This endpoint allows you to evaluate different approaches to reaching your revenue target:
- **Conservative**: Minimal changes, focus on easy wins
- **Balanced**: Mix of difficulty, optimize for revenue only
- **Aggressive**: Allow larger changes, multi-objective optimization
- **Growth-focused**: Prioritize member growth and retention

**Request Body:**
```json
{
  "studio_id": "STU001",
  "target_revenue": 35000,
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
  "scenarios": [
    {
      "name": "Conservative",
      "constraints": {
        "max_retention_increase": 0.03,
        "max_ticket_increase": 10
      },
      "objectives": ["revenue"],
      "method": "auto"
    },
    {
      "name": "Aggressive",
      "constraints": {
        "max_retention_increase": 0.10,
        "max_ticket_increase": 50
      },
      "objectives": ["revenue", "retention", "growth"],
      "method": "ensemble"
    }
  ]
}
```

**Scenario Parameters:**
- `name`: Display name for the scenario
- `constraints`: Limits on lever changes (optional)
  - `max_retention_increase`: Maximum increase in retention rate
  - `max_ticket_increase`: Maximum increase in ticket price ($)
  - `max_new_members_increase`: Maximum increase in new member count
- `objectives`: List of optimization goals
  - `revenue`: Maximize revenue (always included)
  - `retention`: Improve member retention
  - `growth`: Increase member count
- `method`: Optimization algorithm
  - `auto`: Automatic selection
  - `gradient`: Gradient-based optimization
  - `ensemble`: Multiple algorithms combined

**Response:**
```json
{
  "studio_id": "STU001",
  "target_revenue": 35000,
  "current_state": {
    "retention_rate": 0.70,
    "avg_ticket_price": 140.0,
    "total_members": 180
  },
  "scenarios": [
    {
      "name": "Conservative",
      "achievable_revenue": 33200.50,
      "achievement_rate": 0.95,
      "confidence_score": 0.82,
      "recommended_levers": {
        "retention_rate": 0.73,
        "avg_ticket_price": 150.0,
        "class_attendance_rate": 0.68,
        "new_members": 22,
        "staff_utilization_rate": 0.82,
        "upsell_rate": 0.23,
        "total_classes_held": 105,
        "total_members": 185
      },
      "key_changes": [
        {
          "lever": "avg_ticket_price",
          "change": "+$10",
          "difficulty": "medium"
        },
        {
          "lever": "retention_rate",
          "change": "+3%",
          "difficulty": "low"
        }
      ],
      "overall_score": 8.5
    },
    {
      "name": "Aggressive",
      "achievable_revenue": 35800.25,
      "achievement_rate": 1.02,
      "confidence_score": 0.68,
      "recommended_levers": {
        "retention_rate": 0.80,
        "avg_ticket_price": 180.0,
        "class_attendance_rate": 0.75,
        "new_members": 30,
        "staff_utilization_rate": 0.90,
        "upsell_rate": 0.35,
        "total_classes_held": 130,
        "total_members": 210
      },
      "key_changes": [
        {
          "lever": "avg_ticket_price",
          "change": "+$40",
          "difficulty": "high"
        },
        {
          "lever": "retention_rate",
          "change": "+10%",
          "difficulty": "high"
        }
      ],
      "overall_score": 7.2
    }
  ],
  "recommended_scenario": "Conservative",
  "recommendation_reason": "Best balance of achievement rate (95%), confidence (82%), and implementation difficulty. Lower risk profile while still reaching near-target revenue.",
  "comparison_summary": {
    "highest_revenue": "Aggressive",
    "highest_confidence": "Conservative",
    "easiest_implementation": "Conservative",
    "best_overall_score": "Conservative"
  }
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/inverse/compare-scenarios" \
  -H "Content-Type: application/json" \
  -d '{
    "studio_id": "STU001",
    "target_revenue": 35000,
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
    "scenarios": [
      {
        "name": "Conservative",
        "constraints": {"max_retention_increase": 0.03, "max_ticket_increase": 10},
        "objectives": ["revenue"],
        "method": "auto"
      },
      {
        "name": "Aggressive",
        "constraints": {"max_retention_increase": 0.10, "max_ticket_increase": 50},
        "objectives": ["revenue", "retention", "growth"],
        "method": "ensemble"
      }
    ]
  }'
```

---

### Lever Management

#### GET `/api/v1/predict/levers`

Get metadata for all available business levers.

**Use Case:**
This endpoint provides comprehensive information about all 8 business levers that can be used in predictions and optimizations. Ideal for:
- Dynamic form generation in frontend applications
- Input validation
- Understanding lever constraints and priorities
- Implementation planning

**Query Parameters:**
- `include_details` (boolean, optional): Set to `true` to include feasibility thresholds and action templates for implementation guidance. Default: `false`

**Response (Basic):**
```json
{
  "levers": [
    {
      "name": "retention_rate",
      "display_name": "Member Retention Rate",
      "data_type": "float",
      "constraints": {
        "min": 0.5,
        "max": 1.0
      },
      "description": "Percentage of members who remain active from one month to the next",
      "unit": "percentage",
      "default_value": 0.75,
      "priority": 1
    },
    {
      "name": "avg_ticket_price",
      "display_name": "Average Ticket Price",
      "data_type": "float",
      "constraints": {
        "min": 50.0,
        "max": 500.0
      },
      "description": "Average monthly membership or package price per customer",
      "unit": "dollars",
      "default_value": 150.0,
      "priority": 2
    },
    {
      "name": "new_members",
      "display_name": "New Members per Month",
      "data_type": "integer",
      "constraints": {
        "min": 0,
        "max": 100
      },
      "description": "Number of new members acquired per month",
      "unit": "count",
      "default_value": 25,
      "priority": 3
    }
    // ... 5 more levers
  ],
  "count": 8,
  "version": "1.0.0"
}
```

**Response (With Details - `?include_details=true`):**
```json
{
  "levers": [
    {
      "name": "retention_rate",
      "display_name": "Member Retention Rate",
      "data_type": "float",
      "constraints": {
        "min": 0.5,
        "max": 1.0
      },
      "description": "Percentage of members who remain active from one month to the next",
      "unit": "percentage",
      "default_value": 0.75,
      "priority": 1,
      "feasibility_thresholds": {
        "easy": 0.02,
        "moderate": 0.05,
        "hard": 0.10,
        "very_hard": 0.10
      },
      "action_template": {
        "description": "Improve member retention through enhanced engagement programs and personalized services",
        "timeline_weeks": 8,
        "department": "Member Success",
        "resources": "Engagement software, customer success team"
      }
    }
    // ... 7 more levers with full details
  ],
  "count": 8,
  "version": "1.0.0"
}
```

**Lever Priority Ranking:**
1. **retention_rate** - Highest impact on revenue
2. **avg_ticket_price** - Direct revenue driver
3. **new_members** - Growth engine
4. **upsell_rate** - Revenue per member optimization
5. **class_attendance_rate** - Engagement metric
6. **staff_utilization_rate** - Operational efficiency
7. **total_classes_held** - Capacity management
8. **total_members** - Base metric

**Feasibility Thresholds:**
When `include_details=true`, each lever includes difficulty ratings for implementation:
- **easy**: Small changes that are simple to implement
- **moderate**: Medium changes requiring some effort
- **hard**: Large changes requiring significant resources
- **very_hard**: Very large changes with high complexity

**cURL Example (Basic):**
```bash
curl -X GET "http://localhost:8000/api/v1/predict/levers"
```

**cURL Example (With Details):**
```bash
curl -X GET "http://localhost:8000/api/v1/predict/levers?include_details=true"
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

### Core Model
- **Model Version**: 2.2.0
- **Model Type**: Ridge Regression (Multi-output)
- **Number of Features**: 15 selected features
- **Training Data**: Multi-studio data (~840 studio-months)
- **Expected Performance**: R² 0.40-0.55

### AI & Explainability Features
- **AI Insights Engine**: GPT-4 (gpt-4o model)
  - Executive summaries in business language
  - Strategic recommendations
  - Risk identification
  - Confidence explanations
- **Explainability**: SHAP (SHapley Additive exPlanations)
  - Feature importance analysis
  - Prediction driver identification
  - Impact quantification
- **Quick Wins**: Automated opportunity detection
  - Low-difficulty, high-impact recommendations
  - Expected impact calculations
  - Actionable implementation guidance

### Selected Features
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

### Configuration Requirements
- **Optional - AI Insights**: Requires `OPENAI_API_KEY` environment variable
  - Set in `.env` file or environment
  - Only needed when `include_ai_insights=true` query parameter is used
  - API will function normally without it (AI insights will be null)

---

## Advanced Features

### AI-Powered Insights

The API integrates with OpenAI's GPT-4 to provide human-readable business insights alongside numerical predictions. This feature is **optional** and requires an OpenAI API key.

**How to Enable:**
1. Set `OPENAI_API_KEY` environment variable or add to `.env` file
2. Add `?include_ai_insights=true` query parameter to any prediction endpoint
3. Response will include an `ai_insights` object with strategic recommendations

**What You Get:**
- **Executive Summary**: High-level business interpretation of predictions
- **Key Drivers**: Main factors influencing the results
- **Recommendations**: Actionable steps to improve performance
- **Risks**: Potential concerns and challenges to monitor
- **Confidence Explanation**: Plain-language explanation of prediction reliability

**Benefits:**
- Translate ML predictions into actionable business strategy
- Identify opportunities and risks automatically
- Get context-aware recommendations based on your specific situation
- Understand confidence scores in business terms

**Example Use Cases:**
- Present predictions to non-technical stakeholders
- Generate executive summaries for reports
- Get strategic guidance on optimization decisions
- Identify implementation risks before committing to changes

### Model Explainability (SHAP)

All forward predictions automatically include SHAP-based explainability showing which features have the strongest impact on predictions.

**Included in Response:**
```json
"explanation": {
  "top_features": [
    {
      "feature": "total_members",
      "impact": 0.45,
      "description": "Current member base has the strongest positive impact"
    }
  ],
  "method": "SHAP (SHapley Additive exPlanations)"
}
```

**Benefits:**
- Understand why predictions are what they are
- Identify which levers have the most influence
- Build trust in model recommendations
- Debug unexpected predictions

### Quick Wins

Forward predictions include automated "quick win" recommendations - high-impact, low-difficulty changes you can implement quickly.

**Included in Response:**
```json
"quick_wins": [
  {
    "lever": "class_attendance_rate",
    "current_value": 0.70,
    "recommended_value": 0.75,
    "expected_impact": 1250.50,
    "difficulty": "low",
    "action": "Implement automated class reminders and incentivize regular attendance"
  }
]
```

**Benefits:**
- Identify easy improvements immediately
- Prioritize actions by difficulty and impact
- Get specific implementation suggestions
- Calculate expected ROI before making changes

---

## Support

For issues or questions, please refer to the project documentation or contact the development team.

---

## API Version History

### Version 1.0.0 (Current)

**Release Date**: November 2025

**New Features:**
- AI-Powered Insights using GPT-4 (optional)
  - Executive summaries in business language
  - Strategic recommendations and risk identification
  - Available on all prediction endpoints via `include_ai_insights` query parameter
- Model Explainability using SHAP
  - Automatic feature importance analysis
  - Prediction driver identification
  - Included in all forward predictions
- Quick Wins recommendations
  - Automated opportunity detection
  - Low-difficulty, high-impact suggestions
  - ROI calculations
- Scenario Comparison endpoint (`/api/v1/predict/inverse/compare-scenarios`)
  - Compare multiple optimization strategies
  - Side-by-side evaluation with scoring
  - Automated recommendation selection

**Core Capabilities:**
- Forward prediction for 1-3 months
- Inverse optimization with constraints
- Partial lever prediction
- Studio management endpoints
- Interactive documentation (Swagger/ReDoc)

**Model:**
- Version 2.2.0
- Ridge Regression with 15 selected features
- Trained on ~840 studio-months of data
- R² performance: 0.40-0.55

**Technical Stack:**
- FastAPI framework
- OpenAI GPT-4 (gpt-4o) for AI insights
- SHAP for model explainability
- LangChain for AI orchestration
- uvicorn server

