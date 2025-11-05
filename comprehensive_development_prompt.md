# Comprehensive Development Prompt: Fitness Studio Revenue Simulator Platform

## Project Overview

Build a business intelligence platform that enables fitness studio owners to:

1. **Forward Prediction**: Predict future revenue based on adjusting operational levers (retention rate, average ticket, class attendance, etc.)
2. **Inverse Prediction**: Determine optimal lever values needed to achieve a target revenue goal
3. **Actionable Insights**: Generate AI-powered recommendations based on predicted scenarios

---

## Part 1: Data Generation & Schema Design

### 1.1 Sample Data Requirements

Generate **5 years of synthetic historical data** for a fitness studio with the following specifications:

#### Primary Tables:

**Table: `studios`**

- `studio_id` (Primary Key)
- `studio_name` (string)
- `studio_type` (enum: Yoga, Pilates, CrossFit, Wellness, Boutique Fitness)
- `location` (string: city, state)
- `opening_date` (date)
- `square_footage` (integer)
- `max_capacity` (integer)

**Table: `monthly_metrics`**

- `metric_id` (Primary Key)
- `studio_id` (Foreign Key)
- `month_year` (date, first day of month)
- `total_members` (integer: 100-500 range, with seasonal variation)
- `new_members` (integer: 10-50 per month)
- `churned_members` (integer: 5-40 per month)
- `retention_rate` (float: 60-95%, calculated)
- `avg_ticket_price` (float: $80-$300, with gradual increases)
- `total_classes_held` (integer: 100-400 per month)
- `total_class_attendance` (integer)
- `class_attendance_rate` (float: 50-90%)
- `staff_count` (integer: 5-25)
- `staff_utilization_rate` (float: 60-95%)
- `total_revenue` (float: calculated from components)
- `membership_revenue` (float)
- `class_pack_revenue` (float)
- `retail_revenue` (float)
- `intro_offer_conversions` (integer: 5-30)
- `upsell_rate` (float: 10-40%)
- `average_member_lifetime_months` (float: 6-24)

**Table: `class_types`**

- `class_type_id` (Primary Key)
- `studio_id` (Foreign Key)
- `class_name` (string: Yoga Flow, HIIT, Pilates, etc.)
- `class_price` (float)
- `max_capacity` (integer: 10-30)
- `avg_attendance_rate` (float)

**Table: `member_cohorts`**

- `cohort_id` (Primary Key)
- `studio_id` (Foreign Key)
- `join_month` (date)
- `cohort_size` (integer)
- `month_1_retention` (float)
- `month_3_retention` (float)
- `month_6_retention` (float)
- `month_12_retention` (float)
- `avg_ltv` (float)

**Table: `scenarios`** (for saving simulations)

- `scenario_id` (Primary Key)
- `studio_id` (Foreign Key)
- `scenario_name` (string)
- `created_at` (timestamp)
- `base_month` (date)
- `projection_months` (integer: default 3)
- `lever_inputs` (JSON)
- `predicted_outputs` (JSON)
- `confidence_score` (float)

### 1.2 Data Generation Rules

**Realistic Patterns to Implement:**

1. **Seasonality**:

   - January surge (New Year resolutions): +20-30% new members
   - Summer dip (June-August): -10-15% attendance
   - Fall recovery (September): +15% new members
   - Holiday decline (December): -5-10% attendance

2. **Growth Trends**:

   - Year 1-2: Rapid growth phase (10-15% quarterly revenue growth)
   - Year 3-4: Stabilization (3-7% growth)
   - Year 5: Mature business (1-4% growth)

3. **Correlations** (simulate realistic relationships):

   - Higher retention rate → Lower new member acquisition cost needed
   - Higher class attendance → Higher upsell opportunities (+0.3 correlation)
   - Higher staff count → Better member retention (+0.4 correlation)
   - Higher avg ticket → Slightly lower retention (-0.2 correlation, premium pricing effect)

4. **Noise & Variance**:

   - Add 5-10% random noise to prevent overfitting
   - Include 2-3 outlier months (COVID impact, facility closure, major promotion)

5. **Revenue Calculation Formula**:
   ```
   total_revenue = (
       (total_members * avg_ticket_price) +  # Membership base
       (total_class_attendance * class_drop_in_rate * 15) +  # Drop-ins
       (new_members * intro_offer_price * intro_conversion_rate) +  # Intro offers
       (total_members * retail_revenue_per_member * 0.2) +  # Retail
       (total_members * upsell_rate * avg_upsell_value)  # Upsells
   )
   ```

### 1.3 Data Split Strategy

- **Training Set**: First 48 months (Years 1-4) - 80%
- **Validation Set**: Months 49-54 (6 months) - 10%
- **Test Set**: Months 55-60 (Final 6 months) - 10%

Generate data with clear train/val/test labels for model evaluation.

---

## Part 2: Machine Learning Model Architecture

### 2.1 Forward Prediction Model (Levers → Revenue)

**Problem Type**: Multi-output Regression

**Input Features (Levers)**:

1. `retention_rate` (percentage)
2. `avg_ticket_price` (dollars)
3. `class_attendance_rate` (percentage)
4. `new_members_monthly` (count)
5. `staff_utilization_rate` (percentage)
6. `upsell_rate` (percentage)
7. `total_classes_held` (count)
8. `current_member_count` (count)
9. `month_index` (1-12, for seasonality)
10. `year_index` (1-5, for trend)
11. `prev_month_revenue` (lagged feature)
12. `3_month_avg_retention` (rolling average)

**Output Targets**:

1. `predicted_revenue_month_1` (primary target)
2. `predicted_revenue_month_2`
3. `predicted_revenue_month_3`
4. `predicted_member_count_month_3`
5. `predicted_retention_rate_month_3`

**Model Architecture Options**:

**Option A: Gradient Boosting Ensemble (Recommended for MVP)**

- Use XGBoost or LightGBM
- Advantages: Handles non-linear relationships, fast training, interpretable feature importance
- Hyperparameters to tune:
  - `n_estimators`: 100-500
  - `max_depth`: 3-7
  - `learning_rate`: 0.01-0.1
  - `min_child_weight`: 1-5

**Option B: Neural Network (for better non-linearity)**

```python
Architecture:
- Input Layer: 12 features
- Hidden Layer 1: 64 neurons, ReLU activation, Dropout(0.2)
- Hidden Layer 2: 32 neurons, ReLU activation, Dropout(0.2)
- Hidden Layer 3: 16 neurons, ReLU activation
- Output Layer: 5 targets, Linear activation
- Loss: Mean Squared Error
- Optimizer: Adam (lr=0.001)
```

**Option C: Ensemble Approach (Best Accuracy)**

- Combine XGBoost + Random Forest + Linear Regression
- Use weighted averaging or stacking meta-model
- Weights based on validation performance

### 2.2 Inverse Prediction Model (Target Revenue → Levers)

**Problem Type**: Constrained Optimization / Inverse Regression

**Approach 1: Optimization-Based Method (Recommended)**

Use the trained forward model as the objective function and apply constrained optimization:

```python
from scipy.optimize import minimize

def objective(levers, target_revenue, forward_model):
    """Minimize difference between predicted and target revenue"""
    predicted_revenue = forward_model.predict([levers])[0]
    return abs(predicted_revenue - target_revenue)

def optimize_levers(target_revenue, current_state, forward_model):
    # Define constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - 0.5},  # retention >= 50%
        {'type': 'ineq', 'fun': lambda x: 1.0 - x[0]},  # retention <= 100%
        {'type': 'ineq', 'fun': lambda x: x[1] - 50},   # avg_ticket >= $50
        {'type': 'ineq', 'fun': lambda x: 500 - x[1]},  # avg_ticket <= $500
        # Add constraints for all levers...
    ]

    # Bounds for each lever
    bounds = [
        (0.5, 1.0),    # retention_rate
        (50, 500),     # avg_ticket_price
        (0.4, 1.0),    # class_attendance_rate
        (0, 100),      # new_members_monthly
        (0.6, 1.0),    # staff_utilization_rate
        (0.0, 0.5),    # upsell_rate
    ]

    # Initial guess (current state)
    x0 = current_state

    # Optimize
    result = minimize(
        objective,
        x0,
        args=(target_revenue, forward_model),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x  # Optimal lever values
```

**Approach 2: Multi-Objective Optimization**

When optimizing levers, prioritize:

1. **Feasibility**: Changes should be achievable (e.g., max 5% monthly retention improvement)
2. **Cost**: Prioritize low-cost levers (retention > new acquisition)
3. **Time**: Faster-to-implement changes weighted higher

**Approach 3: Reinforcement Learning (Advanced)**

- Train an RL agent to learn optimal lever adjustments
- Reward function: achieving target revenue with minimal lever changes
- Use PPO or A3C algorithm

### 2.3 Feature Engineering

**Derived Features**:

1. `retention_rate_momentum`: Change in retention over last 3 months
2. `revenue_growth_rate`: MoM revenue change percentage
3. `member_churn_rate`: 1 - retention_rate
4. `revenue_per_member`: total_revenue / total_members
5. `class_utilization`: total_class_attendance / (total_classes \* max_capacity)
6. `seasonal_index`: Cyclical encoding of month (sin/cos transformation)
7. `is_january`: Binary flag for New Year boost
8. `is_summer`: Binary flag for summer dip
9. `ltv_estimate`: avg_ticket_price _ retention_rate _ 12
10. `staff_per_member`: staff_count / total_members

**Interaction Features**:

1. `retention_x_ticket`: retention_rate \* avg_ticket_price
2. `attendance_x_classes`: class_attendance_rate \* total_classes_held
3. `upsell_x_members`: upsell_rate \* current_member_count

### 2.4 Model Evaluation Metrics

**For Forward Prediction**:

1. **RMSE** (Root Mean Squared Error): Primary metric for revenue prediction
2. **MAPE** (Mean Absolute Percentage Error): For interpretability
3. **R² Score**: Goodness of fit (target: >0.85)
4. **MAE** (Mean Absolute Error): Robustness to outliers
5. **Directional Accuracy**: % of times predicted direction matches actual

**Custom Metric: Business Impact Score**

```python
def business_impact_score(y_true, y_pred):
    """
    Penalize errors more heavily when they lead to wrong decisions
    """
    error = y_pred - y_true
    percentage_error = (error / y_true) * 100

    # Heavy penalty if error > 10%
    penalty = np.where(abs(percentage_error) > 10, 2.0, 1.0)

    return np.mean(abs(percentage_error) * penalty)
```

**For Inverse Prediction**:

1. **Revenue Achievement Rate**: % of times target revenue reached within 5% tolerance
2. **Lever Feasibility Score**: % of suggested levers within realistic bounds
3. **Cost Efficiency**: Prioritize lower-cost lever changes

---

## Part 3: Backend API Architecture

### 3.1 Technology Stack

**Core Backend**:

- **Framework**: FastAPI (Python 3.10+)
- **Database**: PostgreSQL 15+ with TimescaleDB extension (for time-series optimization)
- **ML Framework**: scikit-learn, XGBoost, LightGBM, TensorFlow/PyTorch (for NN)
- **API Documentation**: Automatic via FastAPI (Swagger/OpenAPI)

**Additional Services**:

- **Caching**: Redis (for prediction caching)
- **Model Versioning**: MLflow or DVC

**AI/LLM**:

- **OpenAI API** (GPT-4) for insights generation

### 3.2 API Endpoints Design

#### 3.2.1 Core Prediction Endpoints

**POST /api/v1/predict/forward**

```json
Request:
{
  "studio_id": "string",
  "base_month": "2024-01",
  "projection_months": 3,
  "levers": {
    "retention_rate": 0.75,
    "avg_ticket_price": 150.0,
    "class_attendance_rate": 0.70,
    "new_members_monthly": 25,
    "staff_utilization_rate": 0.85,
    "upsell_rate": 0.25,
    "total_classes_held": 120
  },
  "include_confidence_intervals": true
}

Response:
{
  "scenario_id": "uuid",
  "predictions": {
    "month_1": {
      "revenue": 45000.50,
      "member_count": 285,
      "confidence_interval": [42000, 48000],
      "confidence_score": 0.87
    },
    "month_2": {
      "revenue": 46500.25,
      "member_count": 292,
      "confidence_interval": [43200, 49800],
      "confidence_score": 0.84
    },
    "month_3": {
      "revenue": 48200.75,
      "member_count": 298,
      "confidence_interval": [44500, 51900],
      "confidence_score": 0.82
    }
  },
  "growth_rate": 0.086,
  "total_projected_revenue": 139701.50,
  "model_version": "v1.2.3",
  "prediction_accuracy": 0.82
}
```

**POST /api/v1/predict/inverse**

```json
Request:
{
  "studio_id": "string",
  "base_month": "2024-01",
  "target_revenue": 50000.0,
  "target_months": 3,
  "current_state": {
    "retention_rate": 0.70,
    "avg_ticket_price": 140.0,
    "class_attendance_rate": 0.65,
    "new_members_monthly": 20,
    "staff_utilization_rate": 0.80,
    "upsell_rate": 0.20,
    "total_classes_held": 100,
    "current_member_count": 250
  },
  "constraints": {
    "max_retention_increase": 0.05,
    "max_ticket_increase": 20.0,
    "max_new_members_increase": 10,
    "prioritize_low_cost_levers": true
  }
}

Response:
{
  "optimization_id": "uuid",
  "target_revenue": 50000.0,
  "achievable_revenue": 49800.0,
  "achievement_rate": 0.996,
  "recommended_levers": {
    "retention_rate": 0.75,
    "avg_ticket_price": 150.0,
    "class_attendance_rate": 0.72,
    "new_members_monthly": 25,
    "staff_utilization_rate": 0.85,
    "upsell_rate": 0.25,
    "total_classes_held": 115
  },
  "lever_changes": {
    "retention_rate": {
      "current": 0.70,
      "recommended": 0.75,
      "change_pct": 0.071,
      "impact_on_revenue": 2500.0,
      "feasibility_score": 0.85,
      "estimated_cost": "low"
    },
    "avg_ticket_price": {
      "current": 140.0,
      "recommended": 150.0,
      "change_pct": 0.071,
      "impact_on_revenue": 3000.0,
      "feasibility_score": 0.70,
      "estimated_cost": "medium"
    }
    // ... other levers
  },
  "action_plan": [
    {
      "priority": 1,
      "lever": "retention_rate",
      "action": "Implement automated re-engagement campaigns for at-risk members",
      "expected_impact": 2500.0,
      "timeline_weeks": 4
    },
    {
      "priority": 2,
      "lever": "upsell_rate",
      "action": "Launch personal training package promotion",
      "expected_impact": 1800.0,
      "timeline_weeks": 2
    }
  ],
  "confidence_score": 0.88
}
```

#### 3.2.2 Supporting Endpoints

**GET /api/v1/studio/{studio_id}/baseline**

```json
Response:
{
  "studio_id": "string",
  "baseline_period": {
    "start_month": "2023-06",
    "end_month": "2024-05"
  },
  "current_metrics": {
    "retention_rate": 0.75,
    "avg_ticket_price": 150.0,
    "class_attendance_rate": 0.70,
    "total_members": 285,
    "monthly_revenue": 42500.0
  },
  "historical_trends": {
    "revenue_growth_12m": 0.08,
    "retention_trend": "stable",
    "seasonality_pattern": "january_peak_summer_dip"
  },
  "industry_benchmarks": {
    "retention_rate_percentile": 65,
    "avg_ticket_percentile": 72,
    "attendance_percentile": 58
  }
}
```

**GET /api/v1/levers/impact-analysis**

```json
Request Parameters:
- studio_id: string
- analysis_type: "sensitivity" | "elasticity" | "correlation"

Response:
{
  "analysis_type": "sensitivity",
  "lever_impacts": [
    {
      "lever": "retention_rate",
      "impact_score": 0.85,
      "revenue_sensitivity": 3500.0,  // Revenue change per 1% lever change
      "rank": 1,
      "recommendation": "High impact - prioritize retention initiatives"
    },
    {
      "lever": "new_members_monthly",
      "impact_score": 0.78,
      "revenue_sensitivity": 2800.0,
      "rank": 2,
      "recommendation": "Medium-high impact - balance with retention efforts"
    }
    // ... other levers ranked by impact
  ],
  "correlation_matrix": {
    "retention_vs_revenue": 0.82,
    "ticket_vs_revenue": 0.71,
    "attendance_vs_revenue": 0.65
  }
}
```

**POST /api/v1/scenarios/save**

```json
Request:
{
  "studio_id": "string",
  "scenario_name": "Q2 Growth Plan",
  "description": "Conservative growth targeting 5% increase",
  "scenario_type": "forward" | "inverse",
  "inputs": { /* lever values or target */ },
  "outputs": { /* predictions */ },
  "tags": ["growth", "q2", "conservative"]
}

Response:
{
  "scenario_id": "uuid",
  "saved_at": "2024-01-15T10:30:00Z",
  "shareable_url": "https://app.studio.com/scenarios/abc123"
}
```

**GET /api/v1/scenarios/{studio_id}/list**

```json
Response:
{
  "scenarios": [
    {
      "scenario_id": "uuid",
      "scenario_name": "Q2 Growth Plan",
      "created_at": "2024-01-15T10:30:00Z",
      "scenario_type": "forward",
      "target_revenue": 50000.0,
      "predicted_revenue": 48500.0,
      "tags": ["growth", "q2"]
    }
  ],
  "total": 15,
  "page": 1
}
```

**POST /api/v1/insights/generate**

```json
Request:
{
  "scenario_id": "uuid",
  "insight_types": ["next_best_actions", "risk_factors", "opportunities"]
}

Response:
{
  "insights": {
    "next_best_actions": [
      {
        "action": "Reactivate 20% of lapsed members",
        "impact": "High ROI, low cost",
        "expected_revenue_gain": 3500.0,
        "confidence": 0.85,
        "timeline": "4-6 weeks"
      },
      {
        "action": "Launch add-on package (nutrition coaching)",
        "impact": "Medium ROI, medium cost",
        "expected_revenue_gain": 2200.0,
        "confidence": 0.72,
        "timeline": "2-4 weeks"
      }
    ],
    "risk_factors": [
      {
        "risk": "Summer attendance dip approaching",
        "probability": 0.75,
        "potential_revenue_impact": -4000.0,
        "mitigation": "Launch summer challenge program"
      }
    ],
    "opportunities": [
      {
        "opportunity": "High staff utilization indicates capacity for growth",
        "potential_revenue_gain": 5000.0,
        "actions": ["Increase class schedule by 10%", "Hire 1-2 additional instructors"]
      }
    ]
  },
  "generated_at": "2024-01-15T10:35:00Z",
  "model_version": "gpt-4"
}
```

#### 3.2.3 Model Management Endpoints

**POST /api/v1/models/retrain**

```json
Request:
{
  "studio_id": "string",
  "include_latest_months": 3,
  "retrain_strategy": "incremental" | "full",
  "notify_on_completion": true
}

Response:
{
  "job_id": "uuid",
  "status": "queued",
  "estimated_completion": "2024-01-15T11:00:00Z"
}
```

**GET /api/v1/models/performance**

```json
Response:
{
  "model_version": "v1.2.3",
  "deployed_at": "2024-01-10T00:00:00Z",
  "performance_metrics": {
    "rmse": 2847.32,
    "mape": 6.2,
    "r2_score": 0.87,
    "directional_accuracy": 0.91
  },
  "prediction_distribution": {
    "total_predictions": 1543,
    "avg_confidence": 0.84,
    "error_distribution": {
      "within_5_pct": 0.78,
      "within_10_pct": 0.93,
      "over_10_pct": 0.07
    }
  }
}
```

### 3.3 Backend Service Architecture

#### 3.3.1 Core Services

**1. Prediction Service** (`services/prediction_service.py`)

```python
class PredictionService:
    def __init__(self, model_registry, feature_store):
        self.forward_model = model_registry.load_model("forward_predictor")
        self.feature_store = feature_store
        self.optimizer = LeverOptimizer()

    async def predict_forward(
        self,
        studio_id: str,
        levers: LeverInputs,
        projection_months: int = 3
    ) -> PredictionResult:
        """
        Forward prediction: levers → revenue
        """
        # 1. Fetch historical context
        historical_data = await self.feature_store.get_recent_history(
            studio_id, months=12
        )

        # 2. Engineer features
        features = self.engineer_features(levers, historical_data)

        # 3. Make predictions for each month
        predictions = []
        for month in range(1, projection_months + 1):
            pred = self.forward_model.predict(features)
            predictions.append(pred)

            # Update features with predicted values for next month
            features = self.update_features_with_prediction(features, pred)

        # 4. Calculate confidence intervals
        confidence = self.calculate_confidence(predictions, historical_data)

        return PredictionResult(
            predictions=predictions,
            confidence_score=confidence,
            model_version=self.forward_model.version
        )

    async def predict_inverse(
        self,
        studio_id: str,
        target_revenue: float,
        current_state: CurrentState,
        constraints: OptimizationConstraints
    ) -> OptimizationResult:
        """
        Inverse prediction: target revenue → lever values
        """
        # 1. Define optimization problem
        def objective(levers):
            pred_result = self.forward_model.predict(
                self.engineer_features(levers, historical_data)
            )
            return abs(pred_result.revenue - target_revenue)

        # 2. Apply constraints
        bounds, constraints_list = self.build_constraints(
            current_state, constraints
        )

        # 3. Run optimization
        optimal_levers = self.optimizer.optimize(
            objective_fn=objective,
            initial_state=current_state,
            bounds=bounds,
            constraints=constraints_list
        )

        # 4. Validate feasibility
        feasibility = self.assess_feasibility(
            current_state, optimal_levers
        )

        # 5. Generate action plan
        action_plan = await self.generate_action_plan(
            current_state, optimal_levers, feasibility
        )

        return OptimizationResult(
            recommended_levers=optimal_levers,
            action_plan=action_plan,
            feasibility_score=feasibility,
            confidence_score=self.calculate_optimization_confidence()
        )
```

**2. Feature Engineering Service** (`services/feature_engineer.py`)

```python
class FeatureEngineer:
    def engineer_features(
        self,
        levers: LeverInputs,
        historical_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Transform raw lever inputs into model-ready features
        """
        features = {}

        # Direct lever features
        features['retention_rate'] = levers.retention_rate
        features['avg_ticket_price'] = levers.avg_ticket_price
        features['class_attendance_rate'] = levers.class_attendance_rate

        # Derived features
        features['revenue_per_member'] = (
            levers.avg_ticket_price * levers.retention_rate
        )

        # Historical context features
        features['prev_month_revenue'] = historical_data.iloc[-1]['revenue']
        features['3_month_avg_retention'] = (
            historical_data.tail(3)['retention_rate'].mean()
        )
        features['revenue_growth_momentum'] = self.calculate_momentum(
            historical_data['revenue']
        )

        # Seasonal features
        features['month_sin'] = np.sin(2 * np.pi * levers.month_index / 12)
        features['month_cos'] = np.cos(2 * np.pi * levers.month_index / 12)
        features['is_january'] = 1 if levers.month_index == 1 else 0
        features['is_summer'] = 1 if levers.month_index in [6,7,8] else 0

        # Interaction features
        features['retention_x_ticket'] = (
            features['retention_rate'] * features['avg_ticket_price']
        )

        return np.array(list(features.values())).reshape(1, -1)

    def calculate_momentum(self, series: pd.Series) -> float:
        """Calculate growth momentum using exponential moving average"""
        return series.ewm(span=3).mean().iloc[-1] / series.iloc[-1] - 1.0
```

**3. Optimization Service** (`services/optimizer.py`)

```python
class LeverOptimizer:
    def optimize(
        self,
        objective_fn: Callable,
        initial_state: CurrentState,
        bounds: List[Tuple[float, float]],
        constraints: List[Dict],
        strategy: str = "multi_objective"
    ) -> Dict[str, float]:
        """
        Optimize levers to achieve target with multi-objective approach
        """
        if strategy == "multi_objective":
            return self._optimize_multi_objective(
                objective_fn, initial_state, bounds, constraints
            )
        else:
            return self._optimize_single_objective(
                objective_fn, initial_state, bounds, constraints
            )

    def _optimize_multi_objective(self, ...):
        """
        Optimize for:
        1. Achieving target revenue
        2. Minimizing lever changes (feasibility)
        3. Minimizing cost of implementation
        """
        from scipy.optimize import minimize

        def combined_objective(levers):
            # Objective 1: Revenue gap
            revenue_gap = objective_fn(levers)

            # Objective 2: Lever change magnitude
            lever_changes = np.sum((levers - initial_state) ** 2)

            # Objective 3: Implementation cost
            cost = self.calculate_implementation_cost(
                initial_state, levers
            )

            # Weighted combination
            return (
                10.0 * revenue_gap +      # Primary: hit revenue
                1.0 * lever_changes +     # Secondary: minimize changes
                0.5 * cost               # Tertiary: minimize cost
            )

        result = minimize(
            combined_objective,
            x0=initial_state,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return self._format_result(result.x)

    def calculate_implementation_cost(
        self,
        current: np.ndarray,
        proposed: np.ndarray
    ) -> float:
        """
        Estimate relative cost of implementing lever changes
        """
        # Cost weights (higher = more expensive to change)
        cost_weights = {
            'retention_rate': 3.0,      # High cost (programs, staffing)
            'avg_ticket_price': 1.0,    # Low cost (pricing change)
            'new_members': 2.5,         # Medium-high cost (marketing)
            'staff_utilization': 2.0,   # Medium cost (scheduling)
            'upsell_rate': 1.5,         # Low-medium cost (training)
            'class_attendance': 2.0,    # Medium cost (engagement programs)
        }

        total_cost = 0.0
        for i, (curr, prop) in enumerate(zip(current, proposed)):
            change_magnitude = abs(prop - curr) / curr
            lever_name = self.get_lever_name(i)
            total_cost += change_magnitude * cost_weights[lever_name]

        return total_cost
```

**4. Insight Generation Service** (`services/insights_service.py`)

```python
class InsightsService:
    def __init__(self, openai_client):
        self.openai_client = openai_client

    async def generate_action_plan(
        self,
        current_state: CurrentState,
        recommended_levers: Dict[str, float],
        prediction_result: PredictionResult,
        studio_context: StudioContext
    ) -> List[ActionItem]:
        """
        Generate AI-powered next best actions
        """
        # 1. Calculate lever changes and impacts
        lever_changes = self.calculate_lever_changes(
            current_state, recommended_levers
        )

        # 2. Rank by impact and feasibility
        prioritized_levers = self.prioritize_levers(lever_changes)

        # 3. Map to actionable tactics
        actions = []
        for lever in prioritized_levers[:5]:  # Top 5 levers
            tactics = self.get_tactics_for_lever(
                lever.name, lever.change_magnitude, studio_context
            )
            actions.extend(tactics)

        # 4. Enhance with AI suggestions
        ai_enhanced_actions = await self.enhance_with_ai(
            actions, studio_context, prediction_result
        )

        return ai_enhanced_actions

    def get_tactics_for_lever(
        self,
        lever_name: str,
        change_magnitude: float,
        studio_context: StudioContext
    ) -> List[ActionItem]:
        """
        Map lever changes to specific business tactics
        """
        tactics_map = {
            'retention_rate': [
                {
                    'action': 'Launch automated re-engagement campaign',
                    'target_improvement': 0.02,
                    'timeline_weeks': 4,
                    'cost': 'low',
                    'specifics': 'Email + SMS to at-risk members (no classes in 2 weeks)'
                },
                {
                    'action': 'Implement member check-in program',
                    'target_improvement': 0.03,
                    'timeline_weeks': 6,
                    'cost': 'medium',
                    'specifics': 'Monthly 1-on-1 with staff to assess satisfaction'
                },
                {
                    'action': 'Create member loyalty rewards program',
                    'target_improvement': 0.04,
                    'timeline_weeks': 8,
                    'cost': 'medium',
                    'specifics': 'Points system for referrals and milestones'
                }
            ],
            'avg_ticket_price': [
                {
                    'action': 'Introduce premium membership tier',
                    'target_improvement': 15.0,  # $15 increase
                    'timeline_weeks': 3,
                    'cost': 'low',
                    'specifics': 'Add unlimited classes + 1 PT session/month'
                },
                {
                    'action': 'Implement dynamic pricing for peak hours',
                    'target_improvement': 8.0,
                    'timeline_weeks': 2,
                    'cost': 'low',
                    'specifics': '+$5 for 6-8am and 5-7pm slots'
                }
            ],
            'class_attendance_rate': [
                {
                    'action': 'Launch monthly class challenge',
                    'target_improvement': 0.05,
                    'timeline_weeks': 2,
                    'cost': 'low',
                    'specifics': 'Gamified attendance tracking with prizes'
                },
                {
                    'action': 'Optimize class schedule based on demand',
                    'target_improvement': 0.07,
                    'timeline_weeks': 3,
                    'cost': 'low',
                    'specifics': 'Move low-attendance classes to high-demand times'
                }
            ],
            'upsell_rate': [
                {
                    'action': 'Train staff on consultative selling',
                    'target_improvement': 0.05,
                    'timeline_weeks': 4,
                    'cost': 'medium',
                    'specifics': 'Weekly sales training + role-playing sessions'
                },
                {
                    'action': 'Launch add-on package promotion',
                    'target_improvement': 0.08,
                    'timeline_weeks': 2,
                    'cost': 'low',
                    'specifics': 'Bundle PT + nutrition for limited time'
                }
            ]
        }

        # Select tactics that match required change magnitude
        relevant_tactics = []
        for tactic in tactics_map.get(lever_name, []):
            if tactic['target_improvement'] >= change_magnitude * 0.8:
                relevant_tactics.append(tactic)

        return relevant_tactics[:2]  # Top 2 tactics per lever

    async def enhance_with_ai(
        self,
        actions: List[ActionItem],
        studio_context: StudioContext,
        prediction_result: PredictionResult
    ) -> List[ActionItem]:
        """
        Use LLM to personalize and enhance action recommendations
        """
        prompt = f"""
        You are a business consultant for a {studio_context.studio_type} studio.

        Studio Context:
        - Location: {studio_context.location}
        - Current Members: {studio_context.current_members}
        - Current Revenue: ${studio_context.current_revenue}
        - Target Revenue: ${prediction_result.target_revenue}
        - Current Challenges: {studio_context.challenges}

        Proposed Actions:
        {json.dumps([a.dict() for a in actions], indent=2)}

        Please:
        1. Rank these actions by priority (highest impact + feasibility first)
        2. Add specific implementation tips for this studio
        3. Identify potential obstacles and mitigation strategies
        4. Suggest 1-2 additional creative tactics specific to this studio type

        Format as JSON array with fields: priority, action, implementation_tips, obstacles, expected_impact
        """

        response = await self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        enhanced_actions = json.loads(response.choices[0].message.content)
        return enhanced_actions
```

### 3.4 Database Schema (PostgreSQL)

```sql
-- Enable TimescaleDB extension for time-series optimization
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Studios table
CREATE TABLE studios (
    studio_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    studio_name VARCHAR(255) NOT NULL,
    studio_type VARCHAR(50) NOT NULL,
    location VARCHAR(255),
    opening_date DATE,
    square_footage INTEGER,
    max_capacity INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Monthly metrics (main time-series table)
CREATE TABLE monthly_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    studio_id UUID REFERENCES studios(studio_id),
    month_year DATE NOT NULL,
    total_members INTEGER,
    new_members INTEGER,
    churned_members INTEGER,
    retention_rate FLOAT,
    avg_ticket_price FLOAT,
    total_classes_held INTEGER,
    total_class_attendance INTEGER,
    class_attendance_rate FLOAT,
    staff_count INTEGER,
    staff_utilization_rate FLOAT,
    total_revenue FLOAT,
    membership_revenue FLOAT,
    class_pack_revenue FLOAT,
    retail_revenue FLOAT,
    intro_offer_conversions INTEGER,
    upsell_rate FLOAT,
    average_member_lifetime_months FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Convert to TimescaleDB hypertable for optimized time-series queries
SELECT create_hypertable('monthly_metrics', 'month_year');

-- Create index for fast studio lookups
CREATE INDEX idx_monthly_metrics_studio_month ON monthly_metrics(studio_id, month_year DESC);

-- Scenarios table (for saved simulations)
CREATE TABLE scenarios (
    scenario_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    studio_id UUID REFERENCES studios(studio_id),
    scenario_name VARCHAR(255) NOT NULL,
    scenario_description TEXT,
    scenario_type VARCHAR(20) CHECK (scenario_type IN ('forward', 'inverse')),
    base_month DATE,
    projection_months INTEGER,
    lever_inputs JSONB,
    predicted_outputs JSONB,
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tags TEXT[]
);

CREATE INDEX idx_scenarios_studio ON scenarios(studio_id, created_at DESC);
CREATE INDEX idx_scenarios_tags ON scenarios USING GIN(tags);

-- Model performance tracking
CREATE TABLE model_performance (
    performance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50),
    deployed_at TIMESTAMP,
    evaluation_date DATE,
    rmse FLOAT,
    mape FLOAT,
    r2_score FLOAT,
    mae FLOAT,
    directional_accuracy FLOAT,
    total_predictions INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions log (for monitoring)
CREATE TABLE prediction_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    studio_id UUID REFERENCES studios(studio_id),
    prediction_type VARCHAR(20),
    model_version VARCHAR(50),
    input_levers JSONB,
    predicted_output JSONB,
    confidence_score FLOAT,
    actual_outcome FLOAT,  -- Filled in later for accuracy tracking
    prediction_error FLOAT,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_prediction_logs_studio ON prediction_logs(studio_id, predicted_at DESC);

-- Materialized view for fast baseline metrics
CREATE MATERIALIZED VIEW studio_baselines AS
SELECT
    studio_id,
    AVG(retention_rate) as avg_retention_rate,
    AVG(avg_ticket_price) as avg_ticket_price,
    AVG(class_attendance_rate) as avg_attendance_rate,
    AVG(total_revenue) as avg_monthly_revenue,
    STDDEV(total_revenue) as revenue_stddev,
    COUNT(*) as months_of_data
FROM monthly_metrics
WHERE month_year >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY studio_id;

CREATE UNIQUE INDEX ON studio_baselines(studio_id);

-- Refresh materialized view daily
CREATE OR REPLACE FUNCTION refresh_studio_baselines()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY studio_baselines;
END;
$$ LANGUAGE plpgsql;
```

### 3.5 Caching Strategy (Redis)

```python
# services/cache_service.py
import redis
import json
import hashlib
from typing import Optional

class CacheService:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour

    def cache_key(self, prefix: str, **params) -> str:
        """Generate consistent cache key from parameters"""
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        return f"{prefix}:{param_hash}"

    async def get_prediction(self, studio_id: str, levers: dict) -> Optional[dict]:
        """Retrieve cached prediction result"""
        key = self.cache_key("prediction", studio_id=studio_id, **levers)
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    async def set_prediction(
        self,
        studio_id: str,
        levers: dict,
        result: dict,
        ttl: int = None
    ):
        """Cache prediction result"""
        key = self.cache_key("prediction", studio_id=studio_id, **levers)
        self.redis.setex(
            key,
            ttl or self.default_ttl,
            json.dumps(result)
        )

    async def invalidate_studio_cache(self, studio_id: str):
        """Clear all cached predictions for a studio"""
        pattern = f"prediction:*{studio_id}*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
```

---

## Part 4: Model Training Pipeline

### 4.1 Training Script Structure

```python
# training/train_forward_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import joblib
import mlflow

class ModelTrainer:
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        self.models = {}

    def prepare_data(self):
        """Prepare features and targets with proper train/val/test split"""
        # Feature engineering
        features = self.engineer_features(self.data)

        # Multi-output targets
        targets = self.data[[
            'revenue_month_1',
            'revenue_month_2',
            'revenue_month_3',
            'member_count_month_3',
            'retention_rate_month_3'
        ]]

        # Time-based split
        split_idx_val = int(len(self.data) * 0.8)
        split_idx_test = int(len(self.data) * 0.9)

        X_train = features[:split_idx_val]
        y_train = targets[:split_idx_val]

        X_val = features[split_idx_val:split_idx_test]
        y_val = targets[split_idx_val:split_idx_test]

        X_test = features[split_idx_test:]
        y_test = targets[split_idx_test:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)

    def train_ensemble(self, X_train, y_train, X_val, y_val):
        """Train ensemble of models"""
        mlflow.start_run()

        # Model 1: XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror'
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        self.models['xgboost'] = xgb_model

        # Model 2: Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model

        # Model 3: Linear Regression (for baseline)
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        self.models['linear'] = lr_model

        # Validate all models
        val_scores = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_val)
            rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
            mape = np.mean(np.abs((y_pred - y_val) / y_val)) * 100
            val_scores[name] = {'rmse': rmse, 'mape': mape}

            mlflow.log_metric(f"{name}_val_rmse", rmse)
            mlflow.log_metric(f"{name}_val_mape", mape)

        # Determine ensemble weights based on validation performance
        weights = self.calculate_ensemble_weights(val_scores)

        mlflow.log_params(weights)
        mlflow.end_run()

        return weights

    def calculate_ensemble_weights(self, val_scores: dict) -> dict:
        """Calculate optimal ensemble weights using inverse error"""
        rmse_values = {name: scores['rmse'] for name, scores in val_scores.items()}

        # Inverse RMSE (lower error = higher weight)
        inverse_rmse = {name: 1.0 / rmse for name, rmse in rmse_values.items()}
        total_inverse = sum(inverse_rmse.values())

        # Normalize to sum to 1.0
        weights = {name: inv / total_inverse for name, inv in inverse_rmse.items()}

        return weights

    def predict_ensemble(self, X, weights: dict):
        """Make prediction using weighted ensemble"""
        predictions = []

        for name, model in self.models.items():
            pred = model.predict(X)
            weighted_pred = pred * weights[name]
            predictions.append(weighted_pred)

        return np.sum(predictions, axis=0)

    def evaluate_test_set(self, X_test, y_test, weights):
        """Final evaluation on test set"""
        y_pred = self.predict_ensemble(X_test, weights)

        metrics = {
            'rmse': np.sqrt(np.mean((y_pred - y_test) ** 2)),
            'mae': np.mean(np.abs(y_pred - y_test)),
            'mape': np.mean(np.abs((y_pred - y_test) / y_test)) * 100,
            'r2': 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
        }

        # Directional accuracy (for month-over-month predictions)
        direction_actual = np.sign(np.diff(y_test[:, 0]))  # Revenue month 1
        direction_pred = np.sign(np.diff(y_pred[:, 0]))
        metrics['directional_accuracy'] = np.mean(direction_actual == direction_pred)

        return metrics

    def save_model(self, version: str):
        """Save all models and metadata"""
        joblib.dump(self.models, f'models/ensemble_v{version}.pkl')
        joblib.dump(self.scaler, f'models/scaler_v{version}.pkl')
        joblib.dump(self.ensemble_weights, f'models/weights_v{version}.pkl')

# Main training execution
if __name__ == "__main__":
    trainer = ModelTrainer('data/studio_data_2019_2025.csv')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.prepare_data()

    weights = trainer.train_ensemble(X_train, y_train, X_val, y_val)
    trainer.ensemble_weights = weights

    test_metrics = trainer.evaluate_test_set(X_test, y_test, weights)
    print("Test Set Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    trainer.save_model(version="1.0.0")
```

### 4.2 Data Generation Script

```python
# scripts/generate_sample_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class StudioDataGenerator:
    def __init__(self, num_years=5, studio_type='Yoga'):
        self.num_years = num_years
        self.num_months = num_years * 12
        self.studio_type = studio_type

    def generate_baseline_metrics(self):
        """Generate base trajectory for 5 years"""
        months = pd.date_range('2019-01-01', periods=self.num_months, freq='MS')

        # Base values
        base_members = 200
        base_retention = 0.75
        base_ticket = 150.0
        base_classes = 120

        data = []
        current_members = base_members

        for i, month in enumerate(months):
            month_idx = month.month
            year_idx = i // 12

            # Seasonality factors
            january_boost = 1.25 if month_idx == 1 else 1.0
            summer_dip = 0.90 if month_idx in [6, 7, 8] else 1.0
            fall_recovery = 1.10 if month_idx == 9 else 1.0

            # Growth phase factors
            if year_idx <= 1:
                growth_factor = 1.012  # 1.2% monthly growth (Year 1-2)
            elif year_idx <= 3:
                growth_factor = 1.005  # 0.5% monthly growth (Year 3-4)
            else:
                growth_factor = 1.002  # 0.2% monthly growth (Year 5)

            # Calculate metrics with noise
            retention_rate = base_retention * summer_dip * (1 + np.random.normal(0, 0.03))
            retention_rate = np.clip(retention_rate, 0.6, 0.95)

            new_members = int(
                25 * january_boost * fall_recovery * growth_factor *
                (1 + np.random.normal(0, 0.15))
            )

            churned_members = int(current_members * (1 - retention_rate))
            current_members = current_members - churned_members + new_members
            current_members = max(current_members, 100)  # Floor

            avg_ticket = base_ticket * (1.005 ** i) * (1 + np.random.normal(0, 0.02))

            class_attendance_rate = 0.70 * summer_dip * (1 + np.random.normal(0, 0.05))
            class_attendance_rate = np.clip(class_attendance_rate, 0.5, 0.9)

            total_classes = int(base_classes * growth_factor * (1 + np.random.normal(0, 0.05)))
            total_attendance = int(total_classes * 20 * class_attendance_rate)  # Assume 20 max capacity

            staff_utilization = 0.80 * (1 + np.random.normal(0, 0.05))
            staff_utilization = np.clip(staff_utilization, 0.65, 0.95)

            upsell_rate = 0.25 * (1 + np.random.normal(0, 0.1))
            upsell_rate = np.clip(upsell_rate, 0.1, 0.45)

            # Revenue calculation
            membership_revenue = current_members * avg_ticket
            class_pack_revenue = total_attendance * 0.2 * 15  # 20% are drop-ins at $15
            retail_revenue = current_members * 10 * 0.3  # 30% buy retail
            upsell_revenue = current_members * upsell_rate * 50

            total_revenue = (
                membership_revenue +
                class_pack_revenue +
                retail_revenue +
                upsell_revenue
            )

            data.append({
                'month_year': month,
                'total_members': current_members,
                'new_members': new_members,
                'churned_members': churned_members,
                'retention_rate': retention_rate,
                'avg_ticket_price': avg_ticket,
                'total_classes_held': total_classes,
                'total_class_attendance': total_attendance,
                'class_attendance_rate': class_attendance_rate,
                'staff_utilization_rate': staff_utilization,
                'total_revenue': total_revenue,
                'membership_revenue': membership_revenue,
                'class_pack_revenue': class_pack_revenue,
                'retail_revenue': retail_revenue,
                'upsell_rate': upsell_rate,
                'month_index': month_idx,
                'year_index': year_idx
            })

        return pd.DataFrame(data)

    def add_target_variables(self, df):
        """Add future revenue targets for supervised learning"""
        df['revenue_month_1'] = df['total_revenue'].shift(-1)
        df['revenue_month_2'] = df['total_revenue'].shift(-2)
        df['revenue_month_3'] = df['total_revenue'].shift(-3)
        df['member_count_month_3'] = df['total_members'].shift(-3)
        df['retention_rate_month_3'] = df['retention_rate'].shift(-3)

        # Remove last 3 rows (no future data)
        df = df[:-3].copy()

        return df

    def add_data_split_labels(self, df):
        """Label train/val/test splits"""
        n = len(df)
        df['split'] = 'train'
        df.loc[int(n * 0.8):int(n * 0.9), 'split'] = 'validation'
        df.loc[int(n * 0.9):, 'split'] = 'test'
        return df

    def generate_full_dataset(self):
        """Generate complete dataset"""
        df = self.generate_baseline_metrics()
        df = self.add_target_variables(df)
        df = self.add_data_split_labels(df)
        return df

# Generate and save
generator = StudioDataGenerator(num_years=5, studio_type='Yoga')
data = generator.generate_full_dataset()
data.to_csv('data/studio_data_2019_2025.csv', index=False)

print(f"Generated {len(data)} months of data")
print(f"Train: {len(data[data['split'] == 'train'])}")
print(f"Validation: {len(data[data['split'] == 'validation'])}")
print(f"Test: {len(data[data['split'] == 'test'])}")
```

---

## Part 5: Implementation Checklist

### Phase 1: Data Foundation (Days 1-2)

- [ ] Generate 5 years of synthetic studio data with realistic patterns
- [ ] Set up PostgreSQL database with TimescaleDB extension
- [ ] Create all database tables and indexes
- [ ] Load sample data into database
- [ ] Validate data quality and distributions

### Phase 2: ML Model Development (Days 3-5)

- [ ] Implement feature engineering pipeline
- [ ] Train forward prediction model (levers → revenue)
- [ ] Evaluate model on validation and test sets
- [ ] Implement inverse prediction using optimization
- [ ] Test optimization convergence and feasibility
- [ ] Save trained models with versioning

### Phase 3: Backend API (Days 6-8)

- [ ] Set up FastAPI project structure
- [ ] Implement PredictionService with forward/inverse methods
- [ ] Implement FeatureEngineer service
- [ ] Implement LeverOptimizer service
- [ ] Create all REST API endpoints
- [ ] Add request validation and error handling
- [ ] Set up Redis caching layer

### Phase 4: AI Insights (Days 9-10)

- [ ] Implement InsightsService with tactics mapping
- [ ] Integrate OpenAI API for AI-enhanced recommendations
- [ ] Build action plan generation logic
- [ ] Create lever impact analysis endpoints

### Phase 5: Testing & Optimization (Days 11-12)

- [ ] Write unit tests for all services
- [ ] Write integration tests for API endpoints
- [ ] Load testing for performance validation
- [ ] Optimize database queries
- [ ] Fine-tune model hyperparameters
- [ ] Validate end-to-end prediction accuracy

### Phase 6: Documentation (Day 13)

- [ ] Set up MLflow for model tracking
- [ ] Implement prediction logging
- [ ] Create API documentation
- [ ] Write deployment guide

---

## Part 6: Key Success Metrics

**Model Performance Targets**:

- RMSE < $3,000 (for monthly revenue predictions)
- MAPE < 8% (mean absolute percentage error)
- R² Score > 0.85
- Directional Accuracy > 90%
- Prediction Confidence > 80%

**API Performance Targets**:

- Forward prediction latency < 200ms (p95)
- Inverse optimization latency < 1000ms (p95)
- Cache hit rate > 60%
- API availability > 99.5%

**Business Impact Targets**:

- Inverse optimization achieves target revenue within 5% tolerance in 85% of cases
- Recommended lever changes are feasible in 90% of scenarios
- AI-generated action plans rated as "actionable" by users in 80% of cases

---

## Part 7: Technology Stack Summary

**Backend**:

- FastAPI 0.104+
- Python 3.10+
- Pydantic for data validation

**Database**:

- PostgreSQL 15+ with TimescaleDB
- Redis 7+ for caching

**Machine Learning**:

- scikit-learn 1.3+
- XGBoost 2.0+
- LightGBM 4.0+
- TensorFlow 2.14+ or PyTorch 2.0+ (for neural networks)
- MLflow for experiment tracking

**AI/LLM**:

- OpenAI API (GPT-4) for insights generation

---

## Part 8: Expected Deliverables

1. **Trained ML Models**:

   - Forward prediction model (ensemble) with >85% R²
   - Optimization module for inverse prediction
   - Model artifacts saved with versioning

2. **Backend API**:

   - RESTful API with 10+ endpoints
   - Complete API documentation (Swagger/OpenAPI)
   - Comprehensive error handling

3. **Database**:

   - Fully populated PostgreSQL database with 5 years of sample data
   - Optimized schemas and indexes
   - Migration scripts

4. **AI Insights Engine**:

   - Lever-to-tactics mapping system
   - OpenAI integration for personalized recommendations
   - Action plan generation with priority ranking

5. **Documentation**:

   - API documentation
   - Model documentation (architecture, features, performance)
   - Deployment guide

6. **Testing Suite**:
   - Unit tests (80%+ coverage)
   - Integration tests for all endpoints

---

## Part 9: Future Enhancements

**Model Improvements**:

- Incorporate external factors (local competition, economic indicators)
- Add member segmentation for personalized predictions
- Implement time-series forecasting (Prophet, LSTM) for longer horizons
- Multi-studio comparison and benchmarking

**Feature Additions**:

- A/B scenario comparison mode
- Auto-optimize button (find optimal lever mix automatically)
- Industry benchmarks overlay
- Real-time model retraining pipeline
- What-if analysis with constraint relaxation

**Integration**:

- Mindbody API integration for real data ingestion
- Slack bot for notifications and queries
- Export to business intelligence tools (Tableau, PowerBI)

---

This comprehensive prompt provides everything needed to build a production-ready fitness studio revenue simulator with predictive analytics, optimization capabilities, and AI-powered insights. The architecture is designed for scalability, maintainability, and real-world business impact.
