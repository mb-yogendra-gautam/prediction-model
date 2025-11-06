# API Request Examples

This directory contains sample JSON requests for testing the Studio Revenue Simulator API.

## Files

### `inverse_prediction_request.json`
Basic inverse prediction request to find optimal lever values for a target revenue goal.

**Usage:**
```bash
curl -X POST http://localhost:8000/api/v1/predictions/inverse \
  -H "Content-Type: application/json" \
  -d @examples/inverse_prediction_request.json
```

**Features demonstrated:**
- Target revenue optimization
- Custom constraints
- Ensemble optimization method
- Single objective (revenue)

### `scenario_comparison_request.json`
Compare multiple optimization scenarios with different constraints and objectives.

**Usage:**
```bash
curl -X POST http://localhost:8000/api/v1/predictions/inverse/compare-scenarios \
  -H "Content-Type: application/json" \
  -d @examples/scenario_comparison_request.json
```

**Scenarios included:**
1. **Conservative**: Easy wins, minimal changes
2. **Balanced**: Moderate growth, good feasibility
3. **Aggressive**: Maximum growth, multi-objective

## Customization

### Modify Current State

Edit the `current_state` section to match your studio's actual values:

```json
{
  "current_state": {
    "retention_rate": 0.75,        // Your current retention (0.5-1.0)
    "avg_ticket_price": 150.0,     // Your current price ($50-$500)
    "class_attendance_rate": 0.70, // Your current attendance (0.4-1.0)
    "new_members": 25,             // Your new members/month (0-100)
    "staff_utilization_rate": 0.85,// Your staff utilization (0.6-1.0)
    "upsell_rate": 0.25,           // Your upsell rate (0.0-0.5)
    "total_classes_held": 120,     // Your classes/month (50-500)
    "total_members": 200           // Your total members (50+)
  }
}
```

### Adjust Constraints

Control how much each lever can change:

```json
{
  "constraints": {
    "max_retention_increase": 0.05,     // Max +5% retention
    "max_ticket_increase": 20.0,        // Max +$20 price
    "max_new_members_increase": 10      // Max +10 new members/month
  }
}
```

### Choose Optimization Method

Options for the `method` field:
- `"auto"`: Try multiple methods, pick best
- `"lbfgs"`: Fast gradient-based (good for smooth functions)
- `"slsqp"`: Good constraint handling
- `"de"`: Global search (best for non-convex)
- `"ensemble"`: Try all methods, return best

### Set Objectives

Options for the `objectives` field:
- `["revenue"]`: Optimize for revenue only
- `["revenue", "retention"]`: Revenue + improve retention
- `["revenue", "retention", "growth"]`: All three objectives

## Testing Workflow

1. **Start the API**:
   ```bash
   python run_api.py
   ```

2. **Test basic inverse prediction**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/predictions/inverse \
     -H "Content-Type: application/json" \
     -d @examples/inverse_prediction_request.json \
     | jq
   ```

3. **Compare scenarios**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/predictions/inverse/compare-scenarios \
     -H "Content-Type: application/json" \
     -d @examples/scenario_comparison_request.json \
     | jq
   ```

4. **Run comprehensive test suite**:
   ```bash
   python test_inverse_prediction.py
   ```

## Python Examples

### Basic Usage

```python
import requests
import json

# Load request
with open('examples/inverse_prediction_request.json') as f:
    request_data = json.load(f)

# Make request
response = requests.post(
    'http://localhost:8000/api/v1/predictions/inverse',
    json=request_data
)

# Process response
result = response.json()
print(f"Target: ${result['target_revenue']:,.2f}")
print(f"Achievable: ${result['achievable_revenue']:,.2f}")
print(f"Achievement Rate: {result['achievement_rate']*100:.1f}%")
```

### Scenario Comparison

```python
import requests
import json

# Load request
with open('examples/scenario_comparison_request.json') as f:
    request_data = json.load(f)

# Make request
response = requests.post(
    'http://localhost:8000/api/v1/predictions/inverse/compare-scenarios',
    json=request_data
)

# Process response
result = response.json()
print(f"Recommended Scenario: {result['recommended_scenario']}")
print("\nComparison:")
print(result['comparison_summary'])
```

## Response Fields

### Key Response Fields

- `achievable_revenue`: Predicted revenue with optimal levers
- `achievement_rate`: How close to target (0-1, where 1.0 = 100%)
- `confidence_score`: Confidence in recommendation (0-1)
- `recommended_levers`: Optimal lever values
- `lever_changes`: Detailed changes with priorities
- `action_plan`: Prioritized action items
- `sensitivity_analysis`: Impact of each lever on revenue
- `feasibility_assessment`: How realistic the changes are
- `uncertainty`: Confidence intervals and prediction ranges

## Tips

1. **Start Conservative**: Use tight constraints first to get realistic recommendations
2. **Compare Scenarios**: Don't rely on a single optimization
3. **Check Feasibility**: "Very hard" recommendations may not be practical
4. **Use Sensitivity**: Focus on high-sensitivity levers for maximum impact
5. **Consider Uncertainty**: Wide confidence intervals indicate higher risk

## More Examples

For more examples and detailed explanations, see:
- [Inverse Prediction Guide](../docs/INVERSE_PREDICTION_GUIDE.md)
- [Test Suite](../test_inverse_prediction.py)
- [API Documentation](../docs/API_DOCUMENTATION.md)

