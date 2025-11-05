# Data Leakage Validation Guide

## Overview

This guide explains how to use the validation tools to detect and investigate potential data leakage in the Studio Revenue Simulator models.

## Why Data Leakage Matters

Data leakage occurs when information from the future "leaks" into the training data, leading to unrealistically high performance that won't generalize to production. Common signs:

- **R² > 0.99**: Extremely rare in forecasting, suggests leakage
- **R² > 0.95**: Unusually high, warrants investigation  
- **Perfect predictions**: Model performs too well on test data
- **Train = Test performance**: No generalization gap

## Current Model Performance

**Multi-Studio Model v2.2.0:**
- Test R² = 0.9990 (99.9% accuracy)
- ⚠️ **This is SUSPICIOUSLY HIGH**

## Validation Tools

### 1. Comprehensive Leakage Check (START HERE)

**Purpose**: Quick automated checks for common leakage patterns

```bash
python training/comprehensive_leakage_check.py
```

**What it checks:**
- ✓ Target columns not in feature set
- ✓ Lagged features use previous period data
- ✓ Rolling windows exclude current period
- ✓ Performance is realistic
- ✓ No future data dependencies

**Output:**
- `reports/audit/comprehensive_leakage_investigation.txt`
- Status: APPROVED / CONDITIONAL / REJECTED

### 2. Walk-Forward Validation

**Purpose**: Test model on truly unseen future data (gold standard)

```bash
python training/walk_forward_validation.py
```

**What it does:**
- Trains on historical data (e.g., 36 months)
- Tests on future periods (e.g., 3 months ahead)
- Moves forward in time and repeats
- Checks for performance degradation

**Expected Results:**
- **With leakage**: R² stays > 0.99 across all folds
- **Without leakage**: R² varies, typically 0.60-0.85

**Output:**
- `reports/audit/walk_forward_validation.txt`
- `reports/audit/walk_forward_validation.json`

### 3. Baseline Comparison

**Purpose**: Verify trained model beats simple baselines

```bash
python training/baseline_comparison.py
```

**Baselines:**
1. **Last Value**: Use current month's revenue
2. **Mean**: Use training data average
3. **Linear Trend**: Extrapolate recent trend

**Expected Results:**
- Trained model should beat baselines by R² > 0.20
- If baseline performs similarly, model may not be useful

**Output:**
- `reports/audit/baseline_comparison.txt`
- `reports/audit/baseline_comparison.json`

## Recommended Workflow

### Step 1: Quick Check (5 minutes)
```bash
python training/comprehensive_leakage_check.py
```

**If REJECTED**: Stop, fix critical issues before proceeding

**If CONDITIONAL or APPROVED**: Continue to Step 2

### Step 2: Walk-Forward Validation (15-30 minutes)
```bash
python training/walk_forward_validation.py
```

**Expected results WITHOUT leakage:**
- R² drops from 0.99 to 0.60-0.85
- RMSE increases significantly
- Performance varies across folds

**If R² stays > 0.95**: Leakage likely present

### Step 3: Baseline Comparison (5 minutes)
```bash
python training/baseline_comparison.py
```

**Check:**
- Does trained model beat "Last Value" baseline?
- Is improvement meaningful (>0.2 R² improvement)?

### Step 4: Manual Code Review

**Review** `src/features/feature_engineer.py`:

**Rolling Features (lines 180-218):**
```python
# CORRECT: Uses .shift(1) to exclude current period
df['3m_avg_revenue'] = df['total_revenue'].shift(1).rolling(window=3).mean()

# WRONG: Includes current period (leakage!)
df['3m_avg_revenue'] = df['total_revenue'].rolling(window=3).mean()
```

**Lagged Features (lines 155-178):**
```python
# CORRECT: Uses .shift(1)
df['prev_month_revenue'] = df['total_revenue'].shift(1)

# WRONG: Uses current value
df['prev_month_revenue'] = df['total_revenue']
```

**Derived Features (lines 131-154):**
```python
# ACCEPTABLE: Uses current period data that IS known at prediction time
df['revenue_per_member'] = df['total_revenue'] / df['total_members']

# QUESTIONABLE: Should these use current or lagged values?
df['estimated_ltv'] = df['avg_ticket_price'] * df['retention_rate'] * 12
```

## Common Leakage Patterns

### Pattern 1: Rolling Window Includes Current
```python
# WRONG
df['3m_avg'] = df['value'].rolling(3).mean()

# CORRECT
df['3m_avg'] = df['value'].shift(1).rolling(3).mean()
```

### Pattern 2: Growth Calculation Using Future
```python
# WRONG: Uses next month's value
df['growth'] = (df['value'].shift(-1) - df['value']) / df['value']

# CORRECT: Uses previous month's value
df['growth'] = (df['value'] - df['value'].shift(1)) / df['value'].shift(1)
```

### Pattern 3: Target Information in Features
```python
# WRONG: Directly uses target
df['feature'] = df['revenue_month_1'] * 2

# CORRECT: Uses only current/past data
df['feature'] = df['total_revenue'] * 2
```

### Pattern 4: Look-Ahead Bias
```python
# WRONG: Scales using all data (includes test set)
scaler.fit(X_all)

# CORRECT: Scales using only train data
scaler.fit(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Fixing Data Leakage

### If Leakage Found:

1. **Identify the source**:
   - Check validation tool outputs
   - Review feature engineering code
   - Look for patterns listed above

2. **Fix the code**:
   - Add `.shift(1)` to rolling features
   - Verify lagged features use proper offsets
   - Remove target information from features

3. **Re-train the model**:
   ```bash
   python src/features/run_feature_engineering.py
   python training/train_improved_model.py
   ```

4. **Re-validate**:
   ```bash
   python training/comprehensive_leakage_check.py
   python training/walk_forward_validation.py
   ```

5. **Verify realistic performance**:
   - R² should drop to 0.60-0.85 range
   - Model should still beat baselines

## Production Deployment Checklist

Before deploying to production:

- [ ] Comprehensive leakage check: APPROVED
- [ ] Walk-forward validation: R² < 0.95
- [ ] Beats baseline models by meaningful margin
- [ ] Code review completed
- [ ] Tested on truly unseen future data
- [ ] Monitoring and alerts configured
- [ ] Rollback plan in place

## Expected Realistic Performance

For revenue forecasting 1-3 months ahead:

| Metric | Good | Excellent | Suspicious |
|--------|------|-----------|------------|
| R² | 0.60-0.80 | 0.80-0.90 | > 0.95 |
| MAPE | 5-15% | 2-5% | < 2% |
| RMSE | Varies by scale | - | Too low |

## Troubleshooting

### "Model performance dropped after fixing leakage"

**This is expected and good!** It means:
- You successfully removed the leakage
- Model is now giving realistic performance
- Predictions will generalize to production

### "Walk-forward validation is very slow"

```python
# Reduce folds or samples
results = validator.walk_forward_validate(
    df, 
    train_months=24,  # Reduce from 36
    test_months=3,
    step=6  # Increase from 3 (fewer folds)
)
```

### "Baseline performs better than trained model"

Either:
- Features aren't predictive
- Model is overfit
- Baseline is actually good (simple is better!)

## Additional Resources

- **Data Leakage Report**: `reports/audit/data_leakage_report.txt`
- **Feature Engineering Code**: `src/features/feature_engineer.py`
- **Model Training Code**: `training/train_improved_model.py`

## Questions?

Common questions:

**Q: Is R² = 0.85 good enough?**  
A: Yes! That's excellent for 3-month revenue forecasting.

**Q: Should I worry about R² = 0.99?**  
A: YES. This is almost certainly data leakage.

**Q: How do I know if my fix worked?**  
A: R² should drop to realistic levels (0.60-0.85), but still beat baselines.

**Q: Can I deploy with R² = 0.96?**  
A: Not recommended. Investigate further with walk-forward validation.

## Contact

For technical issues, review:
1. Validation tool outputs in `reports/audit/`
2. Feature engineering code in `src/features/`
3. This guide for troubleshooting steps

