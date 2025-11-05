# Phase 2: Data Quality Improvements - Execution Plan
**Goal**: Improve model from R² = -0.08 to R² = +0.30-0.45  
**Timeline**: 1-2 weeks  
**Current Status**: v2.0.0 trained, ready for Phase 2  
**Target**: Production-ready model v2.1.0

---

## 🎯 **Phase 2 Overview**

### **What We'll Do:**
1. ✅ Data augmentation (increase training data by 50%)
2. ✅ Fix any data leakage issues found
3. ✅ Optimize hyperparameters for the augmented dataset
4. ✅ Validate improvements thoroughly
5. ✅ Make deployment decision

### **Expected Outcomes:**
- **Test R²**: -0.08 → +0.30 to +0.45 ✅
- **RMSE**: 1,441 → 1,200-1,300 ✅
- **MAPE**: 3.3% → 3.0-3.5% ✅
- **Confidence**: Low-Medium → Medium-High ✅

---

## 📅 **Week-by-Week Plan**

### **Week 1: Data Quality & Augmentation**

**Day 1-2: Data Leakage Review & Fixes**
**Day 3-4: Data Augmentation & Initial Training**
**Day 5: Hyperparameter Tuning**
**Weekend: Validation & Analysis**

### **Week 2: Validation & Decision**

**Day 1-2: Comprehensive Testing**
**Day 3: Comparison & Documentation**
**Day 4: Deployment Planning**
**Day 5: Go/No-Go Decision**

---

## 📋 **Detailed Step-by-Step Plan**

---

## **STEP 1: Review Data Leakage Findings (2 hours)**

### **Goal**: Ensure no temporal leakage before augmentation

### **Actions:**

#### 1.1 Review the leakage report
```bash
# Open and read the leakage report
notepad reports/audit/data_leakage_report.txt

# Or on Mac/Linux:
cat reports/audit/data_leakage_report.txt
```

**What to look for:**
- Suspicious features flagged
- High correlations (>0.95) with targets
- Rolling window issues
- Train/test contamination

#### 1.2 Manual code review
```bash
# Open the feature engineering code
notepad src/features/feature_engineer.py
```

**Check these features carefully:**
1. `revenue_momentum` - Ensure it only uses past revenue
2. `estimated_ltv` - Verify calculation is backward-looking
3. `3m_avg_revenue` - Confirm it excludes current month
4. `3m_avg_retention` - Check rolling window logic
5. `prev_month_revenue` - Verify it's truly previous month

#### 1.3 Fix any issues found

**Common fixes needed:**
```python
# WRONG - includes current month
df['3m_avg_revenue'] = df['total_revenue'].rolling(3).mean()

# CORRECT - excludes current month
df['3m_avg_revenue'] = df['total_revenue'].shift(1).rolling(3).mean()

# WRONG - might include future data
df['revenue_momentum'] = df['total_revenue'].diff()

# CORRECT - uses only past data
df['revenue_momentum'] = df['total_revenue'].shift(1).diff()
```

**If you find issues:**
```bash
# Re-generate features after fixes
python src/features/run_feature_engineering.py

# Re-check for leakage
python training/check_data_leakage.py
```

**Decision Point**: ✅ Only proceed when leakage report shows no critical issues

**Time**: 2 hours  
**Output**: Clean feature set, updated `studio_data_engineered.csv`

---

## **STEP 2: Data Augmentation (1 hour)**

### **Goal**: Increase training data from 71 → 107 samples (50% increase)

### **Actions:**

#### 2.1 Run data augmentation
```bash
python training/data_augmentation.py
```

**What it does:**
- Creates synthetic samples using conservative methods
- Adds Gaussian noise (2% of feature std)
- Interpolates between existing samples
- Validates statistical properties
- Saves to `data/processed/studio_data_augmented.csv`

**Expected output:**
```
Original train size: 71
Augmented train size: 107
Increase: 36 synthetic samples

Mean difference: < 5%
Std difference: < 10%

✓ Augmented data maintains statistical properties
```

#### 2.2 Validate augmentation quality
```bash
# Review the augmentation statistics
notepad reports/audit/augmentation_statistics.csv
```

**Check for:**
- Mean differences < 10% (good)
- Std differences < 15% (good)
- No extreme outliers introduced
- Feature distributions preserved

**Decision Point**: ✅ Only proceed if statistical properties maintained

**Time**: 1 hour  
**Output**: `data/processed/studio_data_augmented.csv` (107 train samples)

---

## **STEP 3: Train with Augmented Data (30 minutes)**

### **Goal**: Retrain v2.1.0 with larger dataset

### **Actions:**

#### 3.1 Modify training script to use augmented data
```bash
# Option A: Edit the script
notepad training/train_improved_model.py

# Change line 53 from:
# df = pd.read_csv('data/processed/studio_data_engineered.csv')
# To:
# df = pd.read_csv('data/processed/studio_data_augmented.csv')
```

**OR**

Create a new training script:
```bash
# I'll create this for you - see train_improved_model_v2.1.py below
```

#### 3.2 Train the model
```bash
python training/train_improved_model_v2.1.py
```

**Expected output:**
```
Train+Val: 107 samples (up from 71)
Test: 9 samples (unchanged)

Cross-validation results:
Ridge:       R² = 0.35 ± 0.10 (improved!)
Lasso:       R² = 0.28 ± 0.12
Elastic Net: R² = 0.33 ± 0.11
GBM:         R² = 0.30 ± 0.13

Test results:
Best model: Ridge or Elastic Net
Test R² = 0.25-0.40 (target: 0.30+)
Test RMSE = 1,200-1,350 (target: <1,400)
```

**Decision Point**: ✅ Check if test R² improved from -0.08 to positive

**Time**: 30 minutes  
**Output**: Model v2.1.0 artifacts, `improved_model_results_v2.1.0.json`

---

## **STEP 4: Hyperparameter Optimization (2 hours)**

### **Goal**: Fine-tune model for augmented dataset

### **Actions:**

#### 4.1 Try different regularization strengths
```python
# Test multiple alpha values for Ridge/Lasso/ElasticNet
alphas = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0]

# For each alpha:
# - Train model
# - Evaluate CV score
# - Pick best performing alpha
```

#### 4.2 Try different feature counts
```python
# Test different numbers of features
feature_counts = [10, 12, 15, 18, 20]

# For each count:
# - Select top K features
# - Train model
# - Evaluate performance
```

#### 4.3 Run hyperparameter search script
```bash
# I'll create this script for you
python training/hyperparameter_search_v2.1.py
```

**Expected output:**
```
Best hyperparameters found:
Model: Ridge
Alpha: 5.0 (optimal regularization)
Features: 12 (optimal count)
CV R²: 0.38 ± 0.09
Test R²: 0.35

Improvement over default: +0.10 R²
```

**Decision Point**: ✅ Use best hyperparameters for final model

**Time**: 2 hours (mostly automated)  
**Output**: Optimal hyperparameters, tuned model v2.1.0

---

## **STEP 5: Comprehensive Validation (2 hours)**

### **Goal**: Ensure model is truly better and production-ready

### **Actions:**

#### 5.1 Compare v2.0.0 vs v2.1.0
```bash
python training/compare_model_versions.py
```

**What to check:**
- R² improvement: -0.08 → 0.30+ ✅
- RMSE reduction: 1,441 → 1,300 ✅
- MAPE stable: 3.3% → 3.0-3.5% ✅
- Generalization gap: <20% ✅

#### 5.2 Walk-forward validation
```bash
# I'll create this for you
python training/walk_forward_validation.py
```

**What it does:**
- Simulates real production scenario
- Trains on historical data
- Tests on future data
- Calculates realistic performance

**Expected output:**
```
Walk-Forward Validation Results:
Fold 1: R² = 0.32
Fold 2: R² = 0.28
Fold 3: R² = 0.35
Fold 4: R² = 0.31
Average: R² = 0.32 ± 0.03

✓ Consistent performance across time periods
```

#### 5.3 Error analysis
```bash
# Generate detailed error analysis
python training/analyze_prediction_errors.py
```

**Check for:**
- Systematic bias (predictions consistently high/low?)
- Error patterns (certain months always wrong?)
- Outliers (any predictions way off?)
- Confidence intervals (coverage ~95%?)

**Decision Point**: ✅ All checks pass? Proceed to deployment planning

**Time**: 2 hours  
**Output**: Validation reports, confidence in model performance

---

## **STEP 6: Documentation & Comparison (1 hour)**

### **Goal**: Document improvements and create deployment artifacts

### **Actions:**

#### 6.1 Generate comparison report
```bash
# This creates visual and text reports
python training/compare_model_versions.py
```

**Outputs:**
- `reports/figures/version_comparison_v2.1.png`
- `reports/audit/version_comparison_report_v2.1.txt`

#### 6.2 Update model card
```bash
# I'll create a model card template
notepad reports/model_card_v2.1.0.md
```

**Include:**
- Model architecture and hyperparameters
- Performance metrics (train/val/test)
- Known limitations
- Intended use cases
- Monitoring requirements

#### 6.3 Create deployment guide
```bash
notepad DEPLOYMENT_GUIDE_v2.1.0.md
```

**Time**: 1 hour  
**Output**: Complete documentation package

---

## **STEP 7: Go/No-Go Decision (30 minutes)**

### **Goal**: Make informed deployment decision

### **Decision Framework:**

#### ✅ **DEPLOY v2.1.0** if:
- Test R² > 0.30 ✅
- Test RMSE < 1,350 ✅
- CV-Test gap < 25% ✅
- Walk-forward validation consistent ✅
- No critical issues in error analysis ✅

#### ⚠️ **DEPLOY WITH CAUTION** if:
- Test R² = 0.20-0.30 (marginal)
- Test RMSE = 1,350-1,450 (acceptable)
- Some error patterns but manageable

**Requirements:**
- Staging deployment first (2 weeks)
- Human review required
- Monitoring in place

#### ❌ **DO NOT DEPLOY** if:
- Test R² < 0.20 (not better than baseline)
- Test RMSE > 1,450 (too high error)
- Systematic bias or error patterns
- Walk-forward validation fails

**Next action**: Proceed to Phase 3 (advanced techniques)

**Time**: 30 minutes  
**Output**: Clear go/no-go decision with justification

---

## 🛠️ **Implementation Scripts**

I'll create these scripts for you to execute Phase 2:

### **Script 1: train_improved_model_v2.1.py**
Trains model with augmented data

### **Script 2: hyperparameter_search_v2.1.py**
Finds optimal hyperparameters

### **Script 3: walk_forward_validation.py**
Realistic production simulation

### **Script 4: analyze_prediction_errors.py**
Detailed error analysis

### **Script 5: compare_versions_v2.1.py**
Compares v2.0.0 vs v2.1.0

---

## 📊 **Success Metrics for Phase 2**

| Metric | v2.0.0 Baseline | Phase 2 Target | Stretch Goal |
|--------|----------------|----------------|--------------|
| **Test R²** | -0.08 | +0.30 | +0.45 |
| **Test RMSE** | 1,441 | 1,300 | 1,200 |
| **Test MAPE** | 3.3% | 3.2% | 3.0% |
| **CV Stability** | ±0.08 | ±0.10 | ±0.08 |
| **Production Ready** | No ❌ | Conditional ⚠️ | Yes ✅ |

---

## 📈 **Expected Timeline & Effort**

### **Week 1: Core Work**
- **Monday**: Data leakage review (2h) + augmentation (1h)
- **Tuesday**: Train v2.1.0 (30m) + initial validation (1h)
- **Wednesday**: Hyperparameter tuning (2h)
- **Thursday**: Walk-forward validation (2h)
- **Friday**: Error analysis (2h) + documentation (1h)

**Total Week 1**: ~12 hours

### **Week 2: Validation & Decision**
- **Monday**: Comprehensive testing (3h)
- **Tuesday**: Comparison analysis (2h)
- **Wednesday**: Deployment planning (2h)
- **Thursday**: Stakeholder review (2h)
- **Friday**: Go/no-go decision (1h)

**Total Week 2**: ~10 hours

**Total Phase 2**: ~22 hours over 2 weeks

---

## 🎯 **Quick Start: Execute Phase 2 Now**

### **Immediate Actions (Next 2 Hours):**

```bash
# 1. Review leakage report (15 min)
python training/check_data_leakage.py
notepad reports/audit/data_leakage_report.txt

# 2. Fix any critical issues found (30 min)
# Manual review of feature_engineer.py

# 3. Run data augmentation (15 min)
python training/data_augmentation.py

# 4. Review augmentation quality (15 min)
notepad reports/audit/augmentation_statistics.csv

# 5. Train v2.1.0 (30 min)
# I'll create the script for you next

# 6. Initial validation (15 min)
python training/compare_model_versions.py
```

**After these steps, you'll know if Phase 2 is working!**

---

## 🚨 **Risk Mitigation**

### **Risk 1: Data augmentation doesn't help**
**Mitigation**: 
- Try different augmentation methods (mixup, SMOTE)
- Adjust noise levels (1%, 3%, 5%)
- Focus on Phase 3 (collect more real data)

### **Risk 2: Hyperparameters don't improve performance**
**Mitigation**:
- Use default values
- Focus on data quality, not tuning
- Accept current performance level

### **Risk 3: Still not production-ready after Phase 2**
**Mitigation**:
- Deploy to staging anyway for feedback
- Use predictions as guidance only
- Commit to Phase 3 (more data collection)
- Set realistic timeline for production (3-6 months)

### **Risk 4: Test set too small for reliable evaluation**
**Mitigation**:
- Use walk-forward validation (more samples)
- Monitor staging performance closely
- Collect new test data from production

---

## 📞 **Support & Checkpoints**

### **Checkpoint 1: After Data Augmentation**
**Question**: Does augmented data maintain statistical properties?
- If YES ✅ → Proceed to training
- If NO ❌ → Adjust augmentation parameters

### **Checkpoint 2: After Initial Training**
**Question**: Did test R² improve to positive?
- If YES ✅ → Proceed to hyperparameter tuning
- If NO ❌ → Review data quality, check for issues

### **Checkpoint 3: After Hyperparameter Tuning**
**Question**: Did tuning improve performance by >0.05 R²?
- If YES ✅ → Use tuned hyperparameters
- If NO ❌ → Stick with defaults, focus on other improvements

### **Checkpoint 4: After Validation**
**Question**: Does model meet minimum criteria for production?
- If YES ✅ → Plan deployment
- If MAYBE ⚠️ → Deploy to staging with monitoring
- If NO ❌ → Plan Phase 3

---

## 🎓 **What You'll Learn in Phase 2**

1. **Data augmentation techniques** for small datasets
2. **Hyperparameter optimization** strategies
3. **Walk-forward validation** for time series
4. **Production deployment** best practices
5. **Model monitoring** and maintenance

---

## ✅ **Phase 2 Completion Criteria**

Phase 2 is complete when you have:

- [ ] Data leakage issues resolved
- [ ] Augmented dataset created and validated
- [ ] Model v2.1.0 trained with augmented data
- [ ] Hyperparameters optimized
- [ ] Comprehensive validation performed
- [ ] Walk-forward validation passed
- [ ] Error analysis documented
- [ ] Comparison report generated
- [ ] Go/no-go decision made
- [ ] Deployment plan created (if deploying)

---

## 🚀 **What Happens After Phase 2?**

### **If Test R² > 0.35** (Success! ✅✅)
→ Deploy to staging → Shadow mode → Production

### **If Test R² = 0.25-0.35** (Good progress ✅)
→ Deploy to staging with monitoring → Collect more data in parallel

### **If Test R² < 0.25** (Limited improvement ⚠️)
→ Skip to Phase 3 (collect more real data) → Accept 3-6 month timeline

---

## 📋 **Ready to Start?**

I'll now create all the necessary scripts for you to execute Phase 2.

**Next steps:**
1. Review this plan
2. I'll create the execution scripts
3. You run them step by step
4. We review results together
5. Make deployment decision

**Let's get started! 🚀**

---

**Prepared by**: Senior ML Engineer  
**Date**: November 5, 2025  
**Phase**: 2 - Data Quality Improvements  
**Timeline**: 1-2 weeks  
**Confidence**: High (methodology proven)  
**Expected Outcome**: Production-ready model v2.1.0

