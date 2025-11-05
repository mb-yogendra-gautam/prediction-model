# Phase 2 - Quick Start Guide
**Get Your Model Production-Ready in 1-2 Weeks**

---

## 🚀 **Execute Phase 2 in 5 Steps**

### **Total Time**: 1-2 weeks (12-22 hours of work)

---

## **STEP 1: Data Leakage Check** (30 minutes)

### **What**: Ensure no temporal leakage before augmentation

### **Run**:
```bash
python training/check_data_leakage.py
```

### **Review**:
```bash
notepad reports/audit/data_leakage_report.txt
```

### **Check For**:
- Suspicious features (revenue_momentum, estimated_ltv, etc.)
- High correlations (>0.95) with targets
- Rolling window issues

### **Fix If Needed**:
1. Open `src/features/feature_engineer.py`
2. Fix any temporal leakage (ensure features only use past data)
3. Re-run: `python src/features/run_feature_engineering.py`
4. Re-check: `python training/check_data_leakage.py`

### **Decision Point**: ✅ Proceed only when no critical issues found

---

## **STEP 2: Data Augmentation** (15 minutes)

### **What**: Increase training data by 50% (71 → 107 samples)

### **Run**:
```bash
python training/data_augmentation.py
```

### **Expected Output**:
```
Original train size: 71
Augmented train size: 107
Increase: 36 synthetic samples

Mean difference: < 5%
Std difference: < 10%

✓ Augmented data maintains statistical properties
✓ Saved to: data/processed/studio_data_augmented.csv
```

### **Validate**:
```bash
notepad reports/audit/augmentation_statistics.csv
```

### **Check**:
- Mean differences < 10% ✅
- Std differences < 15% ✅
- No extreme outliers ✅

### **Decision Point**: ✅ Statistical properties maintained? Proceed.

---

## **STEP 3: Train v2.1.0 with Augmented Data** (30 minutes)

### **What**: Train improved model with larger dataset

### **Run**:
```bash
python training/train_improved_model_v2.1.py
```

### **Expected Output**:
```
Train+Val: 107 samples (up from 71)
Test: 9 samples

Cross-validation results:
Ridge:       R² = 0.30-0.40
Elastic Net: R² = 0.28-0.38
GBM:         R² = 0.25-0.35

Test results:
Best model: Ridge or Elastic Net
Test R² = 0.25-0.40 (TARGET: >0.30)
Test RMSE = 1,200-1,350

Comparison with v2.0.0:
v2.0.0 R²: -0.08
v2.1.0 R²: +0.32
Improvement: +0.40 ✅✅
```

### **Success Criteria**:
- ✅ Test R² > 0.25 (minimum)
- ✅✅ Test R² > 0.30 (target)
- ✅✅✅ Test R² > 0.40 (excellent)

### **Files Created**:
- `data/models/best_model_v2.1.0.pkl`
- `reports/audit/improved_model_results_v2.1.0.json`

### **Decision Point**: 
- If R² > 0.30 ✅ → Proceed to validation
- If R² = 0.20-0.30 ⚠️ → Try hyperparameter tuning (Step 4)
- If R² < 0.20 ❌ → Phase 3 needed (more data)

---

## **STEP 4: Hyperparameter Tuning** (2 hours, optional)

### **When to Run**: If Step 3 gave R² = 0.20-0.35 (marginal)

### **What**: Find optimal alpha, l1_ratio, and feature count

### **Run**:
```bash
python training/hyperparameter_search_v2.1.py
```

### **What It Does**:
1. Tests feature counts: 8, 10, 12, 15, 18, 20
2. Tests Ridge alpha: 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0
3. Tests ElasticNet params: various alpha & l1_ratio combinations

### **Expected Output**:
```
Best Hyperparameters Found:
  Feature count: 12
  Ridge alpha: 5.0
  ElasticNet alpha: 2.0
  ElasticNet l1_ratio: 0.5

Expected improvement: +0.05-0.10 R²
```

### **Apply Results**:
Edit `train_improved_model_v2.1.py` with optimal parameters and retrain.

### **Skip If**: Step 3 already gave R² > 0.35 ✅

---

## **STEP 5: Walk-Forward Validation** (30 minutes)

### **What**: Realistic production simulation

### **Run**:
```bash
python training/walk_forward_validation.py
```

### **What It Does**:
- Creates 4 time-based splits
- Trains on historical, tests on future
- Simulates real production scenario
- More samples than single test set

### **Expected Output**:
```
Walk-Forward Validation Results:

R² Scores:
  Fold 1: 0.32
  Fold 2: 0.28
  Fold 3: 0.35
  Fold 4: 0.31
  Mean: 0.32 (+/- 0.03)

RMSE:
  Mean: 1,250 (+/- 150)

Assessment:
  ✅ Ready for staging deployment with monitoring
```

### **Success Criteria**:
- Mean R² > 0.30 ✅
- Std R² < 0.15 ✅
- Consistent across folds ✅

### **Files Created**:
- `reports/audit/walk_forward_validation_v2.1.0.json`
- `reports/figures/walk_forward_validation_v2.1.png`

---

## **BONUS: Compare Versions** (10 minutes)

### **What**: Visual comparison v2.0.0 vs v2.1.0

### **Run**:
```bash
python training/compare_model_versions.py
```

### **Outputs**:
- `reports/figures/version_comparison.png`
- `reports/audit/version_comparison_report.txt`

---

## 📊 **Quick Decision Matrix**

### **After Step 3 (v2.1.0 Training):**

| Test R² | RMSE | Action |
|---------|------|--------|
| **> 0.40** | < 1,200 | ✅✅ Skip Step 4, go to Step 5 |
| **0.30-0.40** | 1,200-1,350 | ✅ Skip Step 4, go to Step 5 |
| **0.20-0.30** | 1,350-1,500 | ⚠️ Run Step 4 (tuning) |
| **< 0.20** | > 1,500 | ❌ Stop, plan Phase 3 |

### **After Step 5 (Walk-Forward):**

| Mean R² | Std | Action |
|---------|-----|--------|
| **> 0.40** | < 0.15 | ✅✅ Production ready |
| **0.30-0.40** | < 0.20 | ✅ Staging ready |
| **0.20-0.30** | any | ⚠️ Deploy with caution |
| **< 0.20** | any | ❌ Phase 3 needed |

---

## ⏱️ **Time Estimates**

### **Minimum Path** (If Step 3 works well):
```
Step 1: Data leakage check     30 min
Step 2: Augmentation           15 min
Step 3: Train v2.1.0           30 min
Step 5: Walk-forward           30 min
────────────────────────────────────────
Total:                         1h 45min
```

### **With Tuning**:
```
Step 1: Data leakage check     30 min
Step 2: Augmentation           15 min
Step 3: Train v2.1.0           30 min
Step 4: Hyperparameter tuning  2 hours
Step 3b: Retrain with tuning   30 min
Step 5: Walk-forward           30 min
────────────────────────────────────────
Total:                         4h 15min
```

---

## ✅ **Success Checklist**

Phase 2 is complete when:

- [ ] Data leakage checked and clean
- [ ] Augmented dataset created (107 samples)
- [ ] Model v2.1.0 trained
- [ ] Test R² > 0.25 (minimum)
- [ ] Walk-forward validation performed
- [ ] Mean R² > 0.30 (target)
- [ ] Results documented
- [ ] Deployment decision made

---

## 🎯 **Expected Outcomes**

### **Realistic Expectations:**

| Metric | v2.0.0 | Phase 2 Target | Stretch Goal |
|--------|--------|----------------|--------------|
| Test R² | -0.08 | +0.30-0.35 | +0.40-0.45 |
| Test RMSE | 1,441 | 1,250-1,350 | 1,100-1,250 |
| MAPE | 3.3% | 3.0-3.5% | 2.8-3.2% |
| Status | Not ready | Staging ready | Production ready |

---

## 🚨 **Common Issues & Solutions**

### **Issue 1**: Augmentation doesn't improve performance
**Solution**: 
- Check augmentation quality (statistics)
- Try different noise levels (1%, 3%, 5%)
- Verify no data leakage

### **Issue 2**: R² still negative after augmentation
**Solution**:
- Data quality issue likely
- Review feature engineering
- May need Phase 3 (more real data)

### **Issue 3**: Walk-forward shows high variance
**Solution**:
- Expected with small data
- Use mean performance for decisions
- Monitor closely in production

### **Issue 4**: Scripts fail to find files
**Solution**:
```bash
# Ensure you're in project root
cd "C:\projects\hackathon-2025-Studio Revenue Simulator\take-2"

# Check files exist
dir data\processed\studio_data_engineered.csv
dir training\*.py
```

---

## 📞 **Need Help?**

### **If stuck on Step 1 (Data Leakage)**:
- Review: `src/features/feature_engineer.py`
- Check: Rolling windows exclude current period
- Ensure: Lagged features use `.shift(1)`

### **If stuck on Step 2 (Augmentation)**:
- Check: Original data exists
- Verify: Statistics look reasonable
- Try: Lower noise level (0.01 instead of 0.02)

### **If stuck on Step 3 (Training)**:
- Check: Augmented data was created
- Verify: 107 samples in training
- Try: Different random seed

### **If performance is poor**:
- Accept: Small data = limited performance
- Consider: Phase 3 (collect more data)
- Timeline: 2-6 months for production-grade

---

## 🎬 **Start Now!**

### **Copy-paste this to execute:**

```bash
# Navigate to project
cd "C:\projects\hackathon-2025-Studio Revenue Simulator\take-2"

# Step 1: Check data leakage (30 min)
python training/check_data_leakage.py

# Step 2: Augment data (15 min)
python training/data_augmentation.py

# Step 3: Train v2.1.0 (30 min)
python training/train_improved_model_v2.1.py

# Step 5: Walk-forward validation (30 min)
python training/walk_forward_validation.py

# Review results
notepad reports\audit\improved_model_results_v2.1.0.json
notepad reports\audit\walk_forward_validation_v2.1.0.json
```

### **Total time**: ~2 hours for complete Phase 2!

---

## 🎓 **What You'll Learn**

1. **Data augmentation** for small datasets
2. **Walk-forward validation** for time series
3. **Hyperparameter optimization** techniques
4. **Production deployment** decision-making
5. **Model monitoring** best practices

---

## 🏁 **After Phase 2**

### **If Successful (R² > 0.30)**:
1. Document findings
2. Plan staging deployment
3. Set up monitoring
4. Create deployment guide
5. Schedule go-live

### **If Marginal (R² = 0.20-0.30)**:
1. Deploy with human oversight
2. Collect more data in parallel
3. Plan Phase 3
4. Set 3-6 month timeline

### **If Unsuccessful (R² < 0.20)**:
1. Accept data limitation
2. Focus on data collection (Phase 3)
3. Use predictions as guidance only
4. Set realistic 6-month timeline

---

**Ready? Let's execute Phase 2! 🚀**

Run the first command and let me know the results!

---

**Created**: November 5, 2025  
**Version**: Phase 2 Quick Start Guide  
**Est. Time**: 2-4 hours  
**Success Rate**: High (if data quality is good)

