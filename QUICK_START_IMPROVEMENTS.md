# Quick Start Guide - Model Improvements
**Get Better Results in 30 Minutes**

---

## 🚀 Immediate Actions (Do This First)

### Step 1: Check for Data Leakage (5 minutes)

```bash
python training/check_data_leakage.py
```

**What it does**: Detects if your features contain future information that would make predictions unrealistically good in training but fail in production.

**Output**: 
- `reports/audit/data_leakage_report.txt` - Detailed findings
- Console summary of suspicious features

**Action**: Review the report and fix any identified issues in `src/features/feature_engineer.py`

---

### Step 2: Train Improved Model (10 minutes)

```bash
python training/train_improved_model.py
```

**What it does**: 
- Reduces features from 41 → 15 (top performers only)
- Uses simpler, regularized models (Ridge, Lasso, ElasticNet, GBM)
- Performs 5-fold cross-validation
- Evaluates on test set

**Output**:
- `data/models/best_model_v2.0.0.pkl` - New improved model
- `reports/audit/improved_model_results_v2.0.0.json` - Performance metrics
- `reports/figures/model_comparison_v2.png` - Visualization
- Console output showing improvements

**Expected Results**:
- Test R² improvement: -1.41 → 0.20-0.40 (huge improvement!)
- Lower train/test gap (better generalization)
- More reliable predictions

---

### Step 3: Compare Model Versions (5 minutes)

```bash
python training/compare_model_versions.py
```

**What it does**: 
- Compares v1.0.0 (current) vs v2.0.0 (improved)
- Visualizes improvements
- Provides deployment recommendation

**Output**:
- `reports/figures/version_comparison.png` - Visual comparison
- `reports/audit/version_comparison_report.txt` - Detailed report
- Console summary with recommendation

---

### Step 4: Data Augmentation (Optional - 10 minutes)

```bash
python training/data_augmentation.py
```

**What it does**: 
- Increases training data by 50% using synthetic samples
- Maintains statistical properties
- Only augments training data (not validation/test)

**Output**:
- `data/processed/studio_data_augmented.csv` - Augmented dataset
- `reports/audit/augmentation_statistics.csv` - Quality check

**When to use**: If Step 2 still shows overfitting

---

## 📊 Expected Improvements Summary

### Before (v1.0.0):
```
Test Performance:
├─ R² Score:     -1.41  ❌ (worse than baseline)
├─ RMSE:         2,156  ❌ (high error)
├─ MAPE:         5.1%   ❌ (unreliable)
└─ Overfitting:  193%   ❌ (severe)
```

### After Phase 1 (v2.0.0):
```
Test Performance:
├─ R² Score:     ~0.30  ✅ (baseline achieved)
├─ RMSE:         ~1,500 ✅ (30% reduction)
├─ MAPE:         ~4.0%  ✅ (20% improvement)
└─ Overfitting:  ~40%   ✅ (75% reduction)
```

### After Phase 1 + Augmentation (v2.1.0):
```
Test Performance:
├─ R² Score:     ~0.45  ✅✅ (good)
├─ RMSE:         ~1,300 ✅✅ (40% reduction)
├─ MAPE:         ~3.5%  ✅✅ (30% improvement)
└─ Overfitting:  ~30%   ✅✅ (85% reduction)
```

---

## 🔧 Troubleshooting

### Issue: ImportError or Module Not Found

**Solution**: Install required packages
```bash
pip install -r requirements.txt
```

---

### Issue: File Not Found Error

**Solution**: Make sure you're in the project root directory
```bash
cd path/to/hackathon-2025-Studio\ Revenue\ Simulator/take-2
python training/train_improved_model.py
```

---

### Issue: "No metrics found for v1.0.0"

**Solution**: The comparison script needs both versions. Skip Step 3 if you haven't trained v2.0.0 yet.

---

### Issue: Model performance still poor

**Checklist**:
1. ✅ Did you check for data leakage first?
2. ✅ Did you fix any identified leakage issues?
3. ✅ Try data augmentation (Step 4)
4. ✅ Check if you need more real data

---

## 📋 Complete Workflow

```bash
# 1. Check data quality
python training/check_data_leakage.py

# 2. Fix any issues found (manual step - review feature_engineer.py)

# 3. Train improved model
python training/train_improved_model.py

# 4. Compare with old model
python training/compare_model_versions.py

# 5. (Optional) If still overfitting, augment data
python training/data_augmentation.py

# 6. (Optional) Retrain with augmented data
# Edit train_improved_model.py line 53 to use:
# df = pd.read_csv('data/processed/studio_data_augmented.csv')
python training/train_improved_model.py

# 7. Final comparison
python training/compare_model_versions.py
```

---

## 📈 Monitoring Your Progress

### Metrics to Watch:

1. **R² Score** (Higher is better):
   - Bad: < 0 (worse than baseline)
   - Acceptable: 0.3 - 0.5
   - Good: 0.5 - 0.7
   - Excellent: > 0.7

2. **RMSE** (Lower is better):
   - Current: 2,156
   - Target: < 1,500
   - Good: < 1,200
   - Excellent: < 1,000

3. **Train/Test Gap** (Lower is better):
   - Current: 193% (severe overfitting)
   - Acceptable: < 50%
   - Good: < 30%
   - Excellent: < 20%

---

## ✅ Success Checklist

After running the scripts, you should have:

- [ ] Data leakage report reviewed
- [ ] v2.0.0 model trained
- [ ] Test R² improved from negative to positive
- [ ] RMSE reduced by 20-40%
- [ ] Train/test gap reduced significantly
- [ ] Comparison report generated
- [ ] Deployment decision made

---

## 🎯 Decision Framework

### Should I Deploy v2.0.0?

**Deploy if**:
- ✅ Test R² > 0.40
- ✅ RMSE < 1,400
- ✅ Train/test gap < 50%
- ✅ No data leakage found

**Don't deploy if**:
- ❌ Test R² still negative
- ❌ RMSE > 1,800
- ❌ Data leakage detected but not fixed
- ❌ Train/test gap > 70%

**Need more work if**:
- ⚠️  Test R² between 0.20-0.40
- ⚠️  RMSE between 1,400-1,800
- ⚠️  Train/test gap between 50-70%

**Action**: Proceed to Phase 2 (data augmentation + collection)

---

## 📞 Need Help?

### Check These Resources:

1. **Technical Details**: See `MODEL_IMPROVEMENT_PLAN.md`
2. **Stakeholder Summary**: See `EXECUTIVE_SUMMARY_MODEL_EVALUATION.md`
3. **Training Logs**: Check console output during training
4. **Error Messages**: Most include helpful suggestions

### Common Questions:

**Q: How long does each step take?**
A: 
- Data leakage check: ~5 minutes
- Train improved model: ~10 minutes
- Compare versions: ~5 minutes
- Data augmentation: ~10 minutes

**Q: Will I lose my old model?**
A: No, v1.0.0 is preserved. New model saved as v2.0.0

**Q: Can I use this in production now?**
A: Only if it passes the success criteria above. Check the comparison report.

**Q: What if improvements aren't enough?**
A: Proceed to Phase 2 in the improvement plan (data collection + augmentation)

**Q: Do I need to retrain from scratch?**
A: No, the new scripts handle everything. Just run them.

---

## 🎓 What You're Learning

This quick start teaches you:

1. **Overfitting Detection**: How to spot when models memorize vs learn
2. **Regularization**: How to prevent overfitting with simpler models
3. **Feature Selection**: How to reduce dimensionality effectively
4. **Cross-Validation**: How to get reliable performance estimates
5. **Data Leakage**: How to ensure temporal integrity
6. **Model Comparison**: How to validate improvements objectively

These are fundamental ML engineering skills!

---

## 🚦 Next Steps Based on Results

### If R² > 0.40 (Good Results):
1. ✅ Deploy to staging environment
2. ✅ Monitor performance on real data
3. ✅ Set up A/B test vs current system
4. ✅ Plan gradual rollout

### If R² = 0.20-0.40 (Acceptable):
1. ⚠️  Run data augmentation (Step 4)
2. ⚠️  Consider collecting more real data
3. ⚠️  Review feature engineering for improvements
4. ⚠️  Test in controlled pilot

### If R² < 0.20 (Needs Work):
1. ❌ Review data leakage report carefully
2. ❌ Audit feature engineering code manually
3. ❌ Fix any temporal integrity issues
4. ❌ Collect more real data (priority)
5. ❌ Retrain after fixes

---

## 📅 Time Investment

**Total time required**: 30 minutes - 2 hours

**Breakdown**:
- Automated scripts: 30 minutes
- Manual review/fixes: 30 minutes - 1 hour (if leakage found)
- Retesting: 30 minutes

**ROI**: Transform a failing model into a functional one in < 2 hours!

---

*Remember: ML is iterative. Don't expect perfection on first try. Each iteration teaches you something and moves you closer to production-ready.*

**Good luck! 🚀**

