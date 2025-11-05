# Studio Revenue Simulator - Model Improvement Plan
**As Evaluated by an Experienced ML Engineer**

---

## 📋 Executive Summary

**Current State**: Model suffers from severe overfitting with negative R² scores on validation/test sets  
**Root Cause**: Model complexity far exceeds available data (41 features, 3 models, only 63 training samples)  
**Impact**: Production deployment would result in **poor predictions and unreliable forecasts**  
**Recommendation**: Follow phased improvement approach below

---

## 🔴 Critical Issues Identified

### 1. **Severe Overfitting**
```
Dataset      R² (Revenue)   RMSE      Status
─────────────────────────────────────────────
Train        0.91           734       ✓ Good
Validation   -1.19          1469      ✗ FAILED
Test         -1.41          2156      ✗ FAILED
```

**Analysis**: 
- Negative R² indicates model is **worse than predicting the mean**
- 3x increase in error from train to test
- Model has memorized training data, not learned patterns

### 2. **Insufficient Data**
```
Samples: 80 total (63 train, 8 val, 9 test)
Features: 41 engineered features
Ratio: 1.5 samples per feature

Industry Standard: 10-20+ samples per feature
Your Status: ✗ CRITICAL
```

### 3. **Model Complexity**
```
Current Configuration:
├─ XGBoost:       300 trees × depth 5  = 1,500 leaf nodes
├─ LightGBM:      300 trees × depth 5  = 1,500 leaf nodes  
└─ Random Forest: 200 trees × depth 10 = 2,000 leaf nodes
   TOTAL CAPACITY: 5,000+ decision boundaries for 63 samples!
```

### 4. **Unreliable Evaluation**
- Test set: Only 9 samples (statistically insignificant)
- Confidence interval coverage: 33% (should be 95%)
- High variance in all metrics

### 5. **Potential Data Leakage**
Features that may contaminate predictions:
- `revenue_momentum` (16.6% importance) - May include future revenue
- `estimated_ltv` (10.9%) - Calculated from future behavior
- `3m_avg_revenue` - Rolling windows may leak

---

## 🎯 Improvement Roadmap

### **PHASE 1: Emergency Fixes (Week 1)**
*Goal: Get model to baseline performance*

#### 1.1 ✅ Simplify Model Architecture

**Current Problem**: 3 complex ensemble models for tiny dataset

**Solution A - Linear Models (RECOMMENDED):**
```python
# Start with regularized linear models
models = {
    'ridge': Ridge(alpha=10.0),           # L2 regularization
    'lasso': Lasso(alpha=1.0),             # L1 + feature selection
    'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5)
}
```

**Benefits:**
- Fewer parameters than tree models
- Built-in regularization
- Interpretable coefficients
- Less prone to overfitting

**Solution B - Simplified Trees (ALTERNATIVE):**
```python
# If you need non-linear models
xgboost_config = {
    'n_estimators': 30,      # Down from 300
    'max_depth': 2,          # Down from 5
    'min_child_weight': 10,  # Up from 1
    'reg_lambda': 5.0,       # Strong L2
    'subsample': 0.6,        # Use 60% of data per tree
    'colsample_bytree': 0.6  # Use 60% of features
}
```

#### 1.2 ✅ Feature Selection

**Reduce from 41 → 15 features** using importance ranking:

```python
# Keep top features only:
top_features = [
    'revenue_momentum',      # 16.6%
    'estimated_ltv',         # 10.9%
    'membership_revenue',    # 10.8%
    'retention_x_ticket',    # 10.7%
    'avg_ticket_price',      # 8.9%
    'mom_member_growth',     # 6.2%
    'total_revenue',         # 5.6%
    '3m_avg_attendance',     # 4.0%
    'prev_month_revenue',    # 4.0%
    '3m_avg_revenue',        # 3.2%
    'class_utilization',     # 3.0%
    '3m_avg_retention',      # 1.5%
    'month_cos',             # 1.4%
    'class_attendance_rate', # 1.3%
    'retention_rate'         # 1.3%
]
```

**Why**: 
- Better samples-to-features ratio (63:15 = 4.2:1)
- Remove redundant/low-importance features
- Reduce model complexity

#### 1.3 ✅ Cross-Validation Instead of Fixed Split

**Current**: 63 train / 8 val / 9 test (tiny test set)

**Better**: 5-Fold Cross-Validation
```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')

print(f"Mean R²: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
```

**Benefits:**
- Use all data for both training and evaluation
- More reliable performance estimates
- Reduces evaluation variance

#### 1.4 ✅ Check for Data Leakage

**Audit these features:**
```python
suspicious_features = [
    'revenue_momentum',    # Verify: Uses only past data?
    'estimated_ltv',       # Verify: Not calculated from future?
    '3m_avg_revenue',      # Verify: Doesn't include current month?
    'prev_month_revenue'   # Verify: Actually previous month?
]
```

**Action**: Review `feature_engineer.py` line-by-line to ensure temporal integrity

---

### **PHASE 2: Data Quality (Week 2)**
*Goal: Improve data quantity and quality*

#### 2.1 📊 Data Augmentation

**Problem**: Only 80 samples is fundamentally insufficient

**Solutions:**

**A. Synthetic Data Generation (Conservative)**
```python
from sklearn.utils import resample
from scipy.stats import norm

def augment_data(X, y, factor=1.5):
    """Add synthetic samples with controlled noise"""
    n_synthetic = int(len(X) * factor)
    
    # Bootstrap sampling
    X_resampled, y_resampled = resample(X, y, 
                                         n_samples=n_synthetic,
                                         random_state=42)
    
    # Add small Gaussian noise (2% std)
    noise_X = norm(0, 0.02 * X.std(axis=0)).rvs(X_resampled.shape)
    noise_y = norm(0, 0.02 * y.std(axis=0)).rvs(y_resampled.shape)
    
    X_augmented = X_resampled + noise_X
    y_augmented = y_resampled + noise_y
    
    return np.vstack([X, X_augmented]), np.vstack([y, y_augmented])
```

**B. SMOTE for Regression** (Advanced)
```python
from imblearn.over_sampling import SMOTE

# Adapt SMOTE for continuous targets
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_binned)
```

**C. Collect More Real Data** (BEST)
- Extend time series: 6+ years of data
- Multi-studio data: Collect from similar studios
- External data: Industry benchmarks, economic indicators

#### 2.2 🔍 Feature Engineering Improvements

**Remove Collinear Features:**
```python
# Calculate correlation matrix
correlation_matrix = df[features].corr()

# Remove features with correlation > 0.85
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.85:
            high_corr_pairs.append(
                (correlation_matrix.columns[i], 
                 correlation_matrix.columns[j],
                 correlation_matrix.iloc[i, j])
            )
```

**Add Domain-Specific Features:**
```python
# Business cycle indicators
df['quarter'] = df['month_index'] // 3
df['is_quarter_start'] = (df['month_index'] % 3 == 0).astype(int)

# Capacity utilization
df['capacity_utilization'] = (
    df['total_class_attendance'] / 
    (df['total_classes_held'] * df['class_attendance_rate'].mean())
)

# Member lifecycle stage
df['member_maturity'] = df.groupby('studio_id')['month_index'].rank()
```

#### 2.3 📈 Temporal Validation

**Current**: Random split (may have temporal leakage)

**Better**: Time-based split
```python
# Train on earlier data, test on later data
cutoff_date = '2024-01-01'
train_df = df[df['month_year'] < cutoff_date]
test_df = df[df['month_year'] >= cutoff_date]
```

**Best**: Walk-Forward Validation
```python
def walk_forward_validation(df, n_splits=5):
    """Simulate production forecasting"""
    results = []
    
    for i in range(n_splits):
        train_end = len(df) * (i + 1) // (n_splits + 1)
        test_end = len(df) * (i + 2) // (n_splits + 1)
        
        train_data = df[:train_end]
        test_data = df[train_end:test_end]
        
        model.fit(train_data)
        score = model.score(test_data)
        results.append(score)
    
    return results
```

---

### **PHASE 3: Advanced Techniques (Week 3-4)**
*Goal: Push model performance further*

#### 3.1 🧠 Model Architecture Exploration

**Try Simpler Models First:**
```python
# 1. Linear baseline
baseline_model = Ridge(alpha=1.0)

# 2. Gradient Boosting with strong regularization
gbm_model = GradientBoostingRegressor(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.05,
    subsample=0.7,
    min_samples_split=10,
    min_samples_leaf=5
)

# 3. Support Vector Regression
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# 4. Neural Network (if you get more data)
from sklearn.neural_network import MLPRegressor
nn_model = MLPRegressor(
    hidden_layer_sizes=(20, 10),
    alpha=1.0,  # L2 regularization
    early_stopping=True,
    validation_fraction=0.2
)
```

#### 3.2 📊 Ensemble Methods (Only After Above Fixes)

**Stacking with Cross-Validation:**
```python
from sklearn.ensemble import StackingRegressor

base_models = [
    ('ridge', Ridge(alpha=10.0)),
    ('lasso', Lasso(alpha=1.0)),
    ('gbm', GradientBoostingRegressor(max_depth=2))
]

meta_model = Ridge(alpha=1.0)

stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5  # Use CV to avoid overfitting
)
```

#### 3.3 🎯 Bayesian Optimization for Hyperparameters

**Only do this after data quality improvements:**
```python
from skopt import BayesSearchCV

param_space = {
    'alpha': (0.01, 100.0, 'log-uniform'),
    'max_depth': (1, 3),
    'n_estimators': (10, 100)
}

opt = BayesSearchCV(
    model,
    param_space,
    n_iter=30,
    cv=5,
    n_jobs=-1,
    scoring='neg_root_mean_squared_error'
)
```

#### 3.4 📉 Multi-Task Learning

**Current**: Train 5 separate models for 5 targets

**Better**: Share representations across tasks
```python
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge

# Shared ridge model across all targets
multi_task_model = MultiOutputRegressor(
    Ridge(alpha=10.0),
    n_jobs=-1
)

# Or use neural network with shared layers
import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(15, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.revenue_head = nn.Linear(32, 3)
        self.member_head = nn.Linear(32, 1)
        self.retention_head = nn.Linear(32, 1)
```

---

### **PHASE 4: Production Readiness (Week 4)**
*Goal: Deploy reliable model*

#### 4.1 ✅ Model Monitoring

```python
class ModelMonitor:
    def __init__(self, model, thresholds):
        self.model = model
        self.thresholds = thresholds
        self.predictions_log = []
    
    def predict_with_monitoring(self, X):
        pred = self.model.predict(X)
        
        # Check for drift
        if self._detect_drift(X):
            logging.warning("Input drift detected!")
        
        # Check prediction quality
        if self._check_prediction_quality(pred):
            logging.warning("Unusual predictions detected!")
        
        self.predictions_log.append(pred)
        return pred
    
    def _detect_drift(self, X):
        """Detect if input distribution has shifted"""
        # Compare to training distribution
        pass
    
    def _check_prediction_quality(self, pred):
        """Check if predictions are reasonable"""
        # Check for outliers
        pass
```

#### 4.2 📊 Uncertainty Quantification

```python
# Quantile Regression for prediction intervals
from sklearn.ensemble import GradientBoostingRegressor

models = {
    'lower': GradientBoostingRegressor(loss='quantile', alpha=0.05),
    'median': GradientBoostingRegressor(loss='quantile', alpha=0.50),
    'upper': GradientBoostingRegressor(loss='quantile', alpha=0.95)
}

for name, model in models.items():
    model.fit(X_train, y_train)

# Get prediction intervals
lower = models['lower'].predict(X_test)
median = models['median'].predict(X_test)
upper = models['upper'].predict(X_test)
```

#### 4.3 🎯 A/B Testing Framework

```python
class ModelComparison:
    def __init__(self, model_v1, model_v2):
        self.v1 = model_v1
        self.v2 = model_v2
        self.results = {'v1': [], 'v2': []}
    
    def compare(self, X_test, y_test):
        """Compare two model versions"""
        pred_v1 = self.v1.predict(X_test)
        pred_v2 = self.v2.predict(X_test)
        
        rmse_v1 = np.sqrt(mean_squared_error(y_test, pred_v1))
        rmse_v2 = np.sqrt(mean_squared_error(y_test, pred_v2))
        
        # Statistical significance test
        from scipy.stats import ttest_rel
        t_stat, p_value = ttest_rel(
            np.abs(y_test - pred_v1),
            np.abs(y_test - pred_v2)
        )
        
        return {
            'v1_rmse': rmse_v1,
            'v2_rmse': rmse_v2,
            'improvement': (rmse_v1 - rmse_v2) / rmse_v1 * 100,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
```

---

## 📈 Expected Improvements

### Realistic Targets After Improvements:

| Metric | Current | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|---------|
| Test R² | **-1.41** | 0.30 | 0.50 | 0.65 |
| Test RMSE | **2156** | 1500 | 1200 | 1000 |
| MAPE | **5.1%** | 4.5% | 3.8% | 3.2% |
| CI Coverage | **33%** | 70% | 85% | 92% |

### Timeline:
- **Week 1**: Emergency fixes → Baseline performance
- **Week 2**: Data quality → Acceptable performance  
- **Week 3**: Advanced techniques → Good performance
- **Week 4**: Production ready → Reliable deployment

---

## 🔧 Implementation Scripts

I'll create the following scripts for you:

1. ✅ `training/train_improved_model.py` - Simplified training pipeline
2. ✅ `training/feature_selection.py` - Select top K features
3. ✅ `training/cross_validation.py` - CV evaluation
4. ✅ `training/data_augmentation.py` - Synthetic data generation
5. ✅ `training/check_data_leakage.py` - Temporal integrity audit

---

## 📚 Key Takeaways

### Do This:
✅ Simplify model (use Ridge/Lasso first)  
✅ Reduce features (41 → 15)  
✅ Use cross-validation  
✅ Check for data leakage  
✅ Collect more real data  
✅ Monitor model performance  

### Don't Do This:
❌ Complex ensembles on small data  
❌ Hyperparameter tuning without enough data  
❌ Trust metrics on 9-sample test set  
❌ Deploy current model to production  
❌ Ignore warning signs (negative R²)  

---

## 🎯 Success Metrics

**Minimum Acceptable Performance (Production Deployment):**
- Test R² > 0.50
- Test MAPE < 4%
- 95% CI coverage > 85%
- Directional accuracy > 70%
- Stable performance across CV folds

**Current Status**: ❌ Not production ready  
**After Phase 1**: ⚠️ Baseline achieved  
**After Phase 2**: ✅ Production candidate  
**After Phase 3**: ✅✅ High confidence deployment  

---

## 📞 Next Steps

1. Review this plan with stakeholders
2. Prioritize phases based on timeline/resources
3. Run implementation scripts (I'll create these)
4. Monitor improvements at each phase
5. Iterate based on results

**Remember**: ML success is 20% algorithms, 80% data quality!

---

*Generated by: Experienced ML Engineer Evaluation*  
*Date: 2025-11-05*  
*Model Version: v1.0.0*

