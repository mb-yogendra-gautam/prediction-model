# Step-by-Step Guide: Model Training, Validation, Testing & FastAPI Deployment

## Table of Contents

1. [Project Setup](#1-project-setup)
2. [Data Generation & Preparation](#2-data-generation--preparation)
3. [Feature Engineering](#3-feature-engineering)
4. [Model Training](#4-model-training)
5. [Model Validation](#5-model-validation)
6. [Model Testing & Evaluation](#6-model-testing--evaluation)
7. [Model Serialization & Versioning](#7-model-serialization--versioning)
8. [FastAPI Backend Development](#8-fastapi-backend-development)
9. [API Testing & Validation](#9-api-testing--validation)
10. [Deployment](#10-deployment)
11. [Monitoring & Maintenance](#11-monitoring--maintenance)

---

## 1. Project Setup

### 1.1 Create Project Structure

```bash
studio-revenue-simulator/
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_generator.py
│   │   └── data_validator.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_engineer.py
│   │   └── feature_store.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── forward_model.py
│   │   ├── inverse_optimizer.py
│   │   └── model_registry.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── predictions.py
│   │   │   ├── scenarios.py
│   │   │   └── insights.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── requests.py
│   │   │   └── responses.py
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── prediction_service.py
│   │       ├── optimization_service.py
│   │       └── cache_service.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── connection.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       └── config.py
├── training/
│   ├── train_forward_model.py
│   ├── evaluate_model.py
│   └── hyperparameter_tuning.py
├── tests/
│   ├── test_data_generator.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_evaluation.ipynb
├── config/
│   ├── config.yaml
│   └── model_config.yaml
├── requirements.txt
├── .env.example
└── README.md
```

### 1.2 Install Dependencies

**requirements.txt:**

```txt
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Machine Learning
scikit-learn==1.3.2
xgboost==2.0.2
lightgbm==4.1.0
numpy==1.24.3
pandas==2.1.3
scipy==1.11.4

# Database
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
asyncpg==0.29.0

# Caching
redis==5.0.1
hiredis==2.2.3

# ML Tracking
mlflow==2.9.2

# API & Async
httpx==0.25.2
python-multipart==0.0.6

# Environment & Config
python-dotenv==1.0.0
pyyaml==6.0.1

# OpenAI
openai==1.6.1

# Utilities
joblib==1.3.2
python-dateutil==2.8.2

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
```

### 1.3 Environment Configuration

**.env.example:**

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/studio_simulator
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_TTL=3600

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=studio_revenue_predictor

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
MODEL_VERSION=1.0.0
MODEL_PATH=data/models/
FEATURE_SCALER_PATH=data/models/scalers/

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=false

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/api.log
```

---

## 2. Data Generation & Preparation

### 2.1 Generate Synthetic Data

**src/data/data_generator.py:**

```python
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class StudioDataGenerator:
    """Generate realistic synthetic fitness studio data"""

    def __init__(self, num_years: int = 5, studio_type: str = 'Yoga', seed: int = 42):
        self.num_years = num_years
        self.num_months = num_years * 12
        self.studio_type = studio_type
        self.seed = seed
        np.random.seed(seed)

    def generate(self) -> pd.DataFrame:
        """Generate complete dataset with all metrics"""
        logger.info(f"Generating {self.num_months} months of data for {self.studio_type} studio")

        months = pd.date_range('2019-01-01', periods=self.num_months, freq='MS')

        # Base values
        base_members = 200
        base_retention = 0.75
        base_ticket = 150.0
        base_classes = 120
        base_staff = 10

        data = []
        current_members = base_members

        for i, month in enumerate(months):
            month_idx = month.month
            year_idx = i // 12

            # Apply seasonality
            seasonality = self._get_seasonality_factor(month_idx)

            # Apply growth phase
            growth_factor = self._get_growth_factor(year_idx)

            # Calculate metrics with realistic noise
            retention_rate = self._calculate_retention(
                base_retention, seasonality, month_idx
            )

            new_members = self._calculate_new_members(
                seasonality, growth_factor, month_idx
            )

            churned_members = int(current_members * (1 - retention_rate))
            current_members = max(
                current_members - churned_members + new_members,
                100  # Floor
            )

            avg_ticket = base_ticket * (1.005 ** i) * (1 + np.random.normal(0, 0.02))

            class_attendance_rate = self._calculate_attendance_rate(seasonality)

            total_classes = int(
                base_classes * growth_factor * (1 + np.random.normal(0, 0.05))
            )

            total_attendance = int(total_classes * 20 * class_attendance_rate)

            staff_count = max(5, int(base_staff * (current_members / base_members)))
            staff_utilization = np.clip(
                0.80 * (1 + np.random.normal(0, 0.05)),
                0.65, 0.95
            )

            upsell_rate = np.clip(
                0.25 * (1 + np.random.normal(0, 0.1)),
                0.1, 0.45
            )

            # Calculate revenue components
            membership_revenue = current_members * avg_ticket
            class_pack_revenue = total_attendance * 0.2 * 15  # 20% drop-ins at $15
            retail_revenue = current_members * 10 * 0.3  # 30% buy retail avg $10
            upsell_revenue = current_members * upsell_rate * 50  # Avg $50 upsell

            total_revenue = (
                membership_revenue +
                class_pack_revenue +
                retail_revenue +
                upsell_revenue
            )

            # Add outliers for realism (COVID impact, promotions)
            if i == 36:  # COVID impact - month 36
                total_revenue *= 0.6
                current_members = int(current_members * 0.85)
                retention_rate *= 0.8

            data.append({
                'month_year': month,
                'month_index': month_idx,
                'year_index': year_idx,
                'total_members': current_members,
                'new_members': new_members,
                'churned_members': churned_members,
                'retention_rate': retention_rate,
                'avg_ticket_price': avg_ticket,
                'total_classes_held': total_classes,
                'total_class_attendance': total_attendance,
                'class_attendance_rate': class_attendance_rate,
                'staff_count': staff_count,
                'staff_utilization_rate': staff_utilization,
                'total_revenue': total_revenue,
                'membership_revenue': membership_revenue,
                'class_pack_revenue': class_pack_revenue,
                'retail_revenue': retail_revenue,
                'upsell_rate': upsell_rate,
            })

        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} rows of data")

        return df

    def _get_seasonality_factor(self, month_idx: int) -> dict:
        """Get seasonality multipliers for given month"""
        return {
            'january_boost': 1.25 if month_idx == 1 else 1.0,
            'summer_dip': 0.90 if month_idx in [6, 7, 8] else 1.0,
            'fall_recovery': 1.10 if month_idx == 9 else 1.0,
        }

    def _get_growth_factor(self, year_idx: int) -> float:
        """Get growth factor based on business maturity"""
        if year_idx <= 1:
            return 1.012  # 1.2% monthly growth (Year 1-2)
        elif year_idx <= 3:
            return 1.005  # 0.5% monthly growth (Year 3-4)
        else:
            return 1.002  # 0.2% monthly growth (Year 5)

    def _calculate_retention(self, base_retention: float,
                            seasonality: dict, month_idx: int) -> float:
        """Calculate retention rate with seasonality and noise"""
        retention = base_retention * seasonality['summer_dip'] * \
                   (1 + np.random.normal(0, 0.03))
        return np.clip(retention, 0.6, 0.95)

    def _calculate_new_members(self, seasonality: dict,
                              growth_factor: float, month_idx: int) -> int:
        """Calculate new members with seasonality"""
        base_new = 25
        return int(
            base_new *
            seasonality['january_boost'] *
            seasonality['fall_recovery'] *
            growth_factor *
            (1 + np.random.normal(0, 0.15))
        )

    def _calculate_attendance_rate(self, seasonality: dict) -> float:
        """Calculate class attendance rate"""
        base_rate = 0.70
        rate = base_rate * seasonality['summer_dip'] * \
               (1 + np.random.normal(0, 0.05))
        return np.clip(rate, 0.5, 0.9)


class DataPreprocessor:
    """Prepare data for model training"""

    @staticmethod
    def add_target_variables(df: pd.DataFrame) -> pd.DataFrame:
        """Add future revenue targets for supervised learning"""
        logger.info("Adding target variables (future revenue predictions)")

        df = df.copy()

        # Create future targets
        df['revenue_month_1'] = df['total_revenue'].shift(-1)
        df['revenue_month_2'] = df['total_revenue'].shift(-2)
        df['revenue_month_3'] = df['total_revenue'].shift(-3)
        df['member_count_month_3'] = df['total_members'].shift(-3)
        df['retention_rate_month_3'] = df['retention_rate'].shift(-3)

        # Remove rows without targets
        df = df[:-3].copy()

        logger.info(f"Dataset size after adding targets: {len(df)} rows")
        return df

    @staticmethod
    def add_data_splits(df: pd.DataFrame,
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.1) -> pd.DataFrame:
        """Add train/val/test split labels"""
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        df = df.copy()
        df['split'] = 'train'
        df.loc[train_end:val_end, 'split'] = 'validation'
        df.loc[val_end:, 'split'] = 'test'

        logger.info(f"Data splits: Train={len(df[df['split']=='train'])}, "
                   f"Val={len(df[df['split']=='validation'])}, "
                   f"Test={len(df[df['split']=='test'])}")

        return df

    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Tuple[bool, list]:
        """Validate data quality and return issues"""
        issues = []

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            issues.append(f"Missing values found: {missing[missing > 0].to_dict()}")

        # Check for negative values where they shouldn't exist
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (df[col] < 0).any():
                issues.append(f"Negative values found in {col}")

        # Check revenue consistency
        calculated_revenue = (
            df['membership_revenue'] +
            df['class_pack_revenue'] +
            df['retail_revenue']
        )
        revenue_diff = np.abs(df['total_revenue'] - calculated_revenue)
        if (revenue_diff > 100).any():
            issues.append("Revenue calculation inconsistency detected")

        # Check retention rate bounds
        if ((df['retention_rate'] < 0.5) | (df['retention_rate'] > 1.0)).any():
            issues.append("Retention rate out of bounds")

        is_valid = len(issues) == 0
        return is_valid, issues


# Usage script
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate data
    generator = StudioDataGenerator(num_years=5, studio_type='Yoga')
    raw_data = generator.generate()

    # Preprocess
    preprocessor = DataPreprocessor()
    data = preprocessor.add_target_variables(raw_data)
    data = preprocessor.add_data_splits(data)

    # Validate
    is_valid, issues = preprocessor.validate_data_quality(data)
    if not is_valid:
        logger.warning(f"Data quality issues: {issues}")
    else:
        logger.info("Data quality validation passed")

    # Save
    data.to_csv('data/processed/studio_data_2019_2025.csv', index=False)
    logger.info("Data saved to data/processed/studio_data_2019_2025.csv")
```

### 2.2 Run Data Generation

```bash
# Create directories
mkdir -p data/raw data/processed data/models logs

# Generate data
python src/data/data_generator.py
```

---

## 3. Feature Engineering

### 3.1 Create Feature Engineering Pipeline

**src/features/feature_engineer.py:**

```python
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Transform raw features into model-ready features"""

    def __init__(self):
        self.feature_names = []

    def engineer_features(self, df: pd.DataFrame,
                         is_training: bool = True) -> pd.DataFrame:
        """
        Apply all feature engineering transformations

        Args:
            df: Raw data with lever values and historical context
            is_training: Whether this is for training (includes targets)

        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering")

        df = df.copy()

        # 1. Direct lever features (already in data)
        direct_features = [
            'retention_rate',
            'avg_ticket_price',
            'class_attendance_rate',
            'new_members',
            'staff_utilization_rate',
            'upsell_rate',
            'total_classes_held',
            'total_members'
        ]

        # 2. Temporal features
        df = self._add_temporal_features(df)

        # 3. Derived business metrics
        df = self._add_derived_features(df)

        # 4. Lagged features (historical context)
        df = self._add_lagged_features(df)

        # 5. Rolling statistics
        df = self._add_rolling_features(df)

        # 6. Interaction features
        df = self._add_interaction_features(df)

        # 7. Cyclical encoding for seasonality
        df = self._add_cyclical_features(df)

        # Remove rows with NaN from rolling/lagged features
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Removed {initial_rows - len(df)} rows due to NaN in engineered features")

        # Store feature names
        if is_training:
            target_cols = [
                'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
                'member_count_month_3', 'retention_rate_month_3'
            ]
            self.feature_names = [col for col in df.columns
                                 if col not in target_cols + ['month_year', 'split']]

        logger.info(f"Feature engineering complete. Total features: {len(self.feature_names)}")
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add month and year index features"""
        # Already have month_index and year_index from data generation
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived business metrics"""
        logger.info("Adding derived features")

        # Revenue per member
        df['revenue_per_member'] = df['total_revenue'] / df['total_members']

        # Member churn rate
        df['churn_rate'] = 1 - df['retention_rate']

        # Class utilization (assuming 20 person capacity per class)
        df['class_utilization'] = df['total_class_attendance'] / (df['total_classes_held'] * 20)

        # Staff per member ratio
        df['staff_per_member'] = df['staff_count'] / df['total_members']

        # LTV estimate
        df['estimated_ltv'] = df['avg_ticket_price'] * df['retention_rate'] * 12

        # Revenue mix percentages
        df['membership_revenue_pct'] = df['membership_revenue'] / df['total_revenue']
        df['class_pack_revenue_pct'] = df['class_pack_revenue'] / df['total_revenue']

        return df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features for historical context"""
        logger.info("Adding lagged features")

        # Previous month revenue
        df['prev_month_revenue'] = df['total_revenue'].shift(1)

        # Previous month members
        df['prev_month_members'] = df['total_members'].shift(1)

        # Month-over-month growth
        df['mom_revenue_growth'] = (
            (df['total_revenue'] - df['prev_month_revenue']) / df['prev_month_revenue']
        )

        df['mom_member_growth'] = (
            (df['total_members'] - df['prev_month_members']) / df['prev_month_members']
        )

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics"""
        logger.info("Adding rolling features")

        # 3-month rolling averages
        df['3m_avg_retention'] = df['retention_rate'].rolling(window=3).mean()
        df['3m_avg_revenue'] = df['total_revenue'].rolling(window=3).mean()
        df['3m_avg_attendance'] = df['class_attendance_rate'].rolling(window=3).mean()

        # 3-month rolling std (volatility)
        df['3m_std_revenue'] = df['total_revenue'].rolling(window=3).std()

        # Revenue momentum (EMA)
        df['revenue_momentum'] = df['total_revenue'].ewm(span=3).mean()

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features"""
        logger.info("Adding interaction features")

        # Retention × Ticket Price
        df['retention_x_ticket'] = df['retention_rate'] * df['avg_ticket_price']

        # Attendance × Classes
        df['attendance_x_classes'] = df['class_attendance_rate'] * df['total_classes_held']

        # Upsell × Members
        df['upsell_x_members'] = df['upsell_rate'] * df['total_members']

        # Staff utilization × Members
        df['staff_util_x_members'] = df['staff_utilization_rate'] * df['total_members']

        return df

    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical encoding for month seasonality"""
        logger.info("Adding cyclical features")

        # Sin/cos encoding for month (captures seasonality)
        df['month_sin'] = np.sin(2 * np.pi * df['month_index'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_index'] / 12)

        # Binary flags for key months
        df['is_january'] = (df['month_index'] == 1).astype(int)
        df['is_summer'] = df['month_index'].isin([6, 7, 8]).astype(int)
        df['is_fall'] = (df['month_index'] == 9).astype(int)

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return self.feature_names

    def transform_for_prediction(self, lever_inputs: Dict[str, float],
                                 historical_data: pd.DataFrame) -> np.ndarray:
        """
        Transform lever inputs into feature vector for prediction

        Args:
            lever_inputs: Dictionary of lever values
            historical_data: Recent historical data for context

        Returns:
            Feature vector ready for model prediction
        """
        # Create a single-row dataframe
        feature_dict = lever_inputs.copy()

        # Add historical context
        feature_dict['prev_month_revenue'] = historical_data.iloc[-1]['total_revenue']
        feature_dict['3m_avg_retention'] = historical_data.tail(3)['retention_rate'].mean()
        feature_dict['3m_avg_revenue'] = historical_data.tail(3)['total_revenue'].mean()
        feature_dict['revenue_momentum'] = historical_data['total_revenue'].ewm(span=3).mean().iloc[-1]

        # Add derived features
        feature_dict['revenue_per_member'] = (
            lever_inputs['avg_ticket_price'] * lever_inputs['retention_rate']
        )
        feature_dict['churn_rate'] = 1 - lever_inputs['retention_rate']

        # Add interaction features
        feature_dict['retention_x_ticket'] = (
            lever_inputs['retention_rate'] * lever_inputs['avg_ticket_price']
        )

        # Convert to array in correct order
        feature_vector = np.array([feature_dict[name] for name in self.feature_names])

        return feature_vector.reshape(1, -1)
```

---

## 4. Model Training

### 4.1 Build Training Pipeline

**training/train_forward_model.py:**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
import logging
from typing import Tuple, Dict
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForwardPredictionModel:
    """Ensemble model for forward prediction (levers → revenue)"""

    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.ensemble_weights = {}

    def prepare_data(self, df: pd.DataFrame) -> Tuple:
        """Prepare train/val/test splits"""
        logger.info("Preparing data splits")

        # Define feature columns
        target_cols = [
            'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
            'member_count_month_3', 'retention_rate_month_3'
        ]

        exclude_cols = target_cols + ['month_year', 'split', 'year_index']
        self.feature_names = [col for col in df.columns if col not in exclude_cols]

        logger.info(f"Using {len(self.feature_names)} features")

        # Split data
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'validation']
        test_df = df[df['split'] == 'test']

        # Extract features and targets
        X_train = train_df[self.feature_names].values
        y_train = train_df[target_cols].values

        X_val = val_df[self.feature_names].values
        y_val = val_df[target_cols].values

        X_test = test_df[self.feature_names].values
        y_test = test_df[target_cols].values

        # Scale features
        logger.info("Scaling features")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"Train shape: X={X_train_scaled.shape}, y={y_train.shape}")
        logger.info(f"Val shape: X={X_val_scaled.shape}, y={y_val.shape}")
        logger.info(f"Test shape: X={X_test_scaled.shape}, y={y_test.shape}")

        return (
            (X_train_scaled, y_train),
            (X_val_scaled, y_val),
            (X_test_scaled, y_test)
        )

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model for each target"""
        logger.info("Training XGBoost models")

        xgb_config = self.config['models']['xgboost']
        models = []

        for i in range(y_train.shape[1]):
            logger.info(f"Training XGBoost for target {i+1}")

            model = xgb.XGBRegressor(
                n_estimators=xgb_config['n_estimators'],
                max_depth=xgb_config['max_depth'],
                learning_rate=xgb_config['learning_rate'],
                subsample=xgb_config['subsample'],
                colsample_bytree=xgb_config['colsample_bytree'],
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )

            model.fit(
                X_train, y_train[:, i],
                eval_set=[(X_val, y_val[:, i])],
                early_stopping_rounds=20,
                verbose=False
            )

            models.append(model)

        self.models['xgboost'] = models
        logger.info("XGBoost training complete")

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model for each target"""
        logger.info("Training LightGBM models")

        lgb_config = self.config['models']['lightgbm']
        models = []

        for i in range(y_train.shape[1]):
            logger.info(f"Training LightGBM for target {i+1}")

            model = lgb.LGBMRegressor(
                n_estimators=lgb_config['n_estimators'],
                max_depth=lgb_config['max_depth'],
                learning_rate=lgb_config['learning_rate'],
                subsample=lgb_config['subsample'],
                colsample_bytree=lgb_config['colsample_bytree'],
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )

            model.fit(
                X_train, y_train[:, i],
                eval_set=[(X_val, y_val[:, i])],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )

            models.append(model)

        self.models['lightgbm'] = models
        logger.info("LightGBM training complete")

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        logger.info("Training Random Forest")

        rf_config = self.config['models']['random_forest']

        rf = RandomForestRegressor(
            n_estimators=rf_config['n_estimators'],
            max_depth=rf_config['max_depth'],
            min_samples_split=rf_config['min_samples_split'],
            random_state=42,
            n_jobs=-1
        )

        # Use MultiOutputRegressor for multi-target
        model = MultiOutputRegressor(rf)
        model.fit(X_train, y_train)

        self.models['random_forest'] = model
        logger.info("Random Forest training complete")

    def calculate_ensemble_weights(self, X_val, y_val):
        """Calculate optimal weights based on validation performance"""
        logger.info("Calculating ensemble weights")

        val_scores = {}

        for model_name, models in self.models.items():
            # Make predictions
            if model_name == 'random_forest':
                y_pred = models.predict(X_val)
            else:
                # For XGBoost/LightGBM (list of models)
                y_pred = np.column_stack([m.predict(X_val) for m in models])

            # Calculate RMSE
            rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
            val_scores[model_name] = rmse

            logger.info(f"{model_name} validation RMSE: {rmse:.2f}")

        # Calculate weights (inverse of RMSE)
        inverse_rmse = {name: 1.0 / rmse for name, rmse in val_scores.items()}
        total_inverse = sum(inverse_rmse.values())

        self.ensemble_weights = {
            name: inv / total_inverse
            for name, inv in inverse_rmse.items()
        }

        logger.info(f"Ensemble weights: {self.ensemble_weights}")

        return val_scores

    def predict_ensemble(self, X):
        """Make ensemble prediction"""
        predictions = []

        for model_name, models in self.models.items():
            if model_name == 'random_forest':
                pred = models.predict(X)
            else:
                pred = np.column_stack([m.predict(X) for m in models])

            weighted_pred = pred * self.ensemble_weights[model_name]
            predictions.append(weighted_pred)

        return np.sum(predictions, axis=0)

    def train(self, df: pd.DataFrame):
        """Train complete ensemble pipeline"""
        logger.info("Starting model training")

        # Prepare data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_data(df)

        # Train models
        self.train_xgboost(X_train, y_train, X_val, y_val)
        self.train_lightgbm(X_train, y_train, X_val, y_val)
        self.train_random_forest(X_train, y_train)

        # Calculate ensemble weights
        val_scores = self.calculate_ensemble_weights(X_val, y_val)

        logger.info("Model training complete")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test), val_scores

    def save(self, version: str, output_dir: Path):
        """Save all model artifacts"""
        logger.info(f"Saving model version {version}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        joblib.dump(self.models, output_dir / f'ensemble_models_v{version}.pkl')
        joblib.dump(self.scaler, output_dir / f'scaler_v{version}.pkl')
        joblib.dump(self.ensemble_weights, output_dir / f'weights_v{version}.pkl')
        joblib.dump(self.feature_names, output_dir / f'features_v{version}.pkl')

        logger.info(f"Models saved to {output_dir}")

    def load(self, version: str, model_dir: Path):
        """Load model artifacts"""
        logger.info(f"Loading model version {version}")

        model_dir = Path(model_dir)

        self.models = joblib.load(model_dir / f'ensemble_models_v{version}.pkl')
        self.scaler = joblib.load(model_dir / f'scaler_v{version}.pkl')
        self.ensemble_weights = joblib.load(model_dir / f'weights_v{version}.pkl')
        self.feature_names = joblib.load(model_dir / f'features_v{version}.pkl')

        logger.info(f"Models loaded from {model_dir}")


# Main training script
if __name__ == "__main__":
    # Load config
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load engineered data
    df = pd.read_csv('data/processed/studio_data_engineered.csv')

    # Initialize MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    with mlflow.start_run(run_name=f"training_v{config['model']['version']}"):
        # Log config
        mlflow.log_params(config['models']['xgboost'])

        # Train model
        model = ForwardPredictionModel(config)
        (X_train, y_train), (X_val, y_val), (X_test, y_test), val_scores = model.train(df)

        # Log validation metrics
        for model_name, rmse in val_scores.items():
            mlflow.log_metric(f"val_rmse_{model_name}", rmse)

        # Save model
        model.save(
            version=config['model']['version'],
            output_dir=Path(config['model']['output_dir'])
        )

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        logger.info("Training pipeline complete")
```

**config/model_config.yaml:**

```yaml
model:
  version: "1.0.0"
  output_dir: "data/models/"

models:
  xgboost:
    n_estimators: 300
    max_depth: 5
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8

  lightgbm:
    n_estimators: 300
    max_depth: 5
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8

  random_forest:
    n_estimators: 200
    max_depth: 10
    min_samples_split: 5

mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "studio_revenue_predictor"
```

### 4.2 Run Training

```bash
# First, run feature engineering
python -c "
from src.features.feature_engineer import FeatureEngineer
import pandas as pd

df = pd.read_csv('data/processed/studio_data_2019_2025.csv')
engineer = FeatureEngineer()
df_engineered = engineer.engineer_features(df, is_training=True)
df_engineered.to_csv('data/processed/studio_data_engineered.csv', index=False)
print('Feature engineering complete')
"

# Then train model
python training/train_forward_model.py
```

---

## 5. Model Validation

### 5.1 Create Evaluation Script

**training/evaluate_model.py:**

```python
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self, model_dir: Path, version: str):
        self.model_dir = Path(model_dir)
        self.version = version
        self.load_model()

    def load_model(self):
        """Load trained model"""
        logger.info(f"Loading model version {self.version}")

        self.models = joblib.load(
            self.model_dir / f'ensemble_models_v{self.version}.pkl'
        )
        self.scaler = joblib.load(
            self.model_dir / f'scaler_v{self.version}.pkl'
        )
        self.weights = joblib.load(
            self.model_dir / f'weights_v{self.version}.pkl'
        )
        self.feature_names = joblib.load(
            self.model_dir / f'features_v{self.version}.pkl'
        )

    def predict_ensemble(self, X):
        """Make ensemble prediction"""
        predictions = []

        for model_name, models in self.models.items():
            if model_name == 'random_forest':
                pred = models.predict(X)
            else:
                pred = np.column_stack([m.predict(X) for m in models])

            weighted_pred = pred * self.weights[model_name]
            predictions.append(weighted_pred)

        return np.sum(predictions, axis=0)

    def evaluate(self, X, y, dataset_name="Test"):
        """Evaluate model performance"""
        logger.info(f"Evaluating on {dataset_name} set")

        # Make predictions
        y_pred = self.predict_ensemble(X)

        # Calculate metrics
        target_names = [
            'Revenue Month 1', 'Revenue Month 2', 'Revenue Month 3',
            'Members Month 3', 'Retention Month 3'
        ]

        metrics = {}

        for i, name in enumerate(target_names):
            rmse = np.sqrt(mean_squared_error(y[:, i], y_pred[:, i]))
            mae = mean_absolute_error(y[:, i], y_pred[:, i])
            r2 = r2_score(y[:, i], y_pred[:, i])

            # MAPE
            mape = np.mean(np.abs((y[:, i] - y_pred[:, i]) / y[:, i])) * 100

            # Directional accuracy
            if i < 3:  # Only for revenue targets
                direction_actual = np.sign(np.diff(y[:, i]))
                direction_pred = np.sign(np.diff(y_pred[:, i]))
                dir_accuracy = np.mean(direction_actual == direction_pred) * 100
            else:
                dir_accuracy = None

            metrics[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE': mape,
                'Directional Accuracy': dir_accuracy
            }

            logger.info(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, "
                       f"R²={r2:.4f}, MAPE={mape:.2f}%")

        return metrics, y_pred

    def plot_predictions(self, y_true, y_pred, dataset_name="Test", save_dir=None):
        """Plot actual vs predicted"""
        target_names = [
            'Revenue Month 1', 'Revenue Month 2', 'Revenue Month 3',
            'Members Month 3', 'Retention Month 3'
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, name in enumerate(target_names):
            ax = axes[i]

            # Scatter plot
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)

            # Perfect prediction line
            min_val = min(y_true[:, i].min(), y_pred[:, i].min())
            max_val = max(y_true[:, i].max(), y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(name)
            ax.grid(True, alpha=0.3)

        # Remove extra subplot
        fig.delaxes(axes[5])

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / f'predictions_{dataset_name.lower()}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.show()

    def analyze_feature_importance(self, save_dir=None):
        """Analyze feature importance from tree models"""
        logger.info("Analyzing feature importance")

        # Get feature importance from XGBoost (first model - revenue month 1)
        xgb_model = self.models['xgboost'][0]
        importance = xgb_model.feature_importances_

        # Sort features by importance
        indices = np.argsort(importance)[::-1][:20]  # Top 20

        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(20), importance[indices])
        plt.yticks(range(20), [self.feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / 'feature_importance.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")

        plt.show()

        return {self.feature_names[i]: importance[i] for i in indices}

    def calculate_confidence_intervals(self, X, percentile=95):
        """Calculate prediction confidence intervals using ensemble variance"""
        logger.info(f"Calculating {percentile}% confidence intervals")

        # Get predictions from each model
        predictions = []

        for model_name, models in self.models.items():
            if model_name == 'random_forest':
                pred = models.predict(X)
            else:
                pred = np.column_stack([m.predict(X) for m in models])
            predictions.append(pred)

        # Stack predictions
        predictions = np.stack(predictions, axis=0)  # (n_models, n_samples, n_targets)

        # Calculate mean and std
        pred_mean = np.mean(predictions, axis=0)
        pred_std = np.std(predictions, axis=0)

        # Calculate confidence intervals
        z_score = {90: 1.645, 95: 1.96, 99: 2.576}[percentile]
        lower = pred_mean - z_score * pred_std
        upper = pred_mean + z_score * pred_std

        return pred_mean, lower, upper


# Run evaluation
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/processed/studio_data_engineered.csv')

    # Prepare test set
    test_df = df[df['split'] == 'test']

    target_cols = [
        'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
        'member_count_month_3', 'retention_rate_month_3'
    ]
    exclude_cols = target_cols + ['month_year', 'split', 'year_index']

    # Load feature names
    feature_names = joblib.load('data/models/features_v1.0.0.pkl')

    X_test = test_df[feature_names].values
    y_test = test_df[target_cols].values

    # Scale features
    scaler = joblib.load('data/models/scaler_v1.0.0.pkl')
    X_test_scaled = scaler.transform(X_test)

    # Evaluate
    evaluator = ModelEvaluator(model_dir='data/models', version='1.0.0')

    metrics, y_pred = evaluator.evaluate(X_test_scaled, y_test, dataset_name="Test")

    # Plot predictions
    evaluator.plot_predictions(y_test, y_pred, dataset_name="Test",
                              save_dir='reports/figures')

    # Analyze feature importance
    feature_importance = evaluator.analyze_feature_importance(save_dir='reports/figures')

    # Calculate confidence intervals
    pred_mean, lower, upper = evaluator.calculate_confidence_intervals(X_test_scaled)

    logger.info("Evaluation complete")
```

---

## 6. Model Testing & Evaluation

### 6.1 Run Comprehensive Tests

```bash
# Create reports directory
mkdir -p reports/figures

# Run evaluation
python training/evaluate_model.py
```

### 6.2 Business Metrics Evaluation

Create custom business metrics that matter for the use case:

**training/business_metrics.py:**

```python
import numpy as np
from typing import Dict

def calculate_business_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculate business-specific metrics"""

    # Revenue predictions (first 3 targets)
    revenue_true = y_true[:, :3]
    revenue_pred = y_pred[:, :3]

    # 1. Within 5% accuracy rate
    pct_error = np.abs((revenue_pred - revenue_true) / revenue_true) * 100
    within_5pct = np.mean(pct_error <= 5.0) * 100
    within_10pct = np.mean(pct_error <= 10.0) * 100

    # 2. Revenue forecast accuracy (3-month cumulative)
    cumulative_true = np.sum(revenue_true, axis=1)
    cumulative_pred = np.sum(revenue_pred, axis=1)
    forecast_accuracy = 1 - np.mean(np.abs(
        (cumulative_pred - cumulative_true) / cumulative_true
    ))

    # 3. Directional accuracy (did we predict growth/decline correctly?)
    direction_true = np.sign(np.diff(revenue_true, axis=1))
    direction_pred = np.sign(np.diff(revenue_pred, axis=1))
    directional_accuracy = np.mean(direction_true == direction_pred) * 100

    # 4. Business impact score (heavier penalty for large errors)
    penalty = np.where(pct_error > 10, 2.0, 1.0)
    business_impact_score = 100 - np.mean(pct_error * penalty)

    return {
        'within_5_percent': within_5pct,
        'within_10_percent': within_10pct,
        'forecast_accuracy': forecast_accuracy * 100,
        'directional_accuracy': directional_accuracy,
        'business_impact_score': business_impact_score
    }
```

---

## 7. Model Serialization & Versioning

### 7.1 Model Registry

**src/models/model_registry.py:**

```python
import joblib
from pathlib import Path
from typing import Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Centralized model registry for loading/saving models"""

    def __init__(self, base_dir: str = "data/models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, version: str, metadata: dict = None):
        """Save model with versioning"""
        logger.info(f"Saving model version {version}")

        version_dir = self.base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model artifacts
        joblib.dump(model.models, version_dir / 'ensemble_models.pkl')
        joblib.dump(model.scaler, version_dir / 'scaler.pkl')
        joblib.dump(model.ensemble_weights, version_dir / 'weights.pkl')
        joblib.dump(model.feature_names, version_dir / 'features.pkl')

        # Save metadata
        if metadata is None:
            metadata = {}

        metadata.update({
            'version': version,
            'saved_at': datetime.now().isoformat(),
            'n_features': len(model.feature_names)
        })

        joblib.dump(metadata, version_dir / 'metadata.pkl')

        logger.info(f"Model saved to {version_dir}")

    def load_model(self, version: str = "latest"):
        """Load model by version"""
        if version == "latest":
            # Find latest version
            versions = [d.name for d in self.base_dir.iterdir() if d.is_dir()]
            versions.sort(reverse=True)
            version = versions[0] if versions else None

            if version is None:
                raise ValueError("No models found in registry")

        logger.info(f"Loading model version {version}")

        version_dir = self.base_dir / version

        if not version_dir.exists():
            raise ValueError(f"Model version {version} not found")

        # Load artifacts
        models = joblib.load(version_dir / 'ensemble_models.pkl')
        scaler = joblib.load(version_dir / 'scaler.pkl')
        weights = joblib.load(version_dir / 'weights.pkl')
        features = joblib.load(version_dir / 'features.pkl')
        metadata = joblib.load(version_dir / 'metadata.pkl')

        logger.info(f"Loaded model version {version} with {len(features)} features")

        return {
            'models': models,
            'scaler': scaler,
            'weights': weights,
            'features': features,
            'metadata': metadata
        }

    def list_versions(self):
        """List all available model versions"""
        versions = []
        for version_dir in self.base_dir.iterdir():
            if version_dir.is_dir():
                metadata_file = version_dir / 'metadata.pkl'
                if metadata_file.exists():
                    metadata = joblib.load(metadata_file)
                    versions.append(metadata)

        return sorted(versions, key=lambda x: x['saved_at'], reverse=True)
```

---

## 8. FastAPI Backend Development

### 8.1 Project Structure Setup

**src/api/main.py:**

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from src.api.routes import predictions, scenarios, insights
from src.api.services.prediction_service import PredictionService
from src.models.model_registry import ModelRegistry
from src.database.connection import DatabaseManager
import redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    # Startup
    logger.info("Initializing application...")

    # Load model
    registry = ModelRegistry()
    model_artifacts = registry.load_model(version="latest")
    app_state['model'] = model_artifacts
    logger.info(f"Loaded model version {model_artifacts['metadata']['version']}")

    # Initialize database
    db_manager = DatabaseManager()
    await db_manager.connect()
    app_state['db'] = db_manager
    logger.info("Database connection established")

    # Initialize Redis
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    app_state['redis'] = redis_client
    logger.info("Redis connection established")

    # Initialize prediction service
    prediction_service = PredictionService(
        model_artifacts=model_artifacts,
        db_manager=db_manager,
        redis_client=redis_client
    )
    app_state['prediction_service'] = prediction_service

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    await db_manager.disconnect()
    redis_client.close()
    logger.info("Application shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Studio Revenue Simulator API",
    description="Predict fitness studio revenue and optimize business levers",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions.router, prefix="/api/v1/predict", tags=["predictions"])
app.include_router(scenarios.router, prefix="/api/v1/scenarios", tags=["scenarios"])
app.include_router(insights.router, prefix="/api/v1/insights", tags=["insights"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Studio Revenue Simulator",
        "version": "1.0.0"
    }

@app.get("/api/v1/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_version": app_state['model']['metadata']['version'],
        "model_features": len(app_state['model']['features']),
        "database": "connected",
        "cache": "connected"
    }
```

### 8.2 Request/Response Schemas

**src/api/schemas/requests.py:**

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date

class LeverInputs(BaseModel):
    """Input levers for forward prediction"""
    retention_rate: float = Field(..., ge=0.5, le=1.0, description="Member retention rate")
    avg_ticket_price: float = Field(..., ge=50.0, le=500.0, description="Average monthly ticket price")
    class_attendance_rate: float = Field(..., ge=0.4, le=1.0, description="Class attendance rate")
    new_members_monthly: int = Field(..., ge=0, le=100, description="New members per month")
    staff_utilization_rate: float = Field(..., ge=0.6, le=1.0, description="Staff utilization rate")
    upsell_rate: float = Field(..., ge=0.0, le=0.5, description="Upsell rate")
    total_classes_held: int = Field(..., ge=50, le=500, description="Total classes per month")
    current_member_count: int = Field(..., ge=50, description="Current member count")

class ForwardPredictionRequest(BaseModel):
    """Request for forward prediction"""
    studio_id: str
    base_month: date
    projection_months: int = Field(default=3, ge=1, le=12)
    levers: LeverInputs
    include_confidence_intervals: bool = False

class OptimizationConstraints(BaseModel):
    """Constraints for inverse optimization"""
    max_retention_increase: Optional[float] = Field(0.05, description="Max retention rate increase")
    max_ticket_increase: Optional[float] = Field(20.0, description="Max ticket price increase")
    max_new_members_increase: Optional[int] = Field(10, description="Max new member increase")
    prioritize_low_cost_levers: bool = True

class CurrentState(LeverInputs):
    """Current state of the studio"""
    pass

class InversePredictionRequest(BaseModel):
    """Request for inverse prediction (target → levers)"""
    studio_id: str
    base_month: date
    target_revenue: float = Field(..., gt=0)
    target_months: int = Field(default=3, ge=1, le=12)
    current_state: CurrentState
    constraints: Optional[OptimizationConstraints] = None
```

**src/api/schemas/responses.py:**

```python
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

class MonthlyPrediction(BaseModel):
    """Prediction for a single month"""
    revenue: float
    member_count: int
    confidence_interval: Optional[List[float]] = None
    confidence_score: float

class ForwardPredictionResponse(BaseModel):
    """Response for forward prediction"""
    scenario_id: str
    predictions: Dict[str, MonthlyPrediction]
    growth_rate: float
    total_projected_revenue: float
    model_version: str
    prediction_accuracy: float

class LeverChange(BaseModel):
    """Details of a lever change recommendation"""
    current: float
    recommended: float
    change_pct: float
    impact_on_revenue: float
    feasibility_score: float
    estimated_cost: str

class ActionItem(BaseModel):
    """Recommended action"""
    priority: int
    lever: str
    action: str
    expected_impact: float
    timeline_weeks: int

class InversePredictionResponse(BaseModel):
    """Response for inverse prediction"""
    optimization_id: str
    target_revenue: float
    achievable_revenue: float
    achievement_rate: float
    recommended_levers: Dict[str, float]
    lever_changes: Dict[str, LeverChange]
    action_plan: List[ActionItem]
    confidence_score: float
```

### 8.3 Prediction Service

**src/api/services/prediction_service.py:**

```python
import numpy as np
from typing import Dict, Tuple
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PredictionService:
    """Core service for making predictions"""

    def __init__(self, model_artifacts, db_manager, redis_client):
        self.models = model_artifacts['models']
        self.scaler = model_artifacts['scaler']
        self.weights = model_artifacts['weights']
        self.features = model_artifacts['features']
        self.metadata = model_artifacts['metadata']
        self.db = db_manager
        self.redis = redis_client

    async def predict_forward(self, request_data: Dict) -> Dict:
        """Forward prediction: levers → revenue"""
        logger.info(f"Forward prediction for studio {request_data['studio_id']}")

        # Check cache
        cache_key = self._generate_cache_key('forward', request_data)
        cached = self.redis.get(cache_key)
        if cached:
            logger.info("Returning cached prediction")
            return eval(cached)

        # Fetch historical data
        historical_data = await self.db.get_recent_history(
            request_data['studio_id'],
            months=12
        )

        # Engineer features
        features = self._engineer_features_for_prediction(
            request_data['levers'],
            historical_data
        )

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Make predictions
        predictions = self._predict_ensemble(features_scaled)

        # Calculate confidence
        confidence_score = self._calculate_confidence(predictions, historical_data)

        # Format response
        response = {
            'scenario_id': str(uuid.uuid4()),
            'predictions': self._format_predictions(predictions, request_data),
            'growth_rate': self._calculate_growth_rate(predictions),
            'total_projected_revenue': float(np.sum(predictions[:3])),
            'model_version': self.metadata['version'],
            'prediction_accuracy': confidence_score
        }

        # Cache result
        self.redis.setex(cache_key, 3600, str(response))

        # Log prediction
        await self.db.log_prediction(
            studio_id=request_data['studio_id'],
            prediction_type='forward',
            inputs=request_data['levers'],
            outputs=response
        )

        return response

    def _predict_ensemble(self, X):
        """Make ensemble prediction"""
        predictions = []

        for model_name, models in self.models.items():
            if model_name == 'random_forest':
                pred = models.predict(X)
            else:
                pred = np.column_stack([m.predict(X) for m in models])

            weighted_pred = pred * self.weights[model_name]
            predictions.append(weighted_pred)

        return np.sum(predictions, axis=0)[0]  # Return first (and only) row

    def _engineer_features_for_prediction(self, levers: Dict,
                                         historical_data) -> np.ndarray:
        """Engineer features from lever inputs"""
        feature_dict = levers.copy()

        # Add historical context
        feature_dict['prev_month_revenue'] = historical_data['total_revenue'].iloc[-1]
        feature_dict['3m_avg_retention'] = historical_data['retention_rate'].tail(3).mean()
        feature_dict['3m_avg_revenue'] = historical_data['total_revenue'].tail(3).mean()

        # Add derived features
        feature_dict['revenue_per_member'] = (
            levers['avg_ticket_price'] * levers['retention_rate']
        )
        feature_dict['churn_rate'] = 1 - levers['retention_rate']

        # Add interaction features
        feature_dict['retention_x_ticket'] = (
            levers['retention_rate'] * levers['avg_ticket_price']
        )

        # Convert to array (ensure correct feature order)
        feature_vector = np.array([
            feature_dict.get(name, 0) for name in self.features
        ])

        return feature_vector.reshape(1, -1)

    def _calculate_confidence(self, predictions, historical_data) -> float:
        """Calculate prediction confidence based on historical variance"""
        historical_std = historical_data['total_revenue'].std()
        prediction_value = predictions[0]  # First month revenue

        # Confidence inversely proportional to coefficient of variation
        cv = historical_std / historical_data['total_revenue'].mean()
        confidence = max(0.5, 1.0 - cv)

        return float(confidence)

    def _format_predictions(self, predictions, request_data) -> Dict:
        """Format predictions into response structure"""
        projection_months = request_data['projection_months']
        formatted = {}

        for i in range(projection_months):
            month_key = f"month_{i+1}"
            formatted[month_key] = {
                'revenue': float(predictions[i]) if i < 3 else float(predictions[0]),
                'member_count': int(predictions[3]) if i == 2 else request_data['levers']['current_member_count'],
                'confidence_score': 0.85 - (i * 0.03)  # Decreasing confidence over time
            }

        return formatted

    def _calculate_growth_rate(self, predictions) -> float:
        """Calculate projected growth rate"""
        first_month = predictions[0]
        last_month = predictions[2]
        growth_rate = (last_month - first_month) / first_month
        return float(growth_rate)

    def _generate_cache_key(self, pred_type: str, request_data: Dict) -> str:
        """Generate cache key from request"""
        import hashlib
        import json

        # Create deterministic string from request
        key_data = {
            'type': pred_type,
            'studio_id': request_data['studio_id'],
            'levers': request_data.get('levers', {}),
            'target': request_data.get('target_revenue')
        }
        key_str = json.dumps(key_data, sort_keys=True)
        hash_str = hashlib.md5(key_str.encode()).hexdigest()

        return f"prediction:{pred_type}:{hash_str}"
```

### 8.4 Prediction Routes

**src/api/routes/predictions.py:**

```python
from fastapi import APIRouter, HTTPException, Depends
from src.api.schemas.requests import ForwardPredictionRequest, InversePredictionRequest
from src.api.schemas.responses import ForwardPredictionResponse, InversePredictionResponse
from src.api.main import app_state
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

def get_prediction_service():
    """Dependency to get prediction service"""
    return app_state['prediction_service']

@router.post("/forward", response_model=ForwardPredictionResponse)
async def forward_prediction(
    request: ForwardPredictionRequest,
    service = Depends(get_prediction_service)
):
    """
    Forward prediction: Given lever values, predict future revenue
    """
    try:
        result = await service.predict_forward(request.dict())
        return result
    except Exception as e:
        logger.error(f"Forward prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/inverse", response_model=InversePredictionResponse)
async def inverse_prediction(
    request: InversePredictionRequest,
    service = Depends(get_prediction_service)
):
    """
    Inverse prediction: Given target revenue, find optimal lever values
    """
    try:
        result = await service.predict_inverse(request.dict())
        return result
    except Exception as e:
        logger.error(f"Inverse prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 9. API Testing & Validation

### 9.1 Unit Tests

**tests/test_api.py:**

```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'

def test_forward_prediction():
    """Test forward prediction endpoint"""
    request_data = {
        "studio_id": "test-studio-1",
        "base_month": "2024-01-01",
        "projection_months": 3,
        "levers": {
            "retention_rate": 0.75,
            "avg_ticket_price": 150.0,
            "class_attendance_rate": 0.70,
            "new_members_monthly": 25,
            "staff_utilization_rate": 0.85,
            "upsell_rate": 0.25,
            "total_classes_held": 120,
            "current_member_count": 250
        },
        "include_confidence_intervals": False
    }

    response = client.post("/api/v1/predict/forward", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert 'scenario_id' in data
    assert 'predictions' in data
    assert 'month_1' in data['predictions']
    assert data['predictions']['month_1']['revenue'] > 0
```

### 9.2 Run Tests

```bash
# Install pytest
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v
```

---

## 10. Deployment

### 10.1 Run FastAPI Server

**Development:**

```bash
# Run with auto-reload
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Production:**

```bash
# Run with multiple workers
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 10.2 Test API with cURL

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Forward prediction
curl -X POST http://localhost:8000/api/v1/predict/forward \
  -H "Content-Type: application/json" \
  -d '{
    "studio_id": "studio-123",
    "base_month": "2024-01-01",
    "projection_months": 3,
    "levers": {
      "retention_rate": 0.75,
      "avg_ticket_price": 150.0,
      "class_attendance_rate": 0.70,
      "new_members_monthly": 25,
      "staff_utilization_rate": 0.85,
      "upsell_rate": 0.25,
      "total_classes_held": 120,
      "current_member_count": 250
    }
  }'
```

---

## 11. Monitoring & Maintenance

### 11.1 Logging Predictions

Track all predictions for model performance monitoring:

```python
# Add to prediction service
async def log_prediction(self, studio_id, prediction_type, inputs, outputs):
    """Log prediction for monitoring"""
    await self.db.execute("""
        INSERT INTO prediction_logs
        (studio_id, prediction_type, input_levers, predicted_output, predicted_at)
        VALUES ($1, $2, $3, $4, NOW())
    """, studio_id, prediction_type, inputs, outputs)
```

### 11.2 Model Retraining Pipeline

Set up periodic retraining:

```python
# training/retrain_pipeline.py
def retrain_model():
    """Retrain model with latest data"""
    # Fetch new data
    # Retrain model
    # Evaluate performance
    # If performance improves, deploy new version
    pass
```

---

## Summary

This guide provides a complete end-to-end workflow for:

1. ✅ **Data Generation** - Realistic synthetic data with seasonality
2. ✅ **Feature Engineering** - 20+ engineered features
3. ✅ **Model Training** - Ensemble of XGBoost, LightGBM, Random Forest
4. ✅ **Model Validation** - RMSE, MAPE, R², business metrics
5. ✅ **Model Testing** - Comprehensive evaluation on test set
6. ✅ **Model Serialization** - Version-controlled model registry
7. ✅ **FastAPI Deployment** - Production-ready REST API
8. ✅ **API Testing** - Unit tests and integration tests
9. ✅ **Monitoring** - Prediction logging and performance tracking

**Next Steps:**

- Implement inverse prediction (optimization)
- Add AI insights generation with OpenAI
- Set up continuous monitoring
- Implement A/B testing for model versions
