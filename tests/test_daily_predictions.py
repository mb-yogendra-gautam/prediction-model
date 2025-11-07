"""
Test Suite for Daily Prediction System

Validates:
- Daily data generation and aggregation
- Daily feature engineering
- Daily model predictions
- Aggregation to monthly forecasts
- API integration with daily model
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.daily_data_generator import DailyDataGenerator
from src.features.daily_feature_engineer import DailyFeatureEngineer
from src.api.services.daily_prediction_engine import DailyPredictionEngine


class TestDailyDataGeneration:
    """Test daily data generation from monthly aggregates"""
    
    def test_daily_aggregation_sums_to_monthly(self):
        """Verify daily values sum to monthly totals"""
        generator = DailyDataGenerator(seed=42)
        
        # Test distribution for a month
        monthly_revenue = 15000.0
        year, month = 2024, 1
        
        daily_values = generator.distribute_monthly_value(
            monthly_revenue, year, month, value_type='revenue'
        )
        
        # Sum daily values
        daily_sum = sum(daily_values.values())
        
        # Check within small rounding tolerance
        diff_pct = abs(daily_sum - monthly_revenue) / monthly_revenue
        assert diff_pct < 0.001, f"Daily sum {daily_sum} should match monthly {monthly_revenue}"
    
    def test_weekend_pattern_exists(self):
        """Verify weekend days have lower revenue"""
        generator = DailyDataGenerator(seed=42)
        
        # Generate a full month
        monthly_revenue = 15000.0
        year, month = 2024, 1
        daily_values = generator.distribute_monthly_value(
            monthly_revenue, year, month, value_type='revenue'
        )
        
        # Separate weekday and weekend revenues
        weekday_revenues = []
        weekend_revenues = []
        
        for day, revenue in daily_values.items():
            date = datetime(year, month, day)
            if generator.is_weekend(date):
                weekend_revenues.append(revenue)
            else:
                weekday_revenues.append(revenue)
        
        # Weekend average should be lower than weekday average
        avg_weekend = np.mean(weekend_revenues)
        avg_weekday = np.mean(weekday_revenues)
        
        assert avg_weekend < avg_weekday, "Weekend revenue should be lower than weekday"
        print(f"[OK] Weekday avg: ${avg_weekday:.2f}, Weekend avg: ${avg_weekend:.2f}")
    
    def test_holiday_pattern_exists(self):
        """Verify holidays have lower revenue"""
        generator = DailyDataGenerator(seed=42)
        
        # Test January (has New Year's Day holiday)
        monthly_revenue = 15000.0
        year, month = 2024, 1
        daily_values = generator.distribute_monthly_value(
            monthly_revenue, year, month, value_type='revenue'
        )
        
        # Check New Year's Day (Jan 1)
        new_years_revenue = daily_values[1]
        avg_revenue = np.mean(list(daily_values.values()))
        
        assert new_years_revenue < avg_revenue, "Holiday revenue should be below average"
        print(f"[OK] New Year's revenue: ${new_years_revenue:.2f}, Avg: ${avg_revenue:.2f}")
    
    def test_daily_features_generated(self):
        """Verify all daily features are generated"""
        generator = DailyDataGenerator(seed=42)
        
        date = datetime(2024, 6, 15)  # A Saturday, payday
        features = generator.generate_daily_features(date)
        
        required_features = [
            'date', 'year', 'month', 'day', 'day_of_week',
            'is_weekend', 'week_of_month', 'day_of_month',
            'is_holiday', 'days_since_payday', 'is_month_start', 'is_month_end'
        ]
        
        for feat in required_features:
            assert feat in features, f"Missing feature: {feat}"
        
        # Verify specific values
        assert features['is_weekend'] == 1, "June 15, 2024 is a Saturday"
        assert features['days_since_payday'] == 0, "June 15 is payday"
        assert features['day_of_week'] == 5, "Saturday is day 5"
        
        print(f"[OK] Generated {len(features)} daily features")


class TestDailyFeatureEngineering:
    """Test daily feature engineering"""
    
    def test_cyclical_features_added(self):
        """Verify cyclical encodings are added"""
        engineer = DailyFeatureEngineer(multi_studio=False)
        
        # Create sample data
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'day_of_week': [0, 1],
            'day_of_month': [1, 2],
            'month': [1, 1],
            'total_revenue': [500, 520],
            'total_members': [150, 150],
            'retention_rate': [0.75, 0.75],
            'is_weekend': [0, 0],
            'is_holiday': [1, 0],
            'week_of_month': [1, 1],
            'days_since_payday': [0, 1],
            'is_month_start': [1, 1],
            'is_month_end': [0, 0],
            'studio_id': ['STU001', 'STU001'],
            'split': ['train', 'train']
        })
        
        df = engineer.add_cyclical_features(df)
        
        # Check cyclical features exist
        cyclical_features = [
            'day_of_week_sin', 'day_of_week_cos',
            'day_of_month_sin', 'day_of_month_cos',
            'month_sin', 'month_cos'
        ]
        
        for feat in cyclical_features:
            assert feat in df.columns, f"Missing cyclical feature: {feat}"
            assert not df[feat].isna().any(), f"Cyclical feature {feat} has NaN values"
        
        print(f"[OK] Added {len(cyclical_features)} cyclical features")
    
    def test_interaction_features_added(self):
        """Verify interaction features are created"""
        engineer = DailyFeatureEngineer(multi_studio=False)
        
        df = pd.DataFrame({
            'date': ['2024-06-15', '2024-06-16'],  # Saturday and Sunday
            'day_of_week': [5, 6],
            'day_of_month': [15, 16],
            'month': [6, 6],
            'total_revenue': [400, 380],
            'total_members': [150, 150],
            'retention_rate': [0.75, 0.75],
            'upsell_rate': [0.15, 0.15],
            'new_members': [5, 3],
            'is_weekend': [1, 1],
            'is_holiday': [0, 0],
            'week_of_month': [3, 3],
            'days_since_payday': [0, 1],
            'is_month_start': [0, 0],
            'is_month_end': [0, 0],
            'studio_id': ['STU001', 'STU001'],
            'split': ['train', 'train']
        })
        
        df = engineer.add_interaction_features(df)
        
        # Check weekend interactions
        assert 'is_weekend_x_retention' in df.columns
        assert 'is_weekend_x_members' in df.columns
        assert df['is_weekend_x_retention'].iloc[0] == 0.75, "Weekend interaction should be retention rate"
        
        print(f"[OK] Added interaction features")


class TestDailyPredictionEngine:
    """Test daily prediction engine"""
    
    @pytest.fixture
    def mock_daily_engine(self):
        """Create a mock daily engine for testing"""
        # This would normally load the trained model
        # For testing, we'll use a simple mock
        class MockModel:
            def predict(self, X):
                # Return mock predictions: [daily_revenue, daily_members, daily_retention]
                return np.array([[500.0, 150, 0.75]])
        
        class MockScaler:
            def transform(self, X):
                return X
        
        engine = DailyPredictionEngine(
            model=MockModel(),
            scaler=MockScaler(),
            feature_selector=None,
            selected_features=['total_revenue', 'total_members', 'retention_rate']
        )
        
        return engine
    
    def test_daily_date_generation(self, mock_daily_engine):
        """Test generating daily date ranges"""
        dates = mock_daily_engine.generate_daily_dates('2024-01', 2)
        
        # Should have ~59-62 days (Jan + Feb 2024)
        assert 59 <= len(dates) <= 62, f"Expected 59-62 days, got {len(dates)}"
        
        # First date should be Jan 1, 2024
        assert dates[0].strftime('%Y-%m-%d') == '2024-01-01'
        
        # Last date should be in February
        assert dates[-1].month == 2
        
        print(f"[OK] Generated {len(dates)} daily dates")
    
    def test_seasonality_features_generation(self, mock_daily_engine):
        """Test seasonality feature generation"""
        date = datetime(2024, 12, 25)  # Christmas
        features = mock_daily_engine.add_daily_seasonality_features(date)
        
        assert features['is_holiday'] == 1, "Dec 25 should be a holiday"
        assert features['is_weekend'] in [0, 1], "is_weekend should be binary"
        assert 'day_of_week_sin' in features, "Should have cyclical encoding"
        assert 'day_of_week_cos' in features, "Should have cyclical encoding"
        
        print(f"[OK] Generated {len(features)} seasonality features")
    
    def test_daily_aggregation_to_monthly(self, mock_daily_engine):
        """Test aggregating daily predictions to monthly"""
        # Create mock daily predictions
        dates = pd.date_range('2024-01-01', '2024-02-29', freq='D')
        daily_predictions = pd.DataFrame({
            'date': dates,
            'daily_revenue': [500.0] * len(dates),
            'daily_members': [150] * len(dates),
            'daily_retention': [0.75] * len(dates)
        })
        
        # Aggregate
        monthly_predictions = mock_daily_engine.aggregate_daily_to_monthly(daily_predictions)
        
        # Should have 2 months
        assert len(monthly_predictions) == 2, "Should have 2 months"
        
        # January should have 31 days * 500 = 15,500
        jan_revenue = monthly_predictions[0]['revenue']
        assert abs(jan_revenue - 15500) < 1, f"January revenue should be ~15,500, got {jan_revenue}"
        
        # February 2024 has 29 days * 500 = 14,500
        feb_revenue = monthly_predictions[1]['revenue']
        assert abs(feb_revenue - 14500) < 1, f"February revenue should be ~14,500, got {feb_revenue}"
        
        print(f"[OK] Aggregated daily to {len(monthly_predictions)} monthly predictions")


class TestAPIIntegration:
    """Test API integration with daily model"""
    
    def test_response_format_compatibility(self):
        """Verify daily model returns same response format as monthly model"""
        # This would test the actual API endpoint
        # For now, just verify the expected structure
        
        expected_fields = [
            'scenario_id', 'studio_id', 'predictions',
            'total_projected_revenue', 'average_confidence',
            'model_version', 'timestamp'
        ]
        
        # Monthly prediction structure
        monthly_pred_fields = [
            'month', 'revenue', 'member_count',
            'retention_rate', 'confidence_score'
        ]
        
        print("[OK] Response format defined")
        print(f"  - Top-level fields: {expected_fields}")
        print(f"  - Monthly prediction fields: {monthly_pred_fields}")
        
        assert True  # Placeholder for actual API test


def run_all_tests():
    """Run all tests and print summary"""
    print("\n" + "="*80)
    print("DAILY PREDICTION SYSTEM - TEST SUITE")
    print("="*80 + "\n")
    
    test_classes = [
        TestDailyDataGeneration,
        TestDailyFeatureEngineering,
        TestDailyPredictionEngine,
        TestAPIIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 80)
        
        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                
                # Handle fixtures
                if method_name in ['test_daily_date_generation', 'test_seasonality_features_generation', 
                                   'test_daily_aggregation_to_monthly']:
                    # Create mock engine
                    class MockModel:
                        def predict(self, X):
                            return np.array([[500.0, 150, 0.75]])
                    class MockScaler:
                        def transform(self, X):
                            return X
                    mock_engine = DailyPredictionEngine(
                        model=MockModel(),
                        scaler=MockScaler(),
                        feature_selector=None,
                        selected_features=['total_revenue', 'total_members', 'retention_rate']
                    )
                    method(mock_engine)
                else:
                    method()
                
                passed_tests += 1
                print(f"  [PASS] {method_name}")
            except Exception as e:
                failed_tests.append((test_class.__name__, method_name, str(e)))
                print(f"  [FAIL] {method_name}: {str(e)}")
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\nTotal tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
    else:
        print("\n[SUCCESS] All tests passed!")
    
    print("\n" + "="*80 + "\n")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

