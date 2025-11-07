"""
Daily Prediction Engine

Handles daily-level predictions and aggregates them to monthly forecasts.
This service:
- Generates daily predictions for specific date ranges
- Applies seasonality features (weekends, holidays, pay periods)
- Aggregates daily predictions to monthly totals
- Maintains compatibility with existing API response format
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import calendar
import logging

logger = logging.getLogger(__name__)


class DailyPredictionEngine:
    """Engine for generating and aggregating daily predictions"""
    
    # US Federal Holidays (static dates)
    HOLIDAYS = {
        (1, 1): "New Year's Day",
        (7, 4): "Independence Day",
        (12, 25): "Christmas Day",
        (12, 31): "New Year's Eve"
    }
    
    # Floating holidays by month (approximate days)
    FLOATING_HOLIDAYS_BY_MONTH = {
        1: [15],  # MLK Day (3rd Monday)
        2: [14, 15],  # Valentine's Day, Presidents Day
        5: [27],  # Memorial Day (last Monday)
        9: [3],   # Labor Day (1st Monday)
        11: [11, 22, 23],  # Veterans Day, Thanksgiving
    }
    
    def __init__(self, model, scaler, feature_selector, selected_features):
        """
        Initialize daily prediction engine
        
        Args:
            model: Trained daily prediction model
            scaler: Feature scaler
            feature_selector: Feature selector
            selected_features: List of selected feature names
        """
        self.model = model
        self.scaler = scaler
        self.feature_selector = feature_selector
        self.selected_features = selected_features
        
    def is_holiday(self, date: datetime) -> bool:
        """Check if date is a holiday"""
        if (date.month, date.day) in self.HOLIDAYS:
            return True
        if date.month in self.FLOATING_HOLIDAYS_BY_MONTH:
            if date.day in self.FLOATING_HOLIDAYS_BY_MONTH[date.month]:
                return True
        return False
    
    def get_day_of_week(self, date: datetime) -> int:
        """Get day of week (0=Monday, 6=Sunday)"""
        return date.weekday()
    
    def is_weekend(self, date: datetime) -> bool:
        """Check if date is weekend"""
        return date.weekday() >= 5
    
    def get_week_of_month(self, date: datetime) -> int:
        """Get week of month (1-5)"""
        first_day = date.replace(day=1)
        dom = date.day
        adjusted_dom = dom + first_day.weekday()
        return (adjusted_dom - 1) // 7 + 1
    
    def days_since_payday(self, date: datetime) -> int:
        """Calculate days since last payday (1st or 15th)"""
        day = date.day
        if day >= 15:
            return day - 15
        else:
            return day - 1
    
    def is_month_start(self, date: datetime) -> bool:
        """Check if in first 5 days of month"""
        return date.day <= 5
    
    def is_month_end(self, date: datetime) -> bool:
        """Check if in last 5 days of month"""
        last_day = calendar.monthrange(date.year, date.month)[1]
        return date.day > last_day - 5
    
    def generate_daily_dates(self, start_month: str, num_months: int) -> List[datetime]:
        """
        Generate list of daily dates for prediction
        
        Args:
            start_month: Starting month in 'YYYY-MM' format
            num_months: Number of months to predict
            
        Returns:
            List of datetime objects for each day
        """
        year, month = map(int, start_month.split('-'))
        start_date = datetime(year, month, 1)
        
        dates = []
        current_date = start_date
        
        for _ in range(num_months):
            num_days = calendar.monthrange(current_date.year, current_date.month)[1]
            for day in range(1, num_days + 1):
                date = datetime(current_date.year, current_date.month, day)
                dates.append(date)
            
            # Move to next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        
        return dates
    
    def add_daily_seasonality_features(self, date: datetime) -> Dict:
        """
        Generate seasonality features for a specific date
        
        Args:
            date: Date to generate features for
            
        Returns:
            Dictionary of seasonality features
        """
        dow = self.get_day_of_week(date)
        dom = date.day
        month = date.month
        
        features = {
            'day_of_week': dow,
            'is_weekend': int(self.is_weekend(date)),
            'week_of_month': self.get_week_of_month(date),
            'day_of_month': dom,
            'is_holiday': int(self.is_holiday(date)),
            'days_since_payday': self.days_since_payday(date),
            'is_month_start': int(self.is_month_start(date)),
            'is_month_end': int(self.is_month_end(date)),
            
            # Cyclical encodings
            'day_of_week_sin': np.sin(2 * np.pi * dow / 7),
            'day_of_week_cos': np.cos(2 * np.pi * dow / 7),
            'day_of_month_sin': np.sin(2 * np.pi * (dom - 1) / 31),
            'day_of_month_cos': np.cos(2 * np.pi * (dom - 1) / 31),
            'month_sin': np.sin(2 * np.pi * (month - 1) / 12),
            'month_cos': np.cos(2 * np.pi * (month - 1) / 12),
        }
        
        return features
    
    def build_daily_feature_vector(
        self, 
        date: datetime,
        studio_data: Dict,
        historical_daily_data: pd.DataFrame = None
    ) -> Dict:
        """
        Build complete feature vector for daily prediction
        
        Args:
            date: Date to predict for
            studio_data: Studio's current state (members, rates, etc.)
            historical_daily_data: Recent daily data for rolling features (optional)
            
        Returns:
            Dictionary of all features
        """
        # Start with seasonality features
        features = self.add_daily_seasonality_features(date)
        
        # Add studio's current state features
        features.update({
            'total_members': studio_data.get('total_members', 150),
            'retention_rate': studio_data.get('retention_rate', 0.75),
            'avg_ticket_price': studio_data.get('avg_ticket_price', 120),
            'class_attendance_rate': studio_data.get('class_attendance_rate', 0.65),
            'staff_count': studio_data.get('staff_count', 8),
            'avg_classes_per_member': studio_data.get('avg_classes_per_member', 8),
            'upsell_rate': studio_data.get('upsell_rate', 0.15),
        })
        
        # Add revenue components (use monthly averages divided by ~30)
        daily_factor = 1.0 / 30.0
        features.update({
            'membership_revenue': studio_data.get('membership_revenue', 12000) * daily_factor,
            'class_pack_revenue': studio_data.get('class_pack_revenue', 2500) * daily_factor,
            'retail_revenue': studio_data.get('retail_revenue', 600) * daily_factor,
        })
        
        # Add rolling features (if historical data available)
        if historical_daily_data is not None and len(historical_daily_data) > 0:
            recent_revenue = historical_daily_data['total_revenue'].values
            
            features['prev_7d_avg_revenue'] = recent_revenue[-7:].mean() if len(recent_revenue) >= 7 else recent_revenue.mean()
            features['prev_30d_avg_revenue'] = recent_revenue[-30:].mean() if len(recent_revenue) >= 30 else recent_revenue.mean()
            features['prev_7d_std_revenue'] = recent_revenue[-7:].std() if len(recent_revenue) >= 7 else recent_revenue.std()
            features['prev_1d_revenue'] = recent_revenue[-1] if len(recent_revenue) > 0 else 0
            features['prev_7d_revenue'] = recent_revenue[-7] if len(recent_revenue) >= 7 else 0
            
            # Growth rates
            if len(recent_revenue) > 1:
                features['dod_revenue_growth'] = (recent_revenue[-1] - recent_revenue[-2]) / (recent_revenue[-2] + 1)
            else:
                features['dod_revenue_growth'] = 0
                
            if len(recent_revenue) >= 7:
                features['wow_revenue_growth'] = (recent_revenue[-1] - recent_revenue[-7]) / (recent_revenue[-7] + 1)
            else:
                features['wow_revenue_growth'] = 0
        else:
            # Use defaults if no historical data
            avg_daily_revenue = studio_data.get('total_revenue', 15000) * daily_factor
            features['prev_7d_avg_revenue'] = avg_daily_revenue
            features['prev_30d_avg_revenue'] = avg_daily_revenue
            features['prev_7d_std_revenue'] = avg_daily_revenue * 0.15  # 15% volatility
            features['prev_1d_revenue'] = avg_daily_revenue
            features['prev_7d_revenue'] = avg_daily_revenue
            features['dod_revenue_growth'] = 0
            features['wow_revenue_growth'] = 0
        
        features['prev_1d_members'] = features['total_members']
        
        # Add interaction features
        features['is_weekend_x_retention'] = features['is_weekend'] * features['retention_rate']
        features['is_weekend_x_members'] = features['is_weekend'] * features['total_members']
        features['is_holiday_x_members'] = features['is_holiday'] * features['total_members']
        features['payday_x_upsell'] = (1 if features['days_since_payday'] <= 2 else 0) * features['upsell_rate']
        features['dow_x_members'] = features['day_of_week'] * features['total_members'] / 1000
        
        # Add derived features
        features['revenue_per_member'] = (features['prev_7d_avg_revenue'] / (features['total_members'] + 1))
        features['revenue_volatility'] = features['prev_7d_std_revenue'] / (features['prev_7d_avg_revenue'] + 1)
        features['revenue_momentum'] = (features['prev_7d_avg_revenue'] - features['prev_30d_avg_revenue']) / (features['prev_30d_avg_revenue'] + 1)
        
        # Fill in any missing features with 0
        for feat in self.selected_features:
            if feat not in features:
                features[feat] = 0
        
        return features
    
    def predict_daily(
        self, 
        date: datetime,
        studio_data: Dict,
        historical_daily_data: pd.DataFrame = None
    ) -> Dict:
        """
        Predict single day's metrics
        
        Args:
            date: Date to predict
            studio_data: Studio's current state
            historical_daily_data: Recent daily data for context
            
        Returns:
            Dictionary with predicted daily metrics
        """
        # Build feature vector
        features = self.build_daily_feature_vector(date, studio_data, historical_daily_data)
        
        # Create feature array in correct order
        feature_array = np.array([[features.get(feat, 0) for feat in self.selected_features]])
        
        # Scale features
        feature_array_scaled = self.scaler.transform(feature_array)
        
        # Predict
        prediction = self.model.predict(feature_array_scaled)[0]
        
        # Extract predictions (model predicts: [daily_revenue, daily_members, daily_retention])
        return {
            'date': date.strftime('%Y-%m-%d'),
            'daily_revenue': round(float(prediction[0]), 2),
            'daily_members': int(round(prediction[1])),
            'daily_retention': round(float(prediction[2]), 4),
            'is_weekend': features['is_weekend'],
            'is_holiday': features['is_holiday']
        }
    
    def predict_daily_sequence(
        self,
        dates: List[datetime],
        studio_data: Dict,
        initial_historical_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Predict sequence of days (with each prediction informing the next)
        
        Args:
            dates: List of dates to predict
            studio_data: Studio's current state
            initial_historical_data: Initial historical context
            
        Returns:
            DataFrame with all daily predictions
        """
        predictions = []
        
        # Initialize historical data buffer
        if initial_historical_data is not None:
            historical_buffer = initial_historical_data.copy()
        else:
            historical_buffer = pd.DataFrame()
        
        logger.info(f"Predicting {len(dates)} days from {dates[0]} to {dates[-1]}")
        
        for i, date in enumerate(dates):
            # Predict current day
            daily_pred = self.predict_daily(date, studio_data, historical_buffer)
            predictions.append(daily_pred)
            
            # Add prediction to historical buffer for next iteration
            new_row = pd.DataFrame([{
                'date': date,
                'total_revenue': daily_pred['daily_revenue'],
                'total_members': daily_pred['daily_members']
            }])
            historical_buffer = pd.concat([historical_buffer, new_row], ignore_index=True)
            
            # Keep only last 30 days in buffer
            if len(historical_buffer) > 30:
                historical_buffer = historical_buffer.tail(30)
        
        return pd.DataFrame(predictions)
    
    def aggregate_daily_to_monthly(self, daily_predictions: pd.DataFrame) -> List[Dict]:
        """
        Aggregate daily predictions to monthly forecasts
        
        Args:
            daily_predictions: DataFrame with daily predictions
            
        Returns:
            List of monthly prediction dictionaries
        """
        # Convert date column to datetime if needed
        daily_predictions = daily_predictions.copy()
        daily_predictions['date'] = pd.to_datetime(daily_predictions['date'])
        daily_predictions['year_month'] = daily_predictions['date'].dt.to_period('M')
        
        # Group by month
        monthly_groups = daily_predictions.groupby('year_month')
        
        monthly_predictions = []
        for month_idx, (month_period, month_data) in enumerate(monthly_groups, 1):
            # Sum revenue
            total_revenue = month_data['daily_revenue'].sum()
            
            # Average members and retention
            avg_members = month_data['daily_members'].mean()
            avg_retention = month_data['daily_retention'].mean()
            
            # Calculate confidence based on prediction consistency
            revenue_std = month_data['daily_revenue'].std()
            revenue_cv = revenue_std / (total_revenue / len(month_data) + 1)  # Coefficient of variation
            confidence = max(0.5, min(0.95, 1.0 - revenue_cv))
            
            monthly_predictions.append({
                'month': month_idx,
                'revenue': round(total_revenue, 2),
                'member_count': int(round(avg_members)),
                'retention_rate': round(avg_retention, 4),
                'confidence_score': round(confidence, 4),
                'num_days': len(month_data),
                'month_period': str(month_period)
            })
        
        logger.info(f"Aggregated {len(daily_predictions)} daily predictions to {len(monthly_predictions)} months")
        
        return monthly_predictions
    
    def predict_monthly_from_daily(
        self,
        studio_data: Dict,
        start_month: str,
        num_months: int = 3,
        historical_daily_data: pd.DataFrame = None
    ) -> Tuple[List[Dict], pd.DataFrame]:
        """
        Main method: Predict monthly outcomes using daily predictions
        
        Args:
            studio_data: Studio's current state
            start_month: Starting month 'YYYY-MM'
            num_months: Number of months to predict
            historical_daily_data: Recent historical daily data for context
            
        Returns:
            Tuple of (monthly_predictions, daily_predictions_df)
        """
        logger.info(f"Generating monthly forecast from daily predictions: {start_month} + {num_months} months")
        
        # Generate date range
        dates = self.generate_daily_dates(start_month, num_months)
        
        # Predict daily sequence
        daily_predictions = self.predict_daily_sequence(dates, studio_data, historical_daily_data)
        
        # Aggregate to monthly
        monthly_predictions = self.aggregate_daily_to_monthly(daily_predictions)
        
        return monthly_predictions, daily_predictions

