"""
Historical Data Service for Daily Models (v2.3.0)

Provides daily-granularity historical data for:
- Rolling feature calculations (7d, 14d, 30d)
- Growth and momentum indicators
- Context for daily predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DailyHistoricalDataService:
    """Service for loading and providing daily historical data"""

    def __init__(self, data_path: str = "data/processed/multi_studio_daily_data_engineered.csv"):
        """
        Initialize daily historical data service

        Args:
            data_path: Path to daily engineered data CSV
        """
        self.data_path = Path(data_path)
        self.data = None
        self._load_data()

    def _load_data(self):
        """Load daily historical data into memory"""
        try:
            if not self.data_path.exists():
                logger.warning(f"Daily data not found at {self.data_path}, will use fallback generation")
                self.data = None
                return

            logger.info(f"Loading daily historical data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values(['studio_id', 'date'])

            logger.info(f"Loaded {len(self.data)} daily records for {self.data['studio_id'].nunique()} studios")

        except Exception as e:
            logger.error(f"Error loading daily data: {e}")
            self.data = None

    def get_studio_daily_history(self, studio_id: str, n_days: int = 30) -> pd.DataFrame:
        """
        Get last N days of history for a studio

        Args:
            studio_id: Studio identifier
            n_days: Number of days to retrieve

        Returns:
            DataFrame with daily records, or empty DataFrame if not found
        """
        if self.data is None:
            logger.warning(f"No daily data available, generating fallback for {studio_id}")
            return self._generate_fallback_daily_history(studio_id, n_days)

        studio_data = self.data[self.data['studio_id'] == studio_id]

        if len(studio_data) == 0:
            logger.warning(f"Studio {studio_id} not found in daily data, generating fallback")
            return self._generate_fallback_daily_history(studio_id, n_days)

        # Get last n_days
        recent_data = studio_data.tail(n_days).copy()

        logger.debug(f"Retrieved {len(recent_data)} days for studio {studio_id}")
        return recent_data

    def get_rolling_metrics(self, studio_id: str, window: int = 7) -> Dict[str, float]:
        """
        Get rolling average metrics for a studio

        Args:
            studio_id: Studio identifier
            window: Rolling window size in days

        Returns:
            Dictionary with rolling metrics
        """
        history = self.get_studio_daily_history(studio_id, n_days=window * 2)

        if len(history) < window:
            logger.warning(f"Insufficient history for studio {studio_id}, using available data")

        # Calculate rolling metrics
        metrics = {}

        if len(history) > 0:
            # Use last 'window' days for rolling calculations
            recent = history.tail(window)

            metrics['rolling_revenue'] = recent['daily_revenue'].mean() if 'daily_revenue' in recent.columns else 0.0
            metrics['rolling_attendance'] = recent['daily_attendance'].mean() if 'daily_attendance' in recent.columns else 0.0
            metrics['rolling_retention'] = recent['retention_rate'].mean() if 'retention_rate' in recent.columns else 0.75
            metrics['rolling_members'] = recent['total_members'].mean() if 'total_members' in recent.columns else 200

            # Growth metrics
            if len(recent) >= 2:
                first_revenue = recent.iloc[0]['daily_revenue'] if 'daily_revenue' in recent.columns else 2000
                last_revenue = recent.iloc[-1]['daily_revenue'] if 'daily_revenue' in recent.columns else 2000
                metrics['revenue_growth'] = ((last_revenue - first_revenue) / (first_revenue + 1)) if first_revenue > 0 else 0.0
            else:
                metrics['revenue_growth'] = 0.0

        else:
            # Fallback defaults
            metrics = {
                'rolling_revenue': 2000.0,
                'rolling_attendance': 85.0,
                'rolling_retention': 0.75,
                'rolling_members': 200.0,
                'revenue_growth': 0.02
            }

        return metrics

    def get_latest_studio_state(self, studio_id: str) -> Optional[Dict]:
        """
        Get most recent state for a studio

        Args:
            studio_id: Studio identifier

        Returns:
            Dictionary with latest studio metrics, or None if not found
        """
        history = self.get_studio_daily_history(studio_id, n_days=1)

        if len(history) == 0:
            return None

        latest = history.iloc[-1]

        state = {
            'date': latest['date'].strftime('%Y-%m-%d') if 'date' in history.columns else None,
            'daily_revenue': float(latest['daily_revenue']) if 'daily_revenue' in history.columns else 2000.0,
            'daily_attendance': int(latest['daily_attendance']) if 'daily_attendance' in history.columns else 85,
            'total_members': int(latest['total_members']) if 'total_members' in history.columns else 200,
            'retention_rate': float(latest['retention_rate']) if 'retention_rate' in history.columns else 0.75,
            'studio_location': latest['studio_location'] if 'studio_location' in history.columns else 'urban',
            'studio_size_tier': latest['studio_size_tier'] if 'studio_size_tier' in history.columns else 'medium',
            'studio_price_tier': latest['studio_price_tier'] if 'studio_price_tier' in history.columns else 'medium'
        }

        return state

    def _generate_fallback_daily_history(self, studio_id: str, n_days: int) -> pd.DataFrame:
        """
        Generate synthetic daily history when real data not available

        Args:
            studio_id: Studio identifier
            n_days: Number of days to generate

        Returns:
            DataFrame with synthetic daily records
        """
        logger.info(f"Generating {n_days} days of fallback data for {studio_id}")

        # Generate dates
        end_date = pd.Timestamp.now()
        dates = pd.date_range(end=end_date, periods=n_days, freq='D')

        # Generate synthetic data with realistic patterns
        data = []
        base_revenue = 2000 + np.random.randn() * 200
        base_attendance = 85 + int(np.random.randn() * 10)
        base_members = 200 + int(np.random.randn() * 30)

        for i, date in enumerate(dates):
            # Day-of-week pattern
            day_of_week = date.dayofweek
            dow_factor = 1.0
            if day_of_week == 0:  # Monday
                dow_factor = 1.15
            elif day_of_week in [5, 6]:  # Weekend
                dow_factor = 1.10
            elif day_of_week == 6:  # Sunday
                dow_factor = 0.75

            # Trend + noise
            trend_factor = 1 + (i / n_days) * 0.05  # 5% growth over period
            noise_factor = 1 + np.random.randn() * 0.08

            daily_revenue = base_revenue * dow_factor * trend_factor * noise_factor
            daily_attendance = int(base_attendance * dow_factor * trend_factor * (1 + np.random.randn() * 0.10))
            total_members = int(base_members * trend_factor * (1 + np.random.randn() * 0.05))

            record = {
                'studio_id': studio_id,
                'date': date,
                'daily_revenue': round(daily_revenue, 2),
                'daily_attendance': max(20, daily_attendance),
                'total_members': max(100, total_members),
                'retention_rate': 0.75 + np.random.randn() * 0.05,
                'studio_location': 'urban',
                'studio_size_tier': 'medium',
                'studio_price_tier': 'medium'
            }
            data.append(record)

        return pd.DataFrame(data)

    def get_day_of_week_context(self) -> Dict[str, any]:
        """
        Get current day-of-week context for predictions

        Returns:
            Dictionary with day-of-week features
        """
        now = pd.Timestamp.now()
        day_of_week = now.dayofweek  # 0=Monday, 6=Sunday

        context = {
            'day_of_week': day_of_week + 1,  # 1-7 for model
            'is_monday': 1 if day_of_week == 0 else 0,
            'is_friday': 1 if day_of_week == 4 else 0,
            'is_saturday': 1 if day_of_week == 5 else 0,
            'is_sunday': 1 if day_of_week == 6 else 0,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'month': now.month,
            'day_of_month': now.day,
            'week_of_year': now.isocalendar()[1]
        }

        return context

    def get_seasonal_context(self) -> Dict[str, int]:
        """
        Get seasonal flags for current date

        Returns:
            Dictionary with seasonal features
        """
        now = pd.Timestamp.now()
        month = now.month

        context = {
            'is_january': 1 if month == 1 else 0,
            'is_summer_prep': 1 if month in [5, 6] else 0,
            'is_holiday': 0,  # Simplified - would need holiday calendar
            'is_holiday_week': 0
        }

        return context
