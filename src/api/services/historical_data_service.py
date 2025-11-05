"""
Historical Data Service for Loading Studio Historical Data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class HistoricalDataService:
    """Service for fetching historical studio data"""

    def __init__(self, data_path: str = "data/processed/multi_studio_data_engineered.csv"):
        self.data_path = Path(data_path)
        self.data = None
        self._load_data()

    def _load_data(self):
        """Load the engineered multi-studio data"""
        try:
            logger.info(f"Loading historical data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.data)} records for {self.data['studio_id'].nunique()} studios")
        except FileNotFoundError:
            logger.error(f"Historical data file not found: {self.data_path}")
            logger.warning("Historical data service will use fallback synthetic data")
            self.data = None
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            self.data = None

    def get_studio_history(self, studio_id: str, n_months: int = 12) -> Optional[pd.DataFrame]:
        """
        Get historical data for a specific studio
        
        Args:
            studio_id: Studio identifier
            n_months: Number of recent months to retrieve
            
        Returns:
            DataFrame with historical data or None if not found
        """
        if self.data is None:
            logger.warning("No historical data available")
            return self._generate_fallback_history(studio_id, n_months)

        studio_data = self.data[self.data['studio_id'] == studio_id].copy()
        
        if len(studio_data) == 0:
            logger.warning(f"No historical data found for studio {studio_id}, using fallback")
            return self._generate_fallback_history(studio_id, n_months)

        # Sort by date and get last n months
        studio_data = studio_data.sort_values('month_year').tail(n_months)
        logger.info(f"Retrieved {len(studio_data)} months of history for studio {studio_id}")
        
        return studio_data

    def _generate_fallback_history(self, studio_id: str, n_months: int = 12) -> pd.DataFrame:
        """
        Generate synthetic fallback historical data when real data is not available
        
        Args:
            studio_id: Studio identifier
            n_months: Number of months to generate
            
        Returns:
            DataFrame with synthetic historical data
        """
        logger.info(f"Generating {n_months} months of fallback history for {studio_id}")
        
        # Generate realistic synthetic data
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n_months, freq='MS')
        
        data = []
        for i, date in enumerate(dates):
            # Add some realistic variation
            month = date.month
            seasonality = 1.1 if month == 1 else (0.9 if month in [6, 7, 8] else 1.0)
            
            record = {
                'studio_id': studio_id,
                'month_year': date,
                'total_members': int(180 + i * 2 + np.random.randint(-10, 10)),
                'new_members': int(20 * seasonality + np.random.randint(-5, 5)),
                'churned_members': int(15 + np.random.randint(-5, 5)),
                'retention_rate': np.clip(0.75 + np.random.normal(0, 0.03), 0.65, 0.85),
                'avg_ticket_price': 150.0 + i * 0.5 + np.random.uniform(-5, 5),
                'total_classes_held': int(100 + np.random.randint(-10, 10)),
                'total_class_attendance': int(1400 + np.random.randint(-100, 100)),
                'class_attendance_rate': np.clip(0.70 + np.random.normal(0, 0.05), 0.60, 0.80),
                'staff_count': 8,
                'staff_utilization_rate': np.clip(0.80 + np.random.normal(0, 0.03), 0.70, 0.90),
                'upsell_rate': np.clip(0.25 + np.random.normal(0, 0.03), 0.15, 0.35),
                'total_revenue': 0.0,  # Will calculate
                'membership_revenue': 0.0,
                'class_pack_revenue': 0.0,
                'retail_revenue': 0.0
            }
            
            # Calculate revenues
            record['membership_revenue'] = record['total_members'] * record['avg_ticket_price']
            record['class_pack_revenue'] = record['total_class_attendance'] * 0.2 * 15
            record['retail_revenue'] = record['total_members'] * 10 * 0.3
            record['total_revenue'] = (
                record['membership_revenue'] + 
                record['class_pack_revenue'] + 
                record['retail_revenue']
            )
            
            data.append(record)
        
        df = pd.DataFrame(data)
        logger.info(f"Generated fallback history with revenue range: ${df['total_revenue'].min():.2f} - ${df['total_revenue'].max():.2f}")
        
        return df

    def get_all_studio_ids(self) -> list:
        """Get list of all available studio IDs"""
        if self.data is None:
            return []
        return self.data['studio_id'].unique().tolist()

    def get_studio_summary(self, studio_id: str) -> dict:
        """
        Get summary statistics for a studio
        
        Args:
            studio_id: Studio identifier
            
        Returns:
            Dictionary with summary statistics
        """
        history = self.get_studio_history(studio_id, n_months=12)
        
        if history is None or len(history) == 0:
            return {}
        
        return {
            'studio_id': studio_id,
            'months_of_data': len(history),
            'avg_revenue': float(history['total_revenue'].mean()),
            'avg_members': float(history['total_members'].mean()),
            'avg_retention': float(history['retention_rate'].mean()),
            'latest_revenue': float(history['total_revenue'].iloc[-1]),
            'latest_members': int(history['total_members'].iloc[-1])
        }

