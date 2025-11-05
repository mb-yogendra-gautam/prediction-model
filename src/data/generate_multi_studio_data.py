"""
Multi-Studio Data Generator

Generates synthetic data for 12 studios with realistic characteristics
to demonstrate model performance with adequate data volume.

Each studio has unique profiles:
- Size: Small (80-100), Medium (100-130), Large (130-160 members)
- Location: Urban vs Suburban
- Growth trajectory: Stable, Growing, Declining
- Pricing tier: Low ($120), Medium ($150), High ($180)
- Seasonal patterns with studio-specific variations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiStudioDataGenerator:
    """Generate synthetic data for multiple fitness studios"""
    
    def __init__(self, n_studios=12, months=70, seed=42):
        self.n_studios = n_studios
        self.months = months
        self.seed = seed
        np.random.seed(seed)
        
        # Define studio profiles
        self.studio_profiles = self._create_studio_profiles()
        
    def _create_studio_profiles(self):
        """Define characteristics for each studio"""
        profiles = [
            # Large Urban Studios (Growing)
            {
                'studio_id': 'STU001',
                'name': 'Urban Elite Fitness',
                'location_type': 'urban',
                'size_tier': 'large',
                'base_members': 145,
                'growth_rate': 0.015,  # 1.5% monthly growth
                'retention_baseline': 0.75,
                'price_tier': 'high',
                'base_price': 180,
                'seasonality_strength': 0.08,
                'class_capacity': 25,
                'base_staff': 7
            },
            {
                'studio_id': 'STU002',
                'name': 'Downtown Power Studio',
                'location_type': 'urban',
                'size_tier': 'large',
                'base_members': 140,
                'growth_rate': 0.012,
                'retention_baseline': 0.73,
                'price_tier': 'high',
                'base_price': 175,
                'seasonality_strength': 0.10,
                'class_capacity': 24,
                'base_staff': 7
            },
            # Medium Urban Studios (Stable)
            {
                'studio_id': 'STU003',
                'name': 'City Center Fitness',
                'location_type': 'urban',
                'size_tier': 'medium',
                'base_members': 120,
                'growth_rate': 0.005,
                'retention_baseline': 0.72,
                'price_tier': 'medium',
                'base_price': 155,
                'seasonality_strength': 0.12,
                'class_capacity': 22,
                'base_staff': 6
            },
            {
                'studio_id': 'STU004',
                'name': 'Metro Wellness Hub',
                'location_type': 'urban',
                'size_tier': 'medium',
                'base_members': 115,
                'growth_rate': 0.008,
                'retention_baseline': 0.70,
                'price_tier': 'medium',
                'base_price': 150,
                'seasonality_strength': 0.11,
                'class_capacity': 20,
                'base_staff': 5
            },
            # Small Urban Studios (Growing)
            {
                'studio_id': 'STU005',
                'name': 'Urban Boutique Fitness',
                'location_type': 'urban',
                'size_tier': 'small',
                'base_members': 95,
                'growth_rate': 0.020,
                'retention_baseline': 0.68,
                'price_tier': 'medium',
                'base_price': 145,
                'seasonality_strength': 0.15,
                'class_capacity': 18,
                'base_staff': 4
            },
            {
                'studio_id': 'STU006',
                'name': 'City Studio Express',
                'location_type': 'urban',
                'size_tier': 'small',
                'base_members': 90,
                'growth_rate': 0.018,
                'retention_baseline': 0.67,
                'price_tier': 'low',
                'base_price': 125,
                'seasonality_strength': 0.14,
                'class_capacity': 16,
                'base_staff': 4
            },
            # Large Suburban Studios (Stable)
            {
                'studio_id': 'STU007',
                'name': 'Suburban Family Fitness',
                'location_type': 'suburban',
                'size_tier': 'large',
                'base_members': 150,
                'growth_rate': 0.007,
                'retention_baseline': 0.77,
                'price_tier': 'medium',
                'base_price': 145,
                'seasonality_strength': 0.18,
                'class_capacity': 26,
                'base_staff': 7
            },
            {
                'studio_id': 'STU008',
                'name': 'Suburban Wellness Center',
                'location_type': 'suburban',
                'size_tier': 'large',
                'base_members': 155,
                'growth_rate': 0.006,
                'retention_baseline': 0.78,
                'price_tier': 'medium',
                'base_price': 140,
                'seasonality_strength': 0.20,
                'class_capacity': 25,
                'base_staff': 7
            },
            # Medium Suburban Studios (Growing)
            {
                'studio_id': 'STU009',
                'name': 'Neighborhood Fitness',
                'location_type': 'suburban',
                'size_tier': 'medium',
                'base_members': 110,
                'growth_rate': 0.010,
                'retention_baseline': 0.74,
                'price_tier': 'low',
                'base_price': 130,
                'seasonality_strength': 0.16,
                'class_capacity': 20,
                'base_staff': 5
            },
            {
                'studio_id': 'STU010',
                'name': 'Community Health Studio',
                'location_type': 'suburban',
                'size_tier': 'medium',
                'base_members': 118,
                'growth_rate': 0.009,
                'retention_baseline': 0.73,
                'price_tier': 'low',
                'base_price': 128,
                'seasonality_strength': 0.17,
                'class_capacity': 22,
                'base_staff': 6
            },
            # Small Suburban Studios (Declining/Stable)
            {
                'studio_id': 'STU011',
                'name': 'Local Fitness Corner',
                'location_type': 'suburban',
                'size_tier': 'small',
                'base_members': 85,
                'growth_rate': -0.002,  # Slight decline
                'retention_baseline': 0.66,
                'price_tier': 'low',
                'base_price': 120,
                'seasonality_strength': 0.22,
                'class_capacity': 15,
                'base_staff': 4
            },
            {
                'studio_id': 'STU012',
                'name': 'Suburban Fit Studio',
                'location_type': 'suburban',
                'size_tier': 'small',
                'base_members': 88,
                'growth_rate': 0.003,
                'retention_baseline': 0.68,
                'price_tier': 'low',
                'base_price': 122,
                'seasonality_strength': 0.19,
                'class_capacity': 16,
                'base_staff': 4
            }
        ]
        
        return profiles
    
    def generate_data(self):
        """Generate data for all studios"""
        logger.info(f"Generating data for {self.n_studios} studios over {self.months} months...")
        
        all_data = []
        
        for profile in self.studio_profiles:
            logger.info(f"Generating data for {profile['studio_id']} - {profile['name']}")
            studio_data = self._generate_studio_data(profile)
            all_data.append(studio_data)
        
        # Combine all studios
        df = pd.concat(all_data, ignore_index=True)
        
        # Sort by studio and date
        df = df.sort_values(['studio_id', 'month_year']).reset_index(drop=True)
        
        logger.info(f"Generated {len(df)} total studio-months")
        
        return df
    
    def _generate_studio_data(self, profile):
        """Generate time series data for a single studio"""
        
        # Create date range
        start_date = datetime(2019, 1, 1)
        dates = [start_date + timedelta(days=30*i) for i in range(self.months)]
        
        data = []
        
        for month_idx, date in enumerate(dates):
            
            # Calculate trend component
            trend_factor = 1 + (profile['growth_rate'] * month_idx)
            
            # Calculate seasonal component
            month = date.month
            seasonal_factor = self._get_seasonal_factor(month, profile['seasonality_strength'])
            
            # Add random noise
            noise = np.random.normal(0, 0.03)
            
            # Calculate member count
            members = int(profile['base_members'] * trend_factor * seasonal_factor * (1 + noise))
            members = max(50, members)  # Minimum 50 members
            
            # Calculate other metrics
            retention_rate = profile['retention_baseline'] * seasonal_factor * (1 + noise * 0.5)
            retention_rate = np.clip(retention_rate, 0.60, 0.85)
            
            # New members based on growth
            new_members = int(members * (1 - retention_rate) * 1.2)  # Need to replace churned + grow
            churned_members = int(members * (1 - retention_rate))
            
            # Pricing with small variations
            avg_ticket = profile['base_price'] * (1 + np.random.normal(0, 0.05))
            
            # Revenue calculations
            membership_revenue = members * avg_ticket * 0.75  # 75% from memberships
            class_pack_revenue = members * avg_ticket * 0.20  # 20% from class packs
            retail_revenue = members * avg_ticket * 0.05  # 5% from retail
            total_revenue = membership_revenue + class_pack_revenue + retail_revenue
            
            # Class metrics
            classes_per_member = 8 + np.random.normal(0, 1.5)
            classes_per_member = max(5, classes_per_member)
            total_classes_held = int(members * classes_per_member / profile['class_capacity'])
            total_class_attendance = int(total_classes_held * profile['class_capacity'] * 
                                        (0.65 + np.random.normal(0, 0.05)))
            class_attendance_rate = total_class_attendance / (total_classes_held * profile['class_capacity'])
            class_attendance_rate = np.clip(class_attendance_rate, 0.55, 0.80)
            
            # Staff metrics
            staff_count = profile['base_staff'] + int((members - profile['base_members']) / 25)
            staff_count = max(3, staff_count)
            
            # Upsell metrics
            upsell_rate = 0.15 + np.random.normal(0, 0.03)
            upsell_rate = np.clip(upsell_rate, 0.08, 0.25)
            
            # Future targets (simplified - assume similar patterns continue)
            # In reality, these would have their own logic
            revenue_month_1 = total_revenue * (1 + profile['growth_rate'] * 1) * (1 + np.random.normal(0, 0.02))
            revenue_month_2 = total_revenue * (1 + profile['growth_rate'] * 2) * (1 + np.random.normal(0, 0.02))
            revenue_month_3 = total_revenue * (1 + profile['growth_rate'] * 3) * (1 + np.random.normal(0, 0.02))
            
            member_count_month_3 = int(members * (1 + profile['growth_rate'] * 3) * (1 + np.random.normal(0, 0.02)))
            retention_rate_month_3 = retention_rate * (1 + np.random.normal(0, 0.01))
            retention_rate_month_3 = np.clip(retention_rate_month_3, 0.60, 0.85)
            
            # Create record
            record = {
                'studio_id': profile['studio_id'],
                'month_year': date.strftime('%Y-%m-%d'),
                'total_members': members,
                'new_members': new_members,
                'churned_members': churned_members,
                'retention_rate': round(retention_rate, 2),
                'avg_ticket_price': round(avg_ticket, 2),
                'total_revenue': round(total_revenue, 2),
                'membership_revenue': round(membership_revenue, 2),
                'class_pack_revenue': round(class_pack_revenue, 2),
                'retail_revenue': round(retail_revenue, 2),
                'total_classes_held': total_classes_held,
                'total_class_attendance': total_class_attendance,
                'class_attendance_rate': round(class_attendance_rate, 2),
                'staff_count': staff_count,
                'avg_classes_per_member': round(classes_per_member, 2),
                'upsell_rate': round(upsell_rate, 2),
                'studio_location': profile['location_type'],
                'studio_size_tier': profile['size_tier'],
                'studio_price_tier': profile['price_tier'],
                # Targets
                'revenue_month_1': round(revenue_month_1, 2),
                'revenue_month_2': round(revenue_month_2, 2),
                'revenue_month_3': round(revenue_month_3, 2),
                'member_count_month_3': member_count_month_3,
                'retention_rate_month_3': round(retention_rate_month_3, 2)
            }
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def _get_seasonal_factor(self, month, strength):
        """Calculate seasonal adjustment factor"""
        # January: New Year resolutions (high)
        # Summer (Jun-Aug): Vacation season (lower)
        # Fall (Sep-Oct): Back to routine (high)
        # December: Holidays (lower)
        
        seasonal_pattern = {
            1: 1.15,   # January - New Year peak
            2: 1.08,   # February
            3: 1.02,   # March
            4: 0.98,   # April
            5: 0.95,   # May
            6: 0.88,   # June - Summer dip starts
            7: 0.85,   # July - Summer low
            8: 0.87,   # August
            9: 1.05,   # September - Back to routine
            10: 1.03,  # October
            11: 0.97,  # November
            12: 0.90   # December - Holiday season
        }
        
        base_factor = seasonal_pattern[month]
        
        # Apply studio-specific seasonality strength
        adjustment = (base_factor - 1) * strength
        
        return 1 + adjustment
    
    def add_data_splits(self, df):
        """Add train/validation/test splits stratified by studio"""
        logger.info("Adding data splits...")
        
        df = df.sort_values(['studio_id', 'month_year']).reset_index(drop=True)
        
        splits = []
        
        for studio_id in df['studio_id'].unique():
            studio_df = df[df['studio_id'] == studio_id].copy()
            n = len(studio_df)
            
            # 75% train, 15% validation, 10% test (temporal split)
            train_end = int(n * 0.75)
            val_end = int(n * 0.90)
            
            studio_splits = ['train'] * train_end + \
                          ['validation'] * (val_end - train_end) + \
                          ['test'] * (n - val_end)
            
            splits.extend(studio_splits)
        
        df['split'] = splits
        
        logger.info(f"Train: {sum(df['split'] == 'train')} samples")
        logger.info(f"Validation: {sum(df['split'] == 'validation')} samples")
        logger.info(f"Test: {sum(df['split'] == 'test')} samples")
        
        return df
    
    def save_data(self, df, output_path='data/processed/multi_studio_data.csv'):
        """Save generated data"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
        
        return output_path


def main():
    """Generate multi-studio dataset"""
    
    print("\n" + "="*80)
    print("MULTI-STUDIO DATA GENERATION")
    print("="*80 + "\n")
    
    # Initialize generator
    generator = MultiStudioDataGenerator(n_studios=12, months=70, seed=42)
    
    # Generate data
    df = generator.generate_data()
    
    # Add splits
    df = generator.add_data_splits(df)
    
    # Save data
    output_path = generator.save_data(df)
    
    # Print summary
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80)
    print(f"\nTotal studios: {df['studio_id'].nunique()}")
    print(f"Total studio-months: {len(df)}")
    print(f"Date range: {df['month_year'].min()} to {df['month_year'].max()}")
    print(f"\nData splits:")
    print(f"  Train: {sum(df['split'] == 'train')} ({sum(df['split'] == 'train')/len(df)*100:.1f}%)")
    print(f"  Validation: {sum(df['split'] == 'validation')} ({sum(df['split'] == 'validation')/len(df)*100:.1f}%)")
    print(f"  Test: {sum(df['split'] == 'test')} ({sum(df['split'] == 'test')/len(df)*100:.1f}%)")
    
    print(f"\nStudio distribution:")
    for location in df['studio_location'].unique():
        for size in df['studio_size_tier'].unique():
            count = len(df[(df['studio_location'] == location) & (df['studio_size_tier'] == size)]['studio_id'].unique())
            if count > 0:
                print(f"  {location.capitalize()} {size.capitalize()}: {count} studios")
    
    print(f"\nSample statistics:")
    print(f"  Members: {df['total_members'].min():.0f} - {df['total_members'].max():.0f} (mean: {df['total_members'].mean():.0f})")
    print(f"  Revenue: ${df['total_revenue'].min():.0f} - ${df['total_revenue'].max():.0f} (mean: ${df['total_revenue'].mean():.0f})")
    print(f"  Retention: {df['retention_rate'].min():.2f} - {df['retention_rate'].max():.2f} (mean: {df['retention_rate'].mean():.2f})")
    
    print(f"\nData saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Run feature engineering: python src/features/run_feature_engineering.py")
    print("  2. Train model: python training/train_model_v2.2_multi_studio.py")
    print("  3. Evaluate: python training/evaluate_multi_studio_model.py")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

