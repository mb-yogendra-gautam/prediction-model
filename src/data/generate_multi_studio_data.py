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
        
        output_path = Path('data/raw/multi_studio_data_raw.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
        return df
    
    def _generate_studio_data(self, profile):
        """Generate time series data for a single studio"""
        
        # Create date range - one record per month
        start_date = datetime(2019, 1, 1)
        dates = pd.date_range(start=start_date, periods=self.months, freq='MS')  # MS = Month Start
        
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
            
            # ============================================
            # PRODUCT/SERVICE BREAKDOWN (26 new columns)
            # ============================================
            
            # MEMBERSHIP TYPES (6 columns)
            # Distribution influenced by price tier and retention
            # Basic: 40-50% of members, lower price
            # Premium: 30-40% of members, higher price, correlated with retention
            # Family: 10-20% of members, highest price, correlated with retention
            
            retention_factor = retention_rate / 0.70  # Normalize around 0.70 baseline
            price_tier_multiplier = {'low': 0.8, 'medium': 1.0, 'high': 1.2}[profile['price_tier']]
            
            # Basic membership: inversely correlated with retention (churners tend to be basic)
            basic_pct = 0.50 - (retention_factor - 1) * 0.10 + np.random.normal(0, 0.03)
            basic_pct = np.clip(basic_pct, 0.35, 0.60)
            basic_membership_count = int(members * basic_pct)
            basic_membership_revenue = basic_membership_count * avg_ticket * 0.35  # Reduced from 0.75 to bring revenue down
            
            # Premium membership: positively correlated with retention and attendance
            premium_pct = 0.35 + (retention_factor - 1) * 0.08 + (class_attendance_rate - 0.65) * 0.15
            premium_pct = np.clip(premium_pct, 0.25, 0.45)
            premium_membership_count = int(members * premium_pct)
            premium_membership_revenue = premium_membership_count * avg_ticket * 0.50  # Reduced from 1.2 to bring revenue down
            
            # Family membership: positively correlated with retention and location (suburban)
            family_bonus = 0.05 if profile['location_type'] == 'suburban' else 0
            family_pct = 0.15 + (retention_factor - 1) * 0.05 + family_bonus + np.random.normal(0, 0.02)
            family_pct = np.clip(family_pct, 0.08, 0.25)
            family_membership_count = int(members * family_pct)
            family_membership_revenue = family_membership_count * avg_ticket * 0.65  # Reduced from 1.5 to bring revenue down
            
            # CLASS PACKAGES (8 columns)
            # Distribution influenced by attendance rate and class participation
            # Drop-in: negatively correlated with attendance (casual users)
            # Class packs: moderately correlated with attendance
            # Unlimited: highly correlated with attendance and retention
            
            attendance_factor = class_attendance_rate / 0.65  # Normalize
            
            # Drop-in classes: casual users, inversely correlated with attendance commitment
            drop_in_pct = 0.25 - (attendance_factor - 1) * 0.10 + np.random.normal(0, 0.03)
            drop_in_pct = np.clip(drop_in_pct, 0.10, 0.35)
            drop_in_class_count = int(members * drop_in_pct)
            drop_in_class_revenue = drop_in_class_count * 12  # Reduced from $25 to $12
            
            # Class pack 10: moderate users
            class_pack_10_pct = 0.20 + (attendance_factor - 1) * 0.05
            class_pack_10_pct = np.clip(class_pack_10_pct, 0.12, 0.28)
            class_pack_10_count = int(members * class_pack_10_pct * 0.10)  # Reduced from 15% to 10%
            class_pack_10_revenue = class_pack_10_count * 100  # Reduced from $200 to $100
            
            # Class pack 20: committed users, correlated with attendance
            class_pack_20_pct = 0.15 + (attendance_factor - 1) * 0.08
            class_pack_20_pct = np.clip(class_pack_20_pct, 0.08, 0.22)
            class_pack_20_count = int(members * class_pack_20_pct * 0.08)  # Reduced from 12% to 8%
            class_pack_20_revenue = class_pack_20_count * 180  # Reduced from $350 to $180
            
            # Unlimited classes: highly committed, strongly correlated with attendance and retention
            unlimited_pct = 0.30 + (attendance_factor - 1) * 0.15 + (retention_factor - 1) * 0.10
            unlimited_pct = np.clip(unlimited_pct, 0.20, 0.50)
            unlimited_class_count = int(members * unlimited_pct)
            unlimited_class_revenue = unlimited_class_count * avg_ticket * 0.15  # Reduced from 0.40 to 0.15
            
            # RETAIL PRODUCTS (6 columns)
            # Distribution influenced by member engagement (attendance, upsell_rate)
            # Apparel: correlated with new members and upsell rate
            # Supplements: correlated with attendance and premium memberships
            # Equipment: correlated with retention and home workout trends
            
            engagement_factor = (class_attendance_rate * 0.5 + upsell_rate * 0.5)
            
            # Apparel sales: impulse buys, correlated with new members and upsell
            apparel_buy_rate = 0.08 + upsell_rate * 0.20 + (new_members / members) * 0.10
            apparel_buy_rate = np.clip(apparel_buy_rate, 0.05, 0.18)
            apparel_sales_count = int(members * apparel_buy_rate * 0.5)  # Reduced purchases by 50%
            apparel_revenue = apparel_sales_count * 22  # Reduced from $45 to $22
            
            # Supplements: health-conscious members, correlated with premium tier and attendance
            supplements_buy_rate = 0.12 + (premium_pct - 0.35) * 0.30 + (attendance_factor - 1) * 0.08
            supplements_buy_rate = np.clip(supplements_buy_rate, 0.06, 0.22)
            supplements_sales_count = int(members * supplements_buy_rate * 0.5)  # Reduced purchases by 50%
            supplements_revenue = supplements_sales_count * 28  # Reduced from $55 to $28
            
            # Equipment: home workout gear, slightly correlated with retention
            equipment_buy_rate = 0.06 + (retention_factor - 1) * 0.04 + np.random.normal(0, 0.01)
            equipment_buy_rate = np.clip(equipment_buy_rate, 0.03, 0.12)
            equipment_sales_count = int(members * equipment_buy_rate * 0.4)  # Reduced purchases by 60%
            equipment_revenue = equipment_sales_count * 40  # Reduced from $80 to $40
            
            # ADD-ON SERVICES (6 columns)
            # Distribution influenced by price tier, retention, and studio size
            # Personal training: premium service, correlated with retention and premium members
            # Nutrition coaching: wellness-focused, correlated with retention and supplements
            # Wellness services: premium feature, correlated with premium tier and retention
            
            premium_studio_factor = price_tier_multiplier
            
            # Personal training: high-value service, correlated with retention and premium tier
            pt_rate = 0.18 + (retention_factor - 1) * 0.10 + (premium_pct - 0.35) * 0.20
            pt_rate = np.clip(pt_rate * premium_studio_factor, 0.08, 0.35)
            personal_training_count = int(members * pt_rate * 0.4)  # Reduced adoption by 60%
            personal_training_revenue = personal_training_count * 140  # Reduced from $280 to $140
            
            # Nutrition coaching: wellness service, correlated with supplements and premium
            nutrition_rate = 0.10 + (supplements_buy_rate - 0.12) * 0.50 + (retention_factor - 1) * 0.06
            nutrition_rate = np.clip(nutrition_rate * premium_studio_factor, 0.04, 0.22)
            nutrition_coaching_count = int(members * nutrition_rate * 0.5)  # Reduced adoption by 50%
            nutrition_coaching_revenue = nutrition_coaching_count * 75  # Reduced from $150 to $75
            
            # Wellness services: massage, physical therapy, etc.
            wellness_rate = 0.08 + (retention_factor - 1) * 0.05 + (premium_pct - 0.35) * 0.15
            wellness_rate = np.clip(wellness_rate * premium_studio_factor, 0.03, 0.18)
            wellness_services_count = int(members * wellness_rate * 0.5)  # Reduced adoption by 50%
            wellness_services_revenue = wellness_services_count * 48  # Reduced from $95 to $48
            
            # ============================================
            # CALCULATE AGGREGATE REVENUES (sum of products)
            # ============================================
            
            # Total membership revenue = sum of all membership types
            membership_revenue = (
                basic_membership_revenue + 
                premium_membership_revenue + 
                family_membership_revenue
            )
            
            # Total class pack revenue = sum of all class packages
            class_pack_revenue = (
                drop_in_class_revenue + 
                class_pack_10_revenue + 
                class_pack_20_revenue + 
                unlimited_class_revenue
            )
            
            # Total retail revenue = sum of all retail products
            retail_revenue = (
                apparel_revenue + 
                supplements_revenue + 
                equipment_revenue
            )
            
            # Total add-on services revenue (not in original categories, but part of total)
            addon_services_revenue = (
                personal_training_revenue + 
                nutrition_coaching_revenue + 
                wellness_services_revenue
            )
            
            # TOTAL REVENUE = sum of ALL individual products/services
            total_revenue = (
                membership_revenue + 
                class_pack_revenue + 
                retail_revenue + 
                addon_services_revenue
            )
            
            # Recalculate avg_ticket_price based on actual total revenue
            avg_ticket = total_revenue / members if members > 0 else avg_ticket
            
            # # Future targets (simplified - assume similar patterns continue)
            # # In reality, these would have their own logic
            # revenue_month_1 = total_revenue * (1 + profile['growth_rate'] * 1) * (1 + np.random.normal(0, 0.02))
            # revenue_month_2 = total_revenue * (1 + profile['growth_rate'] * 2) * (1 + np.random.normal(0, 0.02))
            # revenue_month_3 = total_revenue * (1 + profile['growth_rate'] * 3) * (1 + np.random.normal(0, 0.02))
            
            # member_count_month_3 = int(members * (1 + profile['growth_rate'] * 3) * (1 + np.random.normal(0, 0.02)))
            # retention_rate_month_3 = retention_rate * (1 + np.random.normal(0, 0.01))
            # retention_rate_month_3 = np.clip(retention_rate_month_3, 0.60, 0.85)
            
            # Create record
            record = {
                'studio_id': profile['studio_id'],
                'month_year': date.strftime('%Y-%m'),
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
                
                # MEMBERSHIP TYPES (6 columns)
                'basic_membership_count': basic_membership_count,
                'basic_membership_revenue': round(basic_membership_revenue, 2),
                'premium_membership_count': premium_membership_count,
                'premium_membership_revenue': round(premium_membership_revenue, 2),
                'family_membership_count': family_membership_count,
                'family_membership_revenue': round(family_membership_revenue, 2),
                
                # CLASS PACKAGES (8 columns)
                'drop_in_class_count': drop_in_class_count,
                'drop_in_class_revenue': round(drop_in_class_revenue, 2),
                'class_pack_10_count': class_pack_10_count,
                'class_pack_10_revenue': round(class_pack_10_revenue, 2),
                'class_pack_20_count': class_pack_20_count,
                'class_pack_20_revenue': round(class_pack_20_revenue, 2),
                'unlimited_class_count': unlimited_class_count,
                'unlimited_class_revenue': round(unlimited_class_revenue, 2),
                
                # RETAIL PRODUCTS (6 columns)
                'apparel_sales_count': apparel_sales_count,
                'apparel_revenue': round(apparel_revenue, 2),
                'supplements_sales_count': supplements_sales_count,
                'supplements_revenue': round(supplements_revenue, 2),
                'equipment_sales_count': equipment_sales_count,
                'equipment_revenue': round(equipment_revenue, 2),
                
                # ADD-ON SERVICES (6 columns)
                'personal_training_count': personal_training_count,
                'personal_training_revenue': round(personal_training_revenue, 2),
                'nutrition_coaching_count': nutrition_coaching_count,
                'nutrition_coaching_revenue': round(nutrition_coaching_revenue, 2),
                'wellness_services_count': wellness_services_count,
                'wellness_services_revenue': round(wellness_services_revenue, 2),
                
                # Targets
                # 'revenue_month_1': round(revenue_month_1, 2),
                # 'revenue_month_2': round(revenue_month_2, 2),
                # 'revenue_month_3': round(revenue_month_3, 2),
                # 'member_count_month_3': member_count_month_3,
                # 'retention_rate_month_3': round(retention_rate_month_3, 2)
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

