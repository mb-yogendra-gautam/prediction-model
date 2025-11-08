"""
Product/Service Analyzer Service

Analyzes correlations between products/services and levers/revenue
to provide data-driven recommendations on what to promote or demote.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ProductServiceAnalyzer:
    """Analyze product/service performance and generate recommendations"""
    
    # Product/Service definitions
    PRODUCTS = {
        'memberships': [
            'basic_membership',
            'premium_membership',
            'family_membership'
        ],
        'class_packages': [
            'drop_in_class',
            'class_pack_10',
            'class_pack_20',
            'unlimited_class'
        ],
        'retail': [
            'apparel',
            'supplements',
            'equipment'
        ],
        'add_on_services': [
            'personal_training',
            'nutrition_coaching',
            'wellness_services'
        ]
    }
    
    # Levers to analyze correlations against
    LEVERS = [
        'retention_rate',
        'avg_ticket_price',
        'class_attendance_rate',
        'new_members',
        'upsell_rate',
        'total_members'
    ]
    
    # Revenue targets
    REVENUE_TARGETS = [
        'total_revenue',
        'revenue_month_1',
        'revenue_month_2',
        'revenue_month_3'
    ]
    
    def __init__(self, training_data: pd.DataFrame):
        """
        Initialize analyzer with training data
        
        Args:
            training_data: DataFrame containing training data with product/service columns
        """
        self.training_data = training_data
        self.product_lever_correlations = None
        self.product_revenue_correlations = None
        self.product_statistics = None
        self.loaded_from_artifacts = False
        
        logger.info("ProductServiceAnalyzer initialized")
        self._compute_correlations()
    
    @classmethod
    def from_artifacts(cls, artifacts: Dict[str, Any]) -> 'ProductServiceAnalyzer':
        """
        Initialize analyzer from pre-computed correlation artifacts
        
        Args:
            artifacts: Dictionary containing pre-computed correlations:
                - product_lever_correlations
                - product_revenue_correlations
                - product_statistics
                - correlation_matrix (optional)
        
        Returns:
            ProductServiceAnalyzer instance with pre-loaded correlations
        """
        # Create an instance without calling __init__
        instance = cls.__new__(cls)
        
        # Set attributes directly from artifacts
        instance.training_data = None  # Not needed when loading from artifacts
        instance.product_lever_correlations = artifacts.get('product_lever_correlations', {})
        instance.product_revenue_correlations = artifacts.get('product_revenue_correlations', {})
        instance.product_statistics = artifacts.get('product_statistics', {})
        instance.loaded_from_artifacts = True
        
        logger.info(f"ProductServiceAnalyzer initialized from artifacts (fast path)")
        logger.info(f"Loaded {len(instance.product_lever_correlations)} products")
        
        return instance
    
    def _compute_correlations(self):
        """Compute correlation matrices"""
        logger.info("Computing product-lever and product-revenue correlations...")
        
        # Get product revenue columns
        product_revenue_cols = self._get_product_revenue_columns()
        
        # Compute correlations with levers
        self.product_lever_correlations = {}
        for product_col in product_revenue_cols:
            if product_col in self.training_data.columns:
                correlations = {}
                for lever in self.LEVERS:
                    if lever in self.training_data.columns:
                        corr = self.training_data[product_col].corr(self.training_data[lever])
                        correlations[lever] = corr
                self.product_lever_correlations[product_col] = correlations
        
        # Compute correlations with revenue targets
        self.product_revenue_correlations = {}
        for product_col in product_revenue_cols:
            if product_col in self.training_data.columns:
                correlations = {}
                for target in self.REVENUE_TARGETS:
                    if target in self.training_data.columns:
                        corr = self.training_data[product_col].corr(self.training_data[target])
                        correlations[target] = corr
                    # If target doesn't exist, use total_revenue as proxy
                    elif 'total_revenue' in self.training_data.columns:
                        corr = self.training_data[product_col].corr(self.training_data['total_revenue'])
                        correlations[target] = corr
                self.product_revenue_correlations[product_col] = correlations
        
        # Compute product statistics
        self.product_statistics = {}
        for product_col in product_revenue_cols:
            if product_col in self.training_data.columns:
                self.product_statistics[product_col] = {
                    'mean': self.training_data[product_col].mean(),
                    'std': self.training_data[product_col].std(),
                    'median': self.training_data[product_col].median(),
                    'min': self.training_data[product_col].min(),
                    'max': self.training_data[product_col].max()
                }
        
        logger.info(f"Computed correlations for {len(product_revenue_cols)} product/service types")
    
    def _get_product_revenue_columns(self) -> List[str]:
        """Get list of product revenue columns from training data"""
        product_cols = []
        
        # Flatten product definitions
        for category, products in self.PRODUCTS.items():
            for product in products:
                revenue_col = f"{product}_revenue"
                if revenue_col in self.training_data.columns:
                    product_cols.append(revenue_col)
        
        return product_cols
    
    def _get_product_count_columns(self) -> List[str]:
        """Get list of product count columns from training data"""
        product_cols = []
        
        # Flatten product definitions
        for category, products in self.PRODUCTS.items():
            for product in products:
                count_col = f"{product}_count"
                if count_col in self.training_data.columns:
                    product_cols.append(count_col)
        
        return product_cols
    
    def get_high_impact_products(
        self,
        metric: str = 'total_revenue',
        correlation_threshold: float = 0.60,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Identify products with highest positive impact on specified metric
        
        Args:
            metric: Target metric (lever or revenue)
            correlation_threshold: Minimum correlation to consider
            top_n: Number of top products to return
            
        Returns:
            List of product recommendations with correlation scores
        """
        high_impact = []
        
        # Check if metric is a lever or revenue target
        if metric in self.LEVERS:
            correlations = self.product_lever_correlations
        elif metric in self.REVENUE_TARGETS or metric == 'total_revenue':
            correlations = self.product_revenue_correlations
            # Map total_revenue to available target
            if metric == 'total_revenue' and metric not in self.REVENUE_TARGETS:
                metric = self.REVENUE_TARGETS[0]  # Use first available
        else:
            logger.warning(f"Unknown metric: {metric}")
            return []
        
        # Find products with strong positive correlations
        for product, corr_dict in correlations.items():
            if metric in corr_dict:
                correlation = corr_dict[metric]
                if correlation >= correlation_threshold and not np.isnan(correlation):
                    # Get product display name
                    product_name = product.replace('_revenue', '').replace('_', ' ').title()
                    
                    # Estimate revenue impact (use product statistics)
                    stats = self.product_statistics.get(product, {})
                    avg_revenue = stats.get('mean', 0)
                    
                    high_impact.append({
                        'product': product_name,
                        'product_key': product,
                        'correlation': correlation,
                        'avg_revenue': avg_revenue,
                        'impact_score': correlation * avg_revenue,  # Combined score
                        'category': self._get_product_category(product)
                    })
        
        # Sort by impact score (correlation * revenue magnitude)
        high_impact.sort(key=lambda x: x['impact_score'], reverse=True)
        
        return high_impact[:top_n]
    
    def get_underperforming_products(
        self,
        metric: str = 'total_revenue',
        correlation_threshold: float = 0.30,
        top_n: int = 3
    ) -> List[Dict]:
        """
        Identify products with weak or negative correlation to specified metric
        
        Args:
            metric: Target metric (lever or revenue)
            correlation_threshold: Maximum correlation to consider underperforming
            top_n: Number of products to return
            
        Returns:
            List of underperforming products
        """
        underperforming = []
        
        # Check if metric is a lever or revenue target
        if metric in self.LEVERS:
            correlations = self.product_lever_correlations
        elif metric in self.REVENUE_TARGETS or metric == 'total_revenue':
            correlations = self.product_revenue_correlations
            # Map total_revenue to available target
            if metric == 'total_revenue' and metric not in self.REVENUE_TARGETS:
                metric = self.REVENUE_TARGETS[0]
        else:
            logger.warning(f"Unknown metric: {metric}")
            return []
        
        # Find products with weak correlations
        for product, corr_dict in correlations.items():
            if metric in corr_dict:
                correlation = corr_dict[metric]
                if correlation < correlation_threshold and not np.isnan(correlation):
                    # Get product display name
                    product_name = product.replace('_revenue', '').replace('_', ' ').title()
                    
                    # Get product statistics
                    stats = self.product_statistics.get(product, {})
                    avg_revenue = stats.get('mean', 0)
                    
                    underperforming.append({
                        'product': product_name,
                        'product_key': product,
                        'correlation': correlation,
                        'avg_revenue': avg_revenue,
                        'category': self._get_product_category(product)
                    })
        
        # Sort by correlation (lowest first)
        underperforming.sort(key=lambda x: x['correlation'])
        
        return underperforming[:top_n]
    
    def _get_product_category(self, product_key: str) -> str:
        """Get category for a product key"""
        product_name = product_key.replace('_revenue', '').replace('_count', '')
        
        for category, products in self.PRODUCTS.items():
            if product_name in products:
                return category.replace('_', ' ').title()
        
        return "Unknown"
    
    def get_product_recommendations(
        self,
        current_levers: Dict[str, float],
        target_metric: str = 'total_revenue'
    ) -> Dict:
        """
        Get comprehensive product recommendations based on current levers
        
        Args:
            current_levers: Current lever values
            target_metric: Target metric to optimize for
            
        Returns:
            Dictionary with promote and demote recommendations
        """
        logger.info(f"Generating product recommendations for {target_metric}")
        
        # Analyze which levers are strong/weak in current state
        lever_strengths = self._analyze_lever_strengths(current_levers)
        
        # Get products to promote (high correlation with target)
        promote = self.get_high_impact_products(
            metric=target_metric,
            correlation_threshold=0.50,
            top_n=5
        )
        
        # Get products to review/demote (low correlation)
        demote = self.get_underperforming_products(
            metric=target_metric,
            correlation_threshold=0.30,
            top_n=3
        )
        
        # Add reasoning based on lever correlations
        for item in promote:
            item['reasoning'] = self._generate_promotion_reasoning(
                item['product_key'],
                lever_strengths
            )
        
        for item in demote:
            item['reasoning'] = self._generate_demotion_reasoning(
                item['product_key'],
                lever_strengths
            )
        
        return {
            'promote': promote,
            'demote': demote,
            'lever_analysis': lever_strengths
        }
    
    def _analyze_lever_strengths(self, current_levers: Dict[str, float]) -> Dict:
        """Analyze which levers are performing well or poorly"""
        strengths = {}
        
        # Define baseline/target values for levers
        lever_targets = {
            'retention_rate': 0.75,
            'class_attendance_rate': 0.70,
            'upsell_rate': 0.20,
            'avg_ticket_price': 150.0
        }
        
        for lever, value in current_levers.items():
            if lever in lever_targets:
                target = lever_targets[lever]
                performance = value / target
                
                if performance >= 1.05:
                    strength = 'strong'
                elif performance >= 0.95:
                    strength = 'moderate'
                else:
                    strength = 'weak'
                
                strengths[lever] = {
                    'current': value,
                    'target': target,
                    'performance': performance,
                    'strength': strength
                }
        
        return strengths
    
    def _generate_promotion_reasoning(
        self,
        product_key: str,
        lever_strengths: Dict
    ) -> str:
        """Generate reasoning for why to promote a product"""
        # Get lever correlations for this product
        lever_corrs = self.product_lever_correlations.get(product_key, {})
        
        # Find strongest lever correlations
        top_levers = sorted(
            [(lever, corr) for lever, corr in lever_corrs.items() if not np.isnan(corr)],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:2]
        
        product_name = product_key.replace('_revenue', '').replace('_', ' ').title()
        
        reasons = []
        for lever, corr in top_levers:
            lever_name = lever.replace('_', ' ').title()
            if corr > 0:
                reasons.append(f"Strong positive correlation with {lever_name} ({corr:.2f})")
        
        # Check if correlated levers are strong
        for lever, corr in top_levers:
            if lever in lever_strengths:
                strength = lever_strengths[lever]['strength']
                if strength == 'strong' and corr > 0:
                    reasons.append(f"Aligned with your strong {lever.replace('_', ' ')} performance")
        
        if not reasons:
            reasons.append("Consistently high revenue contributor")
        
        return "; ".join(reasons)
    
    def _generate_demotion_reasoning(
        self,
        product_key: str,
        lever_strengths: Dict
    ) -> str:
        """Generate reasoning for why to demote/review a product"""
        # Get lever correlations for this product
        lever_corrs = self.product_lever_correlations.get(product_key, {})
        
        product_name = product_key.replace('_revenue', '').replace('_', ' ').title()
        
        # Find correlations
        weak_corrs = [(lever, corr) for lever, corr in lever_corrs.items() 
                      if not np.isnan(corr) and abs(corr) < 0.30]
        
        if weak_corrs:
            return f"Weak correlation with key levers; consider bundling or repositioning"
        elif any(corr < 0 for corr in lever_corrs.values() if not np.isnan(corr)):
            return "Negative correlation with some levers; may need strategy revision"
        else:
            return "Limited impact on revenue; explore alternative offerings"
    
    def analyze_historical_patterns(
        self,
        product_key: str,
        months: int = 12
    ) -> Dict:
        """
        Analyze historical performance trends for a product
        
        Args:
            product_key: Product revenue column key
            months: Number of recent months to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        if product_key not in self.training_data.columns:
            logger.warning(f"Product {product_key} not found in data")
            return {}
        
        # Get recent data (last N months)
        recent_data = self.training_data.tail(months * 12)  # Approximate for multi-studio
        
        # Calculate trend
        product_values = recent_data[product_key].values
        if len(product_values) < 2:
            return {'trend': 'insufficient_data'}
        
        # Simple linear trend
        x = np.arange(len(product_values))
        slope = np.polyfit(x, product_values, 1)[0]
        
        # Determine trend direction
        if slope > 0.01:
            trend = 'increasing'
        elif slope < -0.01:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Calculate volatility
        volatility = np.std(product_values) / np.mean(product_values) if np.mean(product_values) > 0 else 0
        
        return {
            'trend': trend,
            'slope': slope,
            'volatility': volatility,
            'recent_mean': np.mean(product_values),
            'recent_std': np.std(product_values)
        }
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get full correlation matrix for all products vs levers
        
        Returns:
            DataFrame with correlations
        """
        product_cols = self._get_product_revenue_columns()
        
        correlation_data = []
        for product in product_cols:
            row = {'product': product}
            for lever in self.LEVERS:
                if product in self.product_lever_correlations:
                    corr = self.product_lever_correlations[product].get(lever, np.nan)
                    row[lever] = corr
            correlation_data.append(row)
        
        return pd.DataFrame(correlation_data)
    
    def save_correlation_artifacts(self, output_path: str):
        """
        Save correlation analysis as artifacts for model
        
        Args:
            output_path: Path to save pickle file
        """
        import pickle
        
        artifacts = {
            'product_lever_correlations': self.product_lever_correlations,
            'product_revenue_correlations': self.product_revenue_correlations,
            'product_statistics': self.product_statistics,
            'correlation_matrix': self.get_correlation_matrix().to_dict()
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(artifacts, f)
        
        logger.info(f"Correlation artifacts saved to {output_path}")

