"""
Explainability Service for Model Predictions using SHAP
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import shap

logger = logging.getLogger(__name__)


class ExplainabilityService:
    """Service for generating explanations for model predictions using SHAP"""

    # Target names for the 5 model outputs
    TARGET_NAMES = [
        'revenue_month_1',
        'revenue_month_2',
        'revenue_month_3',
        'member_count_month_3',
        'retention_rate_month_3'
    ]

    # Target units for formatting
    TARGET_UNITS = {
        'revenue_month_1': '$',
        'revenue_month_2': '$',
        'revenue_month_3': '$',
        'member_count_month_3': 'members',
        'retention_rate_month_3': 'rate'
    }

    def __init__(
        self,
        model,
        scaler,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None
    ):
        """
        Initialize explainability service

        Args:
            model: Trained MultiOutputRegressor with Ridge estimators
            scaler: StandardScaler for feature scaling
            feature_names: List of feature names in order
            background_data: Optional background dataset for SHAP (scaled features)
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.background_data = background_data

        # Initialize SHAP explainers for each target (5 total)
        self.explainers = {}
        self._initialize_explainers()

        logger.info(f"ExplainabilityService initialized with {len(feature_names)} features")
        logger.info(f"Background data: {'Provided' if background_data is not None else 'None - using zero baseline'}")

    def _initialize_explainers(self):
        """Initialize SHAP LinearExplainer for each of the 5 targets"""
        try:
            # For MultiOutputRegressor, we need separate explainer for each estimator
            for idx, target_name in enumerate(self.TARGET_NAMES):
                estimator = self.model.estimators_[idx]

                # LinearExplainer is fast and exact for Ridge regression
                if self.background_data is not None:
                    explainer = shap.LinearExplainer(estimator, self.background_data)
                else:
                    # Use zero baseline if no background data
                    explainer = shap.LinearExplainer(
                        estimator,
                        np.zeros((1, len(self.feature_names)))
                    )

                self.explainers[target_name] = explainer

            logger.info(f"Successfully initialized {len(self.explainers)} SHAP explainers")

        except Exception as e:
            logger.error(f"Error initializing SHAP explainers: {e}")
            raise

    def explain_prediction(
        self,
        features_scaled: np.ndarray,
        levers: Dict[str, float],
        targets_to_explain: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP-based explanations for a prediction

        Args:
            features_scaled: Scaled feature array (1, n_features)
            levers: Original lever values used for prediction
            targets_to_explain: List of target names to explain (default: all)

        Returns:
            Dictionary containing explanations for each target
        """
        if targets_to_explain is None:
            targets_to_explain = self.TARGET_NAMES

        logger.debug(f"Generating explanations for {len(targets_to_explain)} targets")

        explanations = {
            'method': 'SHAP (SHapley Additive exPlanations)',
            'targets': {},
            'feature_names': self.feature_names,
            'lever_values': levers
        }

        try:
            # Generate explanation for each target
            for target_name in targets_to_explain:
                if target_name not in self.explainers:
                    logger.warning(f"No explainer found for target: {target_name}")
                    continue

                target_explanation = self._explain_single_target(
                    target_name,
                    features_scaled
                )
                explanations['targets'][target_name] = target_explanation

            # Add global summary
            explanations['summary'] = self._generate_summary(explanations['targets'])

            return explanations

        except Exception as e:
            logger.error(f"Error generating explanations: {e}", exc_info=True)
            # Return fallback explanation
            return {
                'method': 'Coefficient-based (SHAP failed)',
                'error': str(e),
                'targets': self._fallback_coefficient_explanation(features_scaled)
            }

    def _explain_single_target(
        self,
        target_name: str,
        features_scaled: np.ndarray
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single target

        Args:
            target_name: Name of target to explain
            features_scaled: Scaled features

        Returns:
            Dictionary with SHAP values and analysis
        """
        explainer = self.explainers[target_name]

        # Get SHAP values
        shap_values = explainer.shap_values(features_scaled)
        base_value = explainer.expected_value

        # Create feature contributions
        contributions = []
        for i, feature_name in enumerate(self.feature_names):
            contributions.append({
                'feature': feature_name,
                'shap_value': float(shap_values[0][i]),
                'feature_value_scaled': float(features_scaled[0][i]),
                'abs_contribution': abs(float(shap_values[0][i]))
            })

        # Sort by absolute SHAP value (importance)
        contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)

        # Add importance ranks
        for rank, contrib in enumerate(contributions, 1):
            contrib['importance_rank'] = rank

        # Calculate prediction
        prediction = base_value + np.sum(shap_values[0])

        return {
            'target': target_name,
            'unit': self.TARGET_UNITS.get(target_name, ''),
            'prediction': float(prediction),
            'base_value': float(base_value),
            'total_shap_contribution': float(np.sum(shap_values[0])),
            'all_contributions': contributions,
            'top_5_drivers': contributions[:5],
            'formula': f"{base_value:.2f} (baseline) + {np.sum(shap_values[0]):.2f} (features) = {prediction:.2f}"
        }

    def _generate_summary(self, target_explanations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate overall summary across all targets

        Args:
            target_explanations: Dictionary of explanations per target

        Returns:
            Summary dictionary
        """
        # Find most important features across all targets
        feature_importance = {}

        for target_name, explanation in target_explanations.items():
            for contrib in explanation.get('all_contributions', []):
                feature = contrib['feature']
                if feature not in feature_importance:
                    feature_importance[feature] = 0
                feature_importance[feature] += contrib['abs_contribution']

        # Sort features by total importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'most_important_features_overall': [
                {'feature': feat, 'total_importance': float(importance)}
                for feat, importance in sorted_features[:10]
            ],
            'targets_explained': list(target_explanations.keys()),
            'explanation_count': len(target_explanations)
        }

    def _fallback_coefficient_explanation(
        self,
        features_scaled: np.ndarray
    ) -> Dict[str, Any]:
        """
        Fallback to coefficient-based explanation if SHAP fails

        Args:
            features_scaled: Scaled features

        Returns:
            Basic coefficient-based explanations
        """
        explanations = {}

        for idx, target_name in enumerate(self.TARGET_NAMES):
            estimator = self.model.estimators_[idx]
            coefficients = estimator.coef_
            intercept = estimator.intercept_

            # Calculate contributions
            contributions = features_scaled[0] * coefficients

            feature_contribs = []
            for i, feature_name in enumerate(self.feature_names):
                feature_contribs.append({
                    'feature': feature_name,
                    'coefficient': float(coefficients[i]),
                    'contribution': float(contributions[i]),
                    'abs_contribution': abs(float(contributions[i]))
                })

            # Sort by importance
            feature_contribs.sort(key=lambda x: x['abs_contribution'], reverse=True)

            explanations[target_name] = {
                'target': target_name,
                'prediction': float(intercept + np.sum(contributions)),
                'baseline': float(intercept),
                'top_5_drivers': feature_contribs[:5]
            }

        return explanations

    def get_feature_importance_global(self, target_idx: int = 0) -> List[Dict[str, Any]]:
        """
        Get global feature importance based on Ridge coefficients

        Args:
            target_idx: Index of target (0-4)

        Returns:
            List of features with their importance scores
        """
        estimator = self.model.estimators_[target_idx]
        coefficients = estimator.coef_

        importance_list = []
        for i, feature_name in enumerate(self.feature_names):
            importance_list.append({
                'feature': feature_name,
                'coefficient': float(coefficients[i]),
                'abs_coefficient': abs(float(coefficients[i])),
                'direction': 'positive' if coefficients[i] > 0 else 'negative'
            })

        # Sort by absolute coefficient
        importance_list.sort(key=lambda x: x['abs_coefficient'], reverse=True)

        # Add ranks
        for rank, item in enumerate(importance_list, 1):
            item['rank'] = rank

        return importance_list

    def explain_feature_to_lever_mapping(self) -> Dict[str, str]:
        """
        Map features to their source levers for better interpretability

        Returns:
            Dictionary mapping feature names to descriptions
        """
        # This will be enhanced with FeatureService integration
        lever_mappings = {
            # Direct lever features
            'total_members': 'Direct lever input',
            'new_members': 'Direct lever input',
            'retention_rate': 'Direct lever input',
            'avg_ticket_price': 'Direct lever input',
            'class_attendance_rate': 'Direct lever input',
            'total_classes_held': 'Direct lever input',
            'staff_utilization_rate': 'Direct lever input',
            'upsell_rate': 'Direct lever input',

            # Derived features
            'churned_members': 'Calculated from total_members × (1 - retention_rate)',
            'total_revenue': 'Sum of membership + class pack + retail revenue',
            'membership_revenue': 'Calculated from total_members × avg_ticket_price',
            'class_pack_revenue': 'Calculated from class metrics',
            'retail_revenue': 'Estimated from member count',
            'staff_count': 'Estimated from total_members',
            'estimated_ltv': 'Calculated from avg_ticket_price × retention_rate × 12',

            # Historical features
            'prev_month_revenue': 'Previous month revenue from historical data',
            'prev_month_members': 'Previous month members from historical data',
            '3m_avg_revenue': '3-month rolling average revenue',
            '3m_avg_retention': '3-month rolling average retention',
            'revenue_momentum': 'Exponential weighted moving average of revenue',

            # Interaction features
            'retention_x_ticket': 'Interaction: retention_rate × avg_ticket_price',
            'attendance_x_classes': 'Interaction: class_attendance_rate × total_classes_held',
            'staff_util_x_members': 'Interaction: staff_utilization_rate × total_members',
            'upsell_x_members': 'Interaction: upsell_rate × total_members'
        }

        return lever_mappings
