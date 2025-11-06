"""
Counterfactual Service for What-If Analysis and Scenario Exploration
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from copy import deepcopy

logger = logging.getLogger(__name__)


class CounterfactualService:
    """Service for generating counterfactual scenarios and what-if analysis"""

    # Lever constraints (min, max)
    DEFAULT_LEVER_CONSTRAINTS = {
        'retention_rate': (0.5, 1.0),
        'avg_ticket_price': (50.0, 500.0),
        'class_attendance_rate': (0.4, 1.0),
        'new_members': (0, 100),
        'staff_utilization_rate': (0.6, 1.0),
        'upsell_rate': (0.0, 0.5),
        'total_classes_held': (50, 500),
        'total_members': (50, 1000)
    }

    # Lever display names
    LEVER_DISPLAY_NAMES = {
        'retention_rate': 'Retention Rate',
        'avg_ticket_price': 'Average Ticket Price',
        'class_attendance_rate': 'Class Attendance Rate',
        'new_members': 'New Members',
        'staff_utilization_rate': 'Staff Utilization Rate',
        'upsell_rate': 'Upsell Rate',
        'total_classes_held': 'Total Classes Held',
        'total_members': 'Total Members'
    }

    def __init__(
        self,
        model,
        scaler,
        feature_service,
        historical_data_service,
        lever_constraints: Dict[str, Tuple[float, float]] = None
    ):
        """
        Initialize counterfactual service

        Args:
            model: Trained prediction model
            scaler: Feature scaler
            feature_service: Feature engineering service
            historical_data_service: Historical data service
            lever_constraints: Optional custom lever constraints
        """
        self.model = model
        self.scaler = scaler
        self.feature_service = feature_service
        self.historical_data_service = historical_data_service
        self.lever_constraints = lever_constraints or self.DEFAULT_LEVER_CONSTRAINTS

        logger.info("CounterfactualService initialized")

    def analyze_lever_sensitivity(
        self,
        studio_id: str,
        base_levers: Dict[str, float],
        levers_to_test: List[str] = None,
        change_percentages: List[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze how changes to each lever affect predictions

        Args:
            studio_id: Studio ID for historical context
            base_levers: Baseline lever values
            levers_to_test: List of lever names to test (default: all)
            change_percentages: Percentage changes to test (default: [-20, -10, 10, 20])

        Returns:
            Dictionary with sensitivity analysis results
        """
        if levers_to_test is None:
            levers_to_test = list(self.lever_constraints.keys())

        if change_percentages is None:
            change_percentages = [-20, -10, 10, 20]

        logger.info(f"Analyzing sensitivity for {len(levers_to_test)} levers")

        # Get baseline prediction
        historical_data = self.historical_data_service.get_studio_history(studio_id, n_months=12)
        baseline_prediction = self._make_prediction(base_levers, historical_data)

        sensitivity_results = {
            'baseline_levers': base_levers,
            'baseline_predictions': baseline_prediction,
            'lever_sensitivities': {}
        }

        # Test each lever
        for lever_name in levers_to_test:
            if lever_name not in base_levers:
                logger.warning(f"Lever {lever_name} not in base_levers, skipping")
                continue

            lever_results = self._test_lever_variations(
                lever_name,
                base_levers,
                historical_data,
                baseline_prediction,
                change_percentages
            )
            sensitivity_results['lever_sensitivities'][lever_name] = lever_results

        # Add summary rankings
        sensitivity_results['lever_impact_ranking'] = self._rank_lever_impacts(
            sensitivity_results['lever_sensitivities']
        )

        return sensitivity_results

    def _test_lever_variations(
        self,
        lever_name: str,
        base_levers: Dict[str, float],
        historical_data,
        baseline_prediction: Dict[str, float],
        change_percentages: List[float]
    ) -> Dict[str, Any]:
        """Test variations of a single lever"""
        base_value = base_levers[lever_name]
        min_val, max_val = self.lever_constraints.get(lever_name, (0, 1000))

        variations = []

        for change_pct in change_percentages:
            # Calculate new value
            new_value = base_value * (1 + change_pct / 100)

            # Clamp to constraints
            new_value = max(min_val, min(max_val, new_value))

            # Create modified levers
            modified_levers = deepcopy(base_levers)
            modified_levers[lever_name] = new_value

            # Make prediction
            prediction = self._make_prediction(modified_levers, historical_data)

            # Calculate impacts
            revenue_delta = prediction['revenue_month_1'] - baseline_prediction['revenue_month_1']
            revenue_pct_change = (revenue_delta / baseline_prediction['revenue_month_1']) * 100 if baseline_prediction['revenue_month_1'] > 0 else 0

            variations.append({
                'change_percent': change_pct,
                'old_value': base_value,
                'new_value': new_value,
                'actual_change_percent': ((new_value - base_value) / base_value) * 100 if base_value > 0 else 0,
                'revenue_month_1_delta': revenue_delta,
                'revenue_month_1_pct_change': revenue_pct_change,
                'new_predictions': prediction
            })

        # Calculate average sensitivity (revenue change per 1% lever change)
        avg_sensitivity = np.mean([
            abs(v['revenue_month_1_delta'] / v['actual_change_percent'])
            for v in variations
            if v['actual_change_percent'] != 0
        ]) if variations else 0

        return {
            'lever': lever_name,
            'display_name': self.LEVER_DISPLAY_NAMES.get(lever_name, lever_name),
            'base_value': base_value,
            'variations': variations,
            'average_sensitivity': float(avg_sensitivity),
            'description': f"${avg_sensitivity:.2f} revenue change per 1% change in {self.LEVER_DISPLAY_NAMES.get(lever_name, lever_name)}"
        }

    def _rank_lever_impacts(self, lever_sensitivities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank levers by their impact on revenue"""
        rankings = []

        for lever_name, sensitivity_data in lever_sensitivities.items():
            rankings.append({
                'lever': lever_name,
                'display_name': sensitivity_data['display_name'],
                'average_sensitivity': sensitivity_data['average_sensitivity']
            })

        # Sort by sensitivity (descending)
        rankings.sort(key=lambda x: x['average_sensitivity'], reverse=True)

        # Add rank
        for rank, item in enumerate(rankings, 1):
            item['rank'] = rank

        return rankings

    def find_target_revenue_scenario(
        self,
        studio_id: str,
        base_levers: Dict[str, float],
        target_revenue: float,
        levers_to_optimize: List[str] = None,
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Find lever combination to achieve target revenue

        Args:
            studio_id: Studio ID
            base_levers: Starting lever values
            target_revenue: Desired revenue (month 1)
            levers_to_optimize: Which levers to adjust (default: all)
            max_iterations: Maximum optimization iterations

        Returns:
            Dictionary with recommended lever values and predictions
        """
        from scipy.optimize import minimize

        if levers_to_optimize is None:
            levers_to_optimize = list(self.lever_constraints.keys())

        logger.info(f"Finding scenario to achieve ${target_revenue:,.2f} revenue")

        historical_data = self.historical_data_service.get_studio_history(studio_id, n_months=12)

        # Create initial guess from base levers
        initial_guess = [base_levers.get(lever, 0.5) for lever in levers_to_optimize]

        # Define objective function (minimize difference from target)
        def objective(x):
            levers = deepcopy(base_levers)
            for i, lever_name in enumerate(levers_to_optimize):
                levers[lever_name] = x[i]

            prediction = self._make_prediction(levers, historical_data)
            return abs(prediction['revenue_month_1'] - target_revenue)

        # Define bounds
        bounds = [self.lever_constraints.get(lever, (0, 1000)) for lever in levers_to_optimize]

        # Optimize
        result = minimize(
            objective,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': max_iterations}
        )

        # Extract optimized levers
        optimized_levers = deepcopy(base_levers)
        for i, lever_name in enumerate(levers_to_optimize):
            optimized_levers[lever_name] = result.x[i]

        # Get final prediction
        final_prediction = self._make_prediction(optimized_levers, historical_data)

        # Calculate changes
        lever_changes = []
        for lever_name in levers_to_optimize:
            old_val = base_levers.get(lever_name, 0)
            new_val = optimized_levers[lever_name]
            pct_change = ((new_val - old_val) / old_val * 100) if old_val > 0 else 0

            lever_changes.append({
                'lever': lever_name,
                'display_name': self.LEVER_DISPLAY_NAMES.get(lever_name, lever_name),
                'old_value': old_val,
                'new_value': new_val,
                'change_percent': pct_change,
                'actionable': self._is_lever_actionable(lever_name, pct_change)
            })

        return {
            'target_revenue': target_revenue,
            'achieved_revenue': final_prediction['revenue_month_1'],
            'difference': abs(final_prediction['revenue_month_1'] - target_revenue),
            'success': abs(final_prediction['revenue_month_1'] - target_revenue) < (target_revenue * 0.05),  # Within 5%
            'optimized_levers': optimized_levers,
            'lever_changes': lever_changes,
            'predictions': final_prediction,
            'optimization_details': {
                'iterations': result.nit,
                'success': result.success,
                'message': result.message
            }
        }

    def compare_scenarios(
        self,
        studio_id: str,
        scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Compare multiple scenarios side-by-side

        Args:
            studio_id: Studio ID
            scenarios: Dictionary of scenario_name -> lever_values

        Returns:
            Side-by-side comparison of predictions
        """
        logger.info(f"Comparing {len(scenarios)} scenarios")

        historical_data = self.historical_data_service.get_studio_history(studio_id, n_months=12)

        comparisons = []

        for scenario_name, levers in scenarios.items():
            prediction = self._make_prediction(levers, historical_data)

            comparisons.append({
                'scenario_name': scenario_name,
                'levers': levers,
                'predictions': prediction
            })

        # Find best scenario by revenue
        best_scenario = max(comparisons, key=lambda x: x['predictions']['revenue_month_1'])

        return {
            'scenarios': comparisons,
            'best_scenario_by_revenue': best_scenario['scenario_name'],
            'comparison_summary': self._create_comparison_summary(comparisons)
        }

    def _create_comparison_summary(self, comparisons: List[Dict]) -> Dict[str, Any]:
        """Create summary statistics across scenarios"""
        revenues = [c['predictions']['revenue_month_1'] for c in comparisons]
        members = [c['predictions']['member_count_month_3'] for c in comparisons]

        return {
            'revenue_range': {
                'min': min(revenues),
                'max': max(revenues),
                'difference': max(revenues) - min(revenues)
            },
            'member_count_range': {
                'min': min(members),
                'max': max(members),
                'difference': max(members) - min(members)
            }
        }

    def _make_prediction(self, levers: Dict[str, float], historical_data) -> Dict[str, float]:
        """Make prediction for given levers"""
        # Engineer features
        features = self.feature_service.engineer_features_from_levers(levers, historical_data)

        # Scale
        features_scaled = self.scaler.transform(features)

        # Predict
        predictions = self.model.predict(features_scaled)[0]

        return {
            'revenue_month_1': float(predictions[0]),
            'revenue_month_2': float(predictions[1]),
            'revenue_month_3': float(predictions[2]),
            'member_count_month_3': float(predictions[3]),
            'retention_rate_month_3': float(predictions[4])
        }

    def _is_lever_actionable(self, lever_name: str, pct_change: float) -> bool:
        """
        Determine if a lever change is actionable/realistic

        Args:
            lever_name: Name of lever
            pct_change: Percentage change

        Returns:
            True if change is realistic to implement
        """
        # Some levers are harder to change than others
        hard_to_change = ['total_members']  # Existing member base
        easy_to_change = ['avg_ticket_price', 'total_classes_held', 'upsell_rate']

        if lever_name in hard_to_change:
            return abs(pct_change) < 10  # Only small changes realistic
        elif lever_name in easy_to_change:
            return abs(pct_change) < 50  # More flexibility
        else:
            return abs(pct_change) < 25  # Moderate changes

    def generate_quick_wins(
        self,
        studio_id: str,
        base_levers: Dict[str, float],
        max_change_pct: float = 10
    ) -> List[Dict[str, Any]]:
        """
        Identify "quick win" opportunities - small lever changes with high impact

        Args:
            studio_id: Studio ID
            base_levers: Current lever values
            max_change_pct: Maximum allowed change percentage

        Returns:
            List of quick win recommendations
        """
        logger.info("Generating quick win recommendations")

        historical_data = self.historical_data_service.get_studio_history(studio_id, n_months=12)
        baseline_prediction = self._make_prediction(base_levers, historical_data)

        quick_wins = []

        # Test small improvements to each lever
        for lever_name in base_levers.keys():
            if lever_name not in self.lever_constraints:
                continue

            base_value = base_levers[lever_name]
            min_val, max_val = self.lever_constraints[lever_name]

            # Try a small increase
            new_value = base_value * (1 + max_change_pct / 100)
            new_value = max(min_val, min(max_val, new_value))

            # Test impact
            modified_levers = deepcopy(base_levers)
            modified_levers[lever_name] = new_value

            prediction = self._make_prediction(modified_levers, historical_data)
            revenue_delta = prediction['revenue_month_1'] - baseline_prediction['revenue_month_1']

            if revenue_delta > 0:  # Only positive impacts
                quick_wins.append({
                    'lever': lever_name,
                    'display_name': self.LEVER_DISPLAY_NAMES.get(lever_name, lever_name),
                    'current_value': base_value,
                    'recommended_value': new_value,
                    'change_percent': ((new_value - base_value) / base_value * 100) if base_value > 0 else 0,
                    'revenue_impact': revenue_delta,
                    'roi': revenue_delta / abs(new_value - base_value) if abs(new_value - base_value) > 0 else 0,
                    'effort': self._estimate_effort(lever_name),
                    'actionable': True
                })

        # Sort by revenue impact
        quick_wins.sort(key=lambda x: x['revenue_impact'], reverse=True)

        return quick_wins[:5]  # Top 5

    def _estimate_effort(self, lever_name: str) -> str:
        """Estimate implementation effort for lever change"""
        low_effort = ['avg_ticket_price', 'upsell_rate']
        medium_effort = ['class_attendance_rate', 'staff_utilization_rate', 'total_classes_held']
        high_effort = ['total_members', 'retention_rate', 'new_members']

        if lever_name in low_effort:
            return 'low'
        elif lever_name in medium_effort:
            return 'medium'
        else:
            return 'high'
