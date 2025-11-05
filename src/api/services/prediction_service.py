"""
Prediction Service for Forward, Inverse, and Partial Predictions
"""

import numpy as np
import uuid
from typing import Dict, List, Any
from datetime import datetime
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
import logging

logger = logging.getLogger(__name__)


class PredictionService:
    """Core service for making predictions"""

    def __init__(
        self, 
        model, 
        scaler, 
        feature_service, 
        historical_data_service, 
        metadata: Dict
    ):
        """
        Initialize prediction service
        
        Args:
            model: Trained prediction model
            scaler: Feature scaler
            feature_service: Feature engineering service
            historical_data_service: Historical data service
            metadata: Model metadata
        """
        self.model = model
        self.scaler = scaler
        self.feature_service = feature_service
        self.historical_data_service = historical_data_service
        self.metadata = metadata
        self.version = metadata.get('version', 'unknown')
        
        # Lever constraints for optimization
        self.lever_constraints = {
            'retention_rate': (0.5, 1.0),
            'avg_ticket_price': (50.0, 500.0),
            'class_attendance_rate': (0.4, 1.0),
            'new_members': (0, 100),
            'staff_utilization_rate': (0.6, 1.0),
            'upsell_rate': (0.0, 0.5),
            'total_classes_held': (50, 500),
            'total_members': (50, 1000)
        }
        
        logger.info(f"Prediction service initialized with model version {self.version}")

    def predict_forward(self, request_data: Dict) -> Dict:
        """
        Forward prediction: levers → revenue
        
        Args:
            request_data: Dictionary with studio_id, levers, projection_months
            
        Returns:
            Dictionary with predictions and metadata
        """
        logger.info(f"Forward prediction for studio {request_data['studio_id']}")
        
        studio_id = request_data['studio_id']
        levers = request_data['levers']
        projection_months = request_data.get('projection_months', 3)
        
        # Get historical data
        historical_data = self.historical_data_service.get_studio_history(studio_id, n_months=12)
        
        # Engineer features
        features = self.feature_service.engineer_features_from_levers(levers, historical_data)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        predictions = self.model.predict(features_scaled)[0]
        
        # predictions shape: [revenue_month_1, revenue_month_2, revenue_month_3, member_count_month_3, retention_rate_month_3]
        monthly_predictions = []
        total_revenue = 0.0
        
        for i in range(min(projection_months, 3)):
            revenue = float(predictions[i])
            
            # Estimate member count for each month
            if i == 2:
                member_count = int(predictions[3])
            else:
                # Interpolate member count
                member_count = int(levers['total_members'] + (predictions[3] - levers['total_members']) * (i + 1) / 3)
            
            # Estimate retention for each month
            if i == 2:
                retention = float(predictions[4])
            else:
                retention = float(levers['retention_rate'])
            
            monthly_predictions.append({
                'month': i + 1,
                'revenue': revenue,
                'member_count': member_count,
                'retention_rate': retention,
                'confidence_score': 0.85 - (i * 0.05)  # Decreasing confidence over time
            })
            
            total_revenue += revenue
        
        # If projection_months > 3, extrapolate
        if projection_months > 3:
            # Use month 3 prediction as baseline for extrapolation
            last_revenue = monthly_predictions[-1]['revenue']
            last_members = monthly_predictions[-1]['member_count']
            growth_rate = (last_revenue - monthly_predictions[0]['revenue']) / (3 * monthly_predictions[0]['revenue'])
            
            for i in range(3, projection_months):
                revenue = last_revenue * (1 + growth_rate)
                monthly_predictions.append({
                    'month': i + 1,
                    'revenue': revenue,
                    'member_count': last_members,
                    'retention_rate': float(predictions[4]),
                    'confidence_score': max(0.5, 0.80 - (i * 0.05))
                })
                total_revenue += revenue
                last_revenue = revenue
        
        # Calculate average confidence
        avg_confidence = np.mean([p['confidence_score'] for p in monthly_predictions])
        
        response = {
            'scenario_id': str(uuid.uuid4()),
            'studio_id': studio_id,
            'predictions': monthly_predictions,
            'total_projected_revenue': float(total_revenue),
            'average_confidence': float(avg_confidence),
            'model_version': self.version,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Forward prediction complete: ${total_revenue:.2f} over {projection_months} months")
        return response

    def predict_inverse(self, request_data: Dict) -> Dict:
        """
        Inverse prediction: target revenue → optimal levers
        
        Args:
            request_data: Dictionary with studio_id, target_revenue, current_state, constraints
            
        Returns:
            Dictionary with optimized lever values and action plan
        """
        logger.info(f"Inverse prediction for studio {request_data['studio_id']}, target: ${request_data['target_revenue']:.2f}")
        
        studio_id = request_data['studio_id']
        target_revenue = request_data['target_revenue']
        current_state = request_data['current_state']
        constraints = request_data.get('constraints', {})
        target_months = request_data.get('target_months', 3)
        
        # Get historical data
        historical_data = self.historical_data_service.get_studio_history(studio_id, n_months=12)
        
        # Define objective function to minimize
        def objective(lever_values):
            """Minimize difference between predicted and target revenue"""
            levers_dict = {
                'retention_rate': lever_values[0],
                'avg_ticket_price': lever_values[1],
                'class_attendance_rate': lever_values[2],
                'new_members': int(lever_values[3]),
                'staff_utilization_rate': lever_values[4],
                'upsell_rate': lever_values[5],
                'total_classes_held': int(lever_values[6]),
                'total_members': int(lever_values[7])
            }
            
            # Engineer features
            features = self.feature_service.engineer_features_from_levers(levers_dict, historical_data)
            features_scaled = self.scaler.transform(features)
            
            # Predict revenue
            predictions = self.model.predict(features_scaled)[0]
            
            # Sum revenue for target months (max 3)
            predicted_revenue = sum(predictions[:min(target_months, 3)])
            
            # Return squared error
            return (predicted_revenue - target_revenue) ** 2
        
        # Initial values from current state
        x0 = [
            current_state['retention_rate'],
            current_state['avg_ticket_price'],
            current_state['class_attendance_rate'],
            current_state['new_members'],
            current_state['staff_utilization_rate'],
            current_state['upsell_rate'],
            current_state['total_classes_held'],
            current_state['total_members']
        ]
        
        # Set up bounds
        bounds = [
            self.lever_constraints['retention_rate'],
            self.lever_constraints['avg_ticket_price'],
            self.lever_constraints['class_attendance_rate'],
            self.lever_constraints['new_members'],
            self.lever_constraints['staff_utilization_rate'],
            self.lever_constraints['upsell_rate'],
            self.lever_constraints['total_classes_held'],
            self.lever_constraints['total_members']
        ]
        
        # Apply custom constraints if provided
        if constraints:
            max_ret_increase = constraints.get('max_retention_increase', 0.05)
            bounds[0] = (
                current_state['retention_rate'],
                min(current_state['retention_rate'] + max_ret_increase, 1.0)
            )
            
            max_ticket_increase = constraints.get('max_ticket_increase', 20.0)
            bounds[1] = (
                current_state['avg_ticket_price'],
                current_state['avg_ticket_price'] + max_ticket_increase
            )
            
            max_members_increase = constraints.get('max_new_members_increase', 10)
            bounds[3] = (
                current_state['new_members'],
                current_state['new_members'] + max_members_increase
            )
        
        # Optimize
        logger.info("Starting optimization...")
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        # Extract optimized levers
        optimized_levers = {
            'retention_rate': float(result.x[0]),
            'avg_ticket_price': float(result.x[1]),
            'class_attendance_rate': float(result.x[2]),
            'new_members': int(result.x[3]),
            'staff_utilization_rate': float(result.x[4]),
            'upsell_rate': float(result.x[5]),
            'total_classes_held': int(result.x[6]),
            'total_members': int(result.x[7])
        }
        
        # Calculate achievable revenue
        features = self.feature_service.engineer_features_from_levers(optimized_levers, historical_data)
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)[0]
        achievable_revenue = float(sum(predictions[:min(target_months, 3)]))
        
        # Calculate lever changes
        lever_changes = []
        for lever_name, optimized_value in optimized_levers.items():
            current_value = current_state[lever_name]
            change_abs = optimized_value - current_value
            change_pct = (change_abs / current_value * 100) if current_value != 0 else 0
            
            if abs(change_pct) > 1.0:  # Only include significant changes
                priority = self._calculate_priority(lever_name, abs(change_pct))
                lever_changes.append({
                    'lever_name': lever_name,
                    'current_value': float(current_value),
                    'recommended_value': float(optimized_value),
                    'change_absolute': float(change_abs),
                    'change_percentage': float(change_pct),
                    'priority': priority
                })
        
        # Sort by priority
        lever_changes.sort(key=lambda x: x['priority'])
        
        # Create action plan
        action_plan = self._create_action_plan(lever_changes, achievable_revenue - sum([current_state['avg_ticket_price'] * current_state['total_members'] for _ in range(min(target_months, 3))]) / target_months)
        
        response = {
            'optimization_id': str(uuid.uuid4()),
            'studio_id': studio_id,
            'target_revenue': float(target_revenue),
            'achievable_revenue': achievable_revenue,
            'achievement_rate': float(min(achievable_revenue / target_revenue, 1.0)),
            'recommended_levers': optimized_levers,
            'lever_changes': lever_changes,
            'action_plan': action_plan,
            'confidence_score': 0.75 if result.success else 0.60,
            'model_version': self.version,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Inverse prediction complete: ${achievable_revenue:.2f} achievable ({response['achievement_rate']*100:.1f}% of target)")
        return response

    def predict_partial_levers(self, request_data: Dict) -> Dict:
        """
        Partial lever prediction: subset of levers → remaining levers
        
        Args:
            request_data: Dictionary with studio_id, input_levers, output_levers
            
        Returns:
            Dictionary with predicted lever values
        """
        logger.info(f"Partial prediction for studio {request_data['studio_id']}")
        
        studio_id = request_data['studio_id']
        input_levers = request_data['input_levers']
        output_lever_names = request_data['output_levers']
        
        # Get historical data
        historical_data = self.historical_data_service.get_studio_history(studio_id, n_months=12)
        
        predicted_levers = []
        
        # For each output lever, predict its value
        for lever_name in output_lever_names:
            if lever_name == 'total_revenue':
                # Special case: predict revenue using forward model
                # Fill in missing levers with historical averages
                complete_levers = self._fill_missing_levers(input_levers, historical_data)
                features = self.feature_service.engineer_features_from_levers(complete_levers, historical_data)
                features_scaled = self.scaler.transform(features)
                predictions = self.model.predict(features_scaled)[0]
                predicted_value = float(predictions[0])  # Month 1 revenue
                confidence = 0.85
                value_range = [predicted_value * 0.9, predicted_value * 1.1]
            
            elif lever_name in self.lever_constraints:
                # Predict lever value based on historical patterns and relationships
                predicted_value, confidence = self._predict_single_lever(
                    lever_name, 
                    input_levers, 
                    historical_data
                )
                bounds = self.lever_constraints[lever_name]
                value_range = [max(bounds[0], predicted_value * 0.95), min(bounds[1], predicted_value * 1.05)]
            
            else:
                logger.warning(f"Unknown output lever: {lever_name}")
                continue
            
            predicted_levers.append({
                'lever_name': lever_name,
                'predicted_value': float(predicted_value),
                'confidence_score': float(confidence),
                'value_range': value_range
            })
        
        # Calculate overall confidence
        overall_confidence = np.mean([pl['confidence_score'] for pl in predicted_levers]) if predicted_levers else 0.0
        
        response = {
            'prediction_id': str(uuid.uuid4()),
            'studio_id': studio_id,
            'input_levers': input_levers,
            'predicted_levers': predicted_levers,
            'overall_confidence': float(overall_confidence),
            'model_version': self.version,
            'timestamp': datetime.now().isoformat(),
            'notes': 'Predictions based on historical patterns and model relationships'
        }
        
        logger.info(f"Partial prediction complete: {len(predicted_levers)} levers predicted")
        return response

    def _fill_missing_levers(self, input_levers: Dict, historical_data) -> Dict:
        """Fill in missing levers with historical averages or reasonable defaults"""
        complete_levers = input_levers.copy()
        
        defaults = {
            'retention_rate': 0.75,
            'avg_ticket_price': 150.0,
            'class_attendance_rate': 0.70,
            'new_members': 20,
            'staff_utilization_rate': 0.80,
            'upsell_rate': 0.25,
            'total_classes_held': 100,
            'total_members': 180
        }
        
        # Use historical averages if available
        if historical_data is not None and len(historical_data) > 0:
            for lever in defaults.keys():
                if lever not in complete_levers and lever in historical_data.columns:
                    complete_levers[lever] = float(historical_data[lever].mean())
        
        # Fill remaining with defaults
        for lever, default_value in defaults.items():
            if lever not in complete_levers:
                complete_levers[lever] = default_value
        
        return complete_levers

    def _predict_single_lever(self, lever_name: str, input_levers: Dict, historical_data) -> tuple:
        """
        Predict a single lever value based on input levers and historical data
        
        Returns:
            (predicted_value, confidence_score)
        """
        # Use historical average as baseline
        if historical_data is not None and len(historical_data) > 0 and lever_name in historical_data.columns:
            baseline = float(historical_data[lever_name].mean())
            std = float(historical_data[lever_name].std())
            
            # Apply some logic based on input levers
            if lever_name == 'retention_rate' and 'avg_ticket_price' in input_levers:
                # Lower prices might improve retention slightly
                price_effect = -0.01 if input_levers['avg_ticket_price'] < 130 else 0.01
                predicted_value = baseline + price_effect
                confidence = 0.70
            
            elif lever_name == 'class_attendance_rate' and 'total_classes_held' in input_levers:
                # More classes might dilute attendance slightly
                class_effect = -0.02 if input_levers['total_classes_held'] > 120 else 0.02
                predicted_value = baseline + class_effect
                confidence = 0.65
            
            elif lever_name == 'new_members' and 'retention_rate' in input_levers:
                # High retention might correlate with referrals
                retention_effect = 5 if input_levers['retention_rate'] > 0.80 else 0
                predicted_value = baseline + retention_effect
                confidence = 0.60
            
            else:
                # Default: use historical average with some noise
                predicted_value = baseline
                confidence = 0.65
            
            # Ensure within bounds
            bounds = self.lever_constraints.get(lever_name, (0, float('inf')))
            predicted_value = np.clip(predicted_value, bounds[0], bounds[1])
        
        else:
            # No historical data, use middle of constraint range
            bounds = self.lever_constraints.get(lever_name, (0, 1))
            predicted_value = (bounds[0] + bounds[1]) / 2
            confidence = 0.50
        
        return predicted_value, confidence

    def _calculate_priority(self, lever_name: str, change_pct: float) -> int:
        """Calculate priority for lever change (1=highest, 10=lowest)"""
        # Priority based on typical impact and ease of implementation
        priority_map = {
            'retention_rate': 1,  # High impact, hard to change
            'avg_ticket_price': 2,  # High impact, medium difficulty
            'new_members': 3,  # High impact, medium difficulty
            'upsell_rate': 4,  # Medium impact, easier to change
            'class_attendance_rate': 5,  # Medium impact
            'staff_utilization_rate': 6,  # Medium impact
            'total_classes_held': 7,  # Lower impact, easier to change
            'total_members': 8  # Outcome more than lever
        }
        
        base_priority = priority_map.get(lever_name, 5)
        
        # Adjust based on magnitude of change
        if change_pct > 10:
            base_priority = max(1, base_priority - 1)
        
        return base_priority

    def _create_action_plan(self, lever_changes: List[Dict], expected_impact: float) -> List[Dict]:
        """Create prioritized action plan from lever changes"""
        action_items = []
        
        action_templates = {
            'retention_rate': {
                'action': 'Improve member retention through enhanced engagement programs and personalized services',
                'timeline_weeks': 8
            },
            'avg_ticket_price': {
                'action': 'Implement strategic pricing increase with value-added services',
                'timeline_weeks': 4
            },
            'new_members': {
                'action': 'Launch targeted marketing campaign and referral program',
                'timeline_weeks': 6
            },
            'upsell_rate': {
                'action': 'Train staff on upselling techniques and introduce premium packages',
                'timeline_weeks': 3
            },
            'class_attendance_rate': {
                'action': 'Optimize class schedule and improve class variety based on member feedback',
                'timeline_weeks': 2
            },
            'staff_utilization_rate': {
                'action': 'Optimize staff scheduling and cross-train instructors',
                'timeline_weeks': 2
            },
            'total_classes_held': {
                'action': 'Expand class offerings during peak demand hours',
                'timeline_weeks': 2
            }
        }
        
        for i, change in enumerate(lever_changes[:5]):  # Top 5 changes
            lever_name = change['lever_name']
            if lever_name in action_templates:
                template = action_templates[lever_name]
                action_items.append({
                    'priority': i + 1,
                    'lever': lever_name,
                    'action': template['action'],
                    'expected_impact': float(expected_impact * (0.5 - i * 0.1)),  # Distribute impact
                    'timeline_weeks': template['timeline_weeks']
                })
        
        return action_items

