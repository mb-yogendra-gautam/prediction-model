"""
Prediction Service for Forward, Inverse, and Partial Predictions
"""

import numpy as np
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
import logging

from .optimizer import LeverOptimizer, ScenarioComparator
from .metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)


class PredictionService:
    """Core service for making predictions"""

    def __init__(
        self,
        model,
        scaler,
        feature_service,
        historical_data_service,
        metadata: Dict,
        explainability_service=None,
        counterfactual_service=None,
        ai_insights_service=None,
        product_service_analyzer=None
    ):
        """
        Initialize prediction service

        Args:
            model: Trained prediction model
            scaler: Feature scaler
            feature_service: Feature engineering service
            historical_data_service: Historical data service
            metadata: Model metadata
            explainability_service: Optional explainability service for SHAP analysis
            counterfactual_service: Optional counterfactual service for what-if analysis
            ai_insights_service: Optional AI insights service for generating business insights with LangChain
            product_service_analyzer: Optional product/service analyzer for correlation-based recommendations
        """
        self.model = model
        self.scaler = scaler
        self.feature_service = feature_service
        self.historical_data_service = historical_data_service
        self.metadata = metadata
        self.version = metadata.get('version', 'unknown')
        self.explainability_service = explainability_service
        self.counterfactual_service = counterfactual_service
        self.ai_insights_service = ai_insights_service
        self.product_service_analyzer = product_service_analyzer
        
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
        
        # Initialize advanced optimizer
        self.optimizer = LeverOptimizer(
            model=model,
            scaler=scaler,
            feature_service=feature_service,
            lever_constraints=self.lever_constraints
        )
        
        # Initialize scenario comparator
        self.scenario_comparator = ScenarioComparator(self.optimizer)

        # Initialize metrics calculator with metadata
        self.metrics_calculator = MetricsCalculator(metadata=metadata)

        logger.info(f"Prediction service initialized with model version {self.version}")

    def _get_mean_revenue(self, studio_id: str) -> Optional[float]:
        """
        Get mean revenue from historical data for a studio
        
        Args:
            studio_id: Studio identifier
            
        Returns:
            Mean revenue value or None if not available
        """
        try:
            history = self.historical_data_service.get_studio_history(studio_id)
            if history is not None and len(history) > 0 and 'total_revenue' in history.columns:
                mean_rev = float(history['total_revenue'].mean())
                logger.debug(f"Mean revenue for studio {studio_id}: ${mean_rev:.2f}")
                return mean_rev
        except Exception as e:
            logger.warning(f"Could not calculate mean revenue for studio {studio_id}: {e}")
        return None

    def predict_forward(self, request_data: Dict, include_ai_insights: bool = False) -> Dict:
        """
        Forward prediction: levers → revenue

        Args:
            request_data: Dictionary with studio_id, levers, projection_months
            include_ai_insights: Whether to generate AI insights using LangChain (default: False)

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

        # Generate explanations if service is available
        explanation = None
        if self.explainability_service:
            try:
                explanation = self.explainability_service.explain_prediction(
                    features_scaled=features_scaled,
                    levers=levers
                )
                logger.debug("Generated SHAP explanations")
            except Exception as e:
                logger.warning(f"Failed to generate explanations: {e}")

        # Generate counterfactual quick wins if service is available
        quick_wins = None
        if self.counterfactual_service:
            try:
                quick_wins = self.counterfactual_service.generate_quick_wins(
                    studio_id=studio_id,
                    base_levers=levers,
                    max_change_pct=10
                )
                logger.debug(f"Generated {len(quick_wins)} quick win recommendations")
            except Exception as e:
                logger.warning(f"Failed to generate quick wins: {e}")

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

        # Add explanation if available
        if explanation:
            response['explanation'] = explanation

        # Add quick wins if available
        if quick_wins:
            response['quick_wins'] = quick_wins

        # Generate AI insights if requested
        # Get product recommendations if analyzer available
        product_recommendations = None
        if self.product_service_analyzer:
            try:
                product_recommendations = self.product_service_analyzer.get_product_recommendations(
                    current_levers=levers,
                    target_metric='total_revenue'
                )
                logger.debug(f"Generated product recommendations: {len(product_recommendations.get('promote', []))} to promote, {len(product_recommendations.get('demote', []))} to demote")
            except Exception as e:
                logger.warning(f"Error getting product recommendations: {str(e)}")
        
        if include_ai_insights and self.ai_insights_service:
            try:
                ai_insights = self.ai_insights_service.generate_forward_insights(
                    studio_id=studio_id,
                    total_revenue=total_revenue,
                    avg_confidence=avg_confidence,
                    predictions=monthly_predictions,
                    explanation=explanation,
                    quick_wins=quick_wins,
                    product_recommendations=product_recommendations
                )
                if ai_insights:
                    response['ai_insights'] = ai_insights
                    logger.debug("Generated AI insights for forward prediction")
            except Exception as e:
                logger.warning(f"Failed to generate AI insights: {e}")

        # Calculate prediction metrics
        try:
            confidence_scores = [p['confidence_score'] for p in monthly_predictions]
            mean_revenue = self._get_mean_revenue(studio_id)
            metrics = self.metrics_calculator.calculate_forward_metrics(
                predictions=monthly_predictions,
                confidence_scores=confidence_scores,
                lever_values=levers,
                feature_importance=explanation.get('feature_importance') if explanation else None,
                mean_revenue=mean_revenue
            )
            response['prediction_metrics'] = metrics
            logger.debug("Calculated prediction metrics")
        except Exception as e:
            logger.warning(f"Failed to calculate prediction metrics: {e}")

        logger.info(f"Forward prediction complete: ${total_revenue:.2f} over {projection_months} months")
        return response

    def predict_inverse(self, request_data: Dict, include_ai_insights: bool = False) -> Dict:
        """
        Enhanced inverse prediction: target revenue → optimal levers
        Uses advanced optimization with sensitivity analysis and feasibility assessment

        Args:
            request_data: Dictionary with:
                - studio_id: Studio identifier
                - target_revenue: Target revenue to achieve
                - current_state: Current lever values
                - constraints: Optional optimization constraints
                - target_months: Target time horizon (default: 3)
                - method: Optimization method ('auto', 'lbfgs', 'slsqp', 'de', 'ensemble')
                - objectives: List of objectives ['revenue', 'retention', 'growth']
            include_ai_insights: Whether to generate AI insights using LangChain (default: False)

        Returns:
            Dictionary with optimized lever values, action plan, sensitivity, and feasibility
        """
        logger.info(f"Enhanced inverse prediction for studio {request_data['studio_id']}, target: ${request_data['target_revenue']:.2f}")
        
        studio_id = request_data['studio_id']
        target_revenue = request_data['target_revenue']
        current_state = request_data['current_state']
        constraints = request_data.get('constraints', {})
        target_months = request_data.get('target_months', 3)
        method = request_data.get('method', 'auto')
        objectives = request_data.get('objectives', ['revenue'])
        
        # Get historical data
        historical_data = self.historical_data_service.get_studio_history(studio_id, n_months=12)
        
        # Run advanced optimization
        optimization_result = self.optimizer.optimize(
            target_revenue=target_revenue,
            current_state=current_state,
            historical_data=historical_data,
            constraints=constraints,
            target_months=target_months,
            method=method,
            objectives=objectives
        )
        
        # Extract results
        optimized_levers = optimization_result['optimized_levers']
        achievable_revenue = optimization_result['achieved_revenue']
        sensitivity = optimization_result['sensitivity_analysis']
        feasibility = optimization_result['feasibility_assessment']
        uncertainty = optimization_result['uncertainty']

        # Generate explanation for optimized levers
        optimized_explanation = None
        if self.explainability_service:
            try:
                # Engineer features for optimized levers
                opt_features = self.feature_service.engineer_features_from_levers(optimized_levers, historical_data)
                opt_features_scaled = self.scaler.transform(opt_features)

                optimized_explanation = self.explainability_service.explain_prediction(
                    features_scaled=opt_features_scaled,
                    levers=optimized_levers
                )
                logger.debug("Generated explanations for optimized levers")
            except Exception as e:
                logger.warning(f"Failed to explain optimized levers: {e}")

        # Calculate lever changes with enhanced information
        lever_changes = []
        for lever_name, optimized_value in optimized_levers.items():
            current_value = current_state[lever_name]
            change_abs = optimized_value - current_value
            change_pct = (change_abs / current_value * 100) if current_value != 0 else 0
            
            if abs(change_abs) > 0.001:  # Include all non-zero changes
                priority = self._calculate_priority(lever_name, abs(change_pct))
                
                # Add sensitivity and feasibility info
                lever_sensitivity = sensitivity.get(lever_name, 0.0)
                lever_feasibility = feasibility['lever_details'].get(lever_name, {})
                
                lever_changes.append({
                    'lever_name': lever_name,
                    'current_value': float(current_value),
                    'recommended_value': float(optimized_value),
                    'change_absolute': float(change_abs),
                    'change_percentage': float(change_pct),
                    'priority': priority,
                    'sensitivity': float(lever_sensitivity),
                    'feasibility': lever_feasibility.get('difficulty', 'unknown'),
                    'feasibility_score': lever_feasibility.get('score', 0.5)
                })
        
        # Sort by priority
        lever_changes.sort(key=lambda x: x['priority'])
        
        # Create enhanced action plan
        action_plan = self._create_action_plan_enhanced(
            lever_changes=lever_changes,
            achievable_revenue=achievable_revenue,
            current_revenue=self._estimate_current_revenue(current_state, historical_data, target_months),
            feasibility=feasibility
        )
        
        # Build comprehensive response
        response = {
            'optimization_id': str(uuid.uuid4()),
            'studio_id': studio_id,
            'target_revenue': float(target_revenue),
            'achievable_revenue': float(achievable_revenue),
            'achievement_rate': float(optimization_result['achievement_rate']),
            'recommended_levers': optimized_levers,
            'lever_changes': lever_changes,
            'action_plan': action_plan,
            'confidence_score': float(optimization_result['confidence_score']),
            'optimization_details': {
                'method': optimization_result['optimization_method'],
                'success': optimization_result['success'],
                'iterations': optimization_result['iterations'],
                'function_evaluations': optimization_result['function_evaluations']
            },
            'sensitivity_analysis': {
                lever: float(score) for lever, score in sensitivity.items()
            },
            'feasibility_assessment': {
                'overall_score': float(feasibility['overall_score']),
                'overall_difficulty': feasibility['overall_difficulty'],
                'implementation_timeline_weeks': self._estimate_implementation_timeline(feasibility)
            },
            'uncertainty': {
                'predicted_revenue_range': uncertainty['confidence_interval_95'],
                'standard_deviation': float(uncertainty['std_revenue']),
                'confidence_level': '95%'
            },
            'model_version': self.version,
            'timestamp': datetime.now().isoformat()
        }

        # Add explanation if available
        if optimized_explanation:
            response['explanation'] = {
                'description': 'SHAP-based explanation of why these optimized levers achieve the target revenue',
                'details': optimized_explanation
            }

        # Generate AI insights if requested
        if include_ai_insights and self.ai_insights_service:
            try:
                ai_insights = self.ai_insights_service.generate_inverse_insights(
                    studio_id=studio_id,
                    target_revenue=target_revenue,
                    achievable_revenue=achievable_revenue,
                    achievement_rate=response['achievement_rate'],
                    confidence_score=response['confidence_score'],
                    lever_changes=lever_changes,
                    action_plan=action_plan,
                    sensitivity=sensitivity,
                    feasibility=feasibility
                )
                if ai_insights:
                    response['ai_insights'] = ai_insights
                    logger.debug("Generated AI insights for inverse prediction")
            except Exception as e:
                logger.warning(f"Failed to generate AI insights: {e}")

        # Calculate prediction metrics
        try:
            mean_revenue = self._get_mean_revenue(studio_id)
            metrics = self.metrics_calculator.calculate_inverse_metrics(
                target_revenue=target_revenue,
                achievable_revenue=achievable_revenue,
                achievement_rate=response['achievement_rate'],
                confidence_score=response['confidence_score'],
                lever_changes=lever_changes,
                mean_revenue=mean_revenue
            )
            response['prediction_metrics'] = metrics
            logger.debug("Calculated prediction metrics")
        except Exception as e:
            logger.warning(f"Failed to calculate prediction metrics: {e}")

        logger.info(
            f"Enhanced inverse prediction complete: ${achievable_revenue:.2f} achievable "
            f"({response['achievement_rate']*100:.1f}% of target), "
            f"confidence: {response['confidence_score']:.2f}"
        )

        return response
    
    def _estimate_current_revenue(self, current_state: Dict, historical_data, months: int) -> float:
        """Estimate current revenue trajectory"""
        try:
            features = self.feature_service.engineer_features_from_levers(current_state, historical_data)
            features_scaled = self.scaler.transform(features)
            predictions = self.model.predict(features_scaled)[0]
            return float(sum(predictions[:min(months, 3)]))
        except Exception as e:
            logger.warning(f"Could not estimate current revenue: {e}")
            # Fallback: simple estimation
            return current_state['avg_ticket_price'] * current_state['total_members'] * months
    
    def _estimate_implementation_timeline(self, feasibility: Dict) -> int:
        """Estimate total implementation timeline in weeks"""
        difficulty = feasibility['overall_difficulty']
        timeline_map = {
            'easy': 4,
            'moderate': 8,
            'hard': 12,
            'very_hard': 16
        }
        return timeline_map.get(difficulty, 8)
    
    def _create_action_plan_enhanced(
        self,
        lever_changes: List[Dict],
        achievable_revenue: float,
        current_revenue: float,
        feasibility: Dict
    ) -> List[Dict]:
        """Create enhanced prioritized action plan from lever changes"""
        action_items = []
        
        action_templates = {
            'retention_rate': {
                'action': 'Improve member retention through enhanced engagement programs and personalized services',
                'timeline_weeks': 8,
                'department': 'Member Success',
                'resources_needed': ['Staff training', 'CRM system', 'Engagement programs']
            },
            'avg_ticket_price': {
                'action': 'Implement strategic pricing increase with value-added services',
                'timeline_weeks': 4,
                'department': 'Sales & Marketing',
                'resources_needed': ['Pricing analysis', 'Value proposition development', 'Communication plan']
            },
            'new_members': {
                'action': 'Launch targeted marketing campaign and referral program',
                'timeline_weeks': 6,
                'department': 'Marketing',
                'resources_needed': ['Marketing budget', 'Campaign materials', 'Referral incentives']
            },
            'upsell_rate': {
                'action': 'Train staff on upselling techniques and introduce premium packages',
                'timeline_weeks': 3,
                'department': 'Sales',
                'resources_needed': ['Sales training', 'Premium packages', 'Incentive program']
            },
            'class_attendance_rate': {
                'action': 'Optimize class schedule and improve class variety based on member feedback',
                'timeline_weeks': 2,
                'department': 'Operations',
                'resources_needed': ['Member survey', 'Schedule optimization', 'Instructor training']
            },
            'staff_utilization_rate': {
                'action': 'Optimize staff scheduling and cross-train instructors',
                'timeline_weeks': 2,
                'department': 'HR & Operations',
                'resources_needed': ['Scheduling software', 'Cross-training program']
            },
            'total_classes_held': {
                'action': 'Expand class offerings during peak demand hours',
                'timeline_weeks': 2,
                'department': 'Operations',
                'resources_needed': ['Additional instructors', 'Equipment', 'Studio space']
            },
            'total_members': {
                'action': 'Grow member base through acquisition and retention strategies',
                'timeline_weeks': 12,
                'department': 'Growth',
                'resources_needed': ['Marketing budget', 'Sales team', 'Member experience improvements']
            }
        }
        
        total_revenue_impact = achievable_revenue - current_revenue
        
        for i, change in enumerate(lever_changes[:6]):  # Top 6 changes
            lever_name = change['lever_name']
            if lever_name in action_templates:
                template = action_templates[lever_name]
                
                # Calculate proportional impact based on sensitivity
                total_sensitivity = sum([abs(c['sensitivity']) for c in lever_changes[:6]])
                if total_sensitivity > 0:
                    impact_weight = abs(change['sensitivity']) / total_sensitivity
                else:
                    impact_weight = 1.0 / len(lever_changes[:6])  # Equal distribution if no sensitivity
                expected_impact = total_revenue_impact * impact_weight
                
                # Adjust timeline based on feasibility
                base_timeline = template['timeline_weeks']
                if change['feasibility'] == 'hard':
                    base_timeline = int(base_timeline * 1.5)
                elif change['feasibility'] == 'very_hard':
                    base_timeline = int(base_timeline * 2)
                
                action_items.append({
                    'priority': i + 1,
                    'lever': lever_name,
                    'action': template['action'],
                    'expected_impact': float(expected_impact),
                    'timeline_weeks': base_timeline,
                    'department': template['department'],
                    'feasibility': change['feasibility'],
                    'resources_needed': template['resources_needed'],
                    'sensitivity_score': float(change['sensitivity'])
                })
        
        return action_items

    def predict_partial_levers(self, request_data: Dict, include_ai_insights: bool = False) -> Dict:
        """
        Partial lever prediction: subset of levers → remaining levers

        Args:
            request_data: Dictionary with studio_id, input_levers, output_levers, projection_months
            include_ai_insights: Whether to generate AI insights using LangChain (default: False)

        Returns:
            Dictionary with predicted lever values
        """
        logger.info(f"Partial prediction for studio {request_data['studio_id']}")
        
        studio_id = request_data['studio_id']
        input_levers = request_data['input_levers']
        output_lever_names = request_data['output_levers']
        projection_months = request_data.get('projection_months', 3)
        
        # Get historical data
        historical_data = self.historical_data_service.get_studio_history(studio_id, n_months=12)
        
        # Fill in missing levers once for all predictions
        complete_levers = self._fill_missing_levers(input_levers, historical_data)
        
        # Get model predictions once (used by multiple levers)
        features = self.feature_service.engineer_features_from_levers(complete_levers, historical_data)
        features_scaled = self.scaler.transform(features)
        model_predictions = self.model.predict(features_scaled)[0]
        # model_predictions shape: [revenue_month_1, revenue_month_2, revenue_month_3, member_count_month_3, retention_rate_month_3]
        
        # SEQUENTIAL PREDICTION: Predict month-by-month with evolving state
        predicted_levers = self._predict_levers_sequentially(
            output_lever_names=output_lever_names,
            model_predictions=model_predictions,
            initial_state=complete_levers,
            historical_data=historical_data,
            projection_months=projection_months
        )
        
        # Calculate overall confidence
        overall_confidence = np.mean([pl['confidence_score'] for pl in predicted_levers]) if predicted_levers else 0.0

        # Generate explanation - ALWAYS generate for all partial predictions
        explanation = None
        if self.explainability_service:
            try:
                # Fill missing levers to get complete prediction
                complete_levers = self._fill_missing_levers(input_levers, historical_data)
                features = self.feature_service.engineer_features_from_levers(complete_levers, historical_data)
                features_scaled = self.scaler.transform(features)

                # Generate comprehensive SHAP explanation for all model outputs
                explanation = self.explainability_service.explain_prediction(
                    features_scaled=features_scaled,
                    levers=complete_levers
                )
                
                # Add lever-specific insights showing how input levers influence outputs
                explanation['lever_insights'] = self._generate_lever_insights(
                    explanation=explanation,
                    input_levers=input_levers,
                    output_lever_names=output_lever_names,
                    predicted_levers=predicted_levers
                )
                
                logger.debug("Generated SHAP explanations for partial lever prediction")
            except Exception as e:
                logger.warning(f"Failed to generate explanations: {e}")

        # Generate product recommendations based on output levers being predicted
        product_recommendations = None
        if self.product_service_analyzer:
            try:
                # Analyze recommendations for each output lever as target metric
                all_recommendations = {}
                
                for output_lever in output_lever_names:
                    # Map output lever to appropriate target metric for analyzer
                    target_metric = self._map_output_lever_to_target_metric(output_lever)
                    
                    if target_metric:
                        lever_recs = self.product_service_analyzer.get_product_recommendations(
                            current_levers=complete_levers,
                            target_metric=target_metric
                        )
                        all_recommendations[output_lever] = {
                            'target_metric': target_metric,
                            'promote': lever_recs.get('promote', []),
                            'demote': lever_recs.get('demote', []),
                            'lever_analysis': lever_recs.get('lever_analysis', {})
                        }
                
                # Consolidate recommendations across all output levers
                product_recommendations = self._consolidate_product_recommendations(all_recommendations)
                
                logger.debug(f"Generated product recommendations for partial: {len(product_recommendations.get('promote', []))} to promote, {len(product_recommendations.get('demote', []))} to demote across {len(output_lever_names)} output levers")
            except Exception as e:
                logger.warning(f"Error getting product recommendations: {str(e)}")

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

        # Add explanation if available
        if explanation:
            response['explanation'] = explanation

        # Add product recommendations if available
        if product_recommendations:
            response['product_recommendations'] = product_recommendations

        # Generate AI insights if requested
        if include_ai_insights and self.ai_insights_service:
            try:
                ai_insights = self.ai_insights_service.generate_partial_insights(
                    studio_id=studio_id,
                    input_levers=input_levers,
                    predicted_levers=predicted_levers,
                    confidence=overall_confidence,
                    notes=response.get('notes'),
                    explanation=explanation,  # Pass SHAP explanation for enhanced insights
                    product_recommendations=product_recommendations  # Pass product recommendations
                )
                if ai_insights:
                    response['ai_insights'] = ai_insights
                    logger.debug("Generated AI insights for partial prediction")
            except Exception as e:
                logger.warning(f"Failed to generate AI insights: {e}")

        # Calculate prediction metrics
        try:
            mean_revenue = self._get_mean_revenue(studio_id)
            metrics = self.metrics_calculator.calculate_partial_metrics(
                input_levers=input_levers,
                predicted_levers=predicted_levers,
                overall_confidence=overall_confidence,
                mean_revenue=mean_revenue
            )
            response['prediction_metrics'] = metrics
            logger.debug("Calculated prediction metrics")
        except Exception as e:
            logger.warning(f"Failed to calculate prediction metrics: {e}")

        logger.info(f"Partial prediction complete: {len(predicted_levers)} levers predicted")
        return response

    def _predict_levers_sequentially(
        self,
        output_lever_names: List[str],
        model_predictions: np.ndarray,
        initial_state: Dict,
        historical_data,
        projection_months: int
    ) -> List[Dict]:
        """
        Predict levers sequentially month-by-month with evolving state
        
        Args:
            output_lever_names: List of levers to predict
            model_predictions: Model output [rev_m1, rev_m2, rev_m3, members_m3, retention_m3]
            initial_state: Initial complete lever state
            historical_data: Historical data for training
            projection_months: Number of months to predict
            
        Returns:
            List of predicted levers with monthly predictions
        """
        # Initialize state and storage
        current_state = initial_state.copy()
        lever_predictions_by_month = {lever: [] for lever in output_lever_names}
        
        # Train Ridge models once for non-model levers
        ridge_models = {}
        non_model_levers = [l for l in output_lever_names 
                           if l not in ['total_revenue', 'total_members', 'retention_rate']]
        
        for lever_name in non_model_levers:
            ridge_models[lever_name] = self._train_ridge_model_for_lever(
                lever_name, historical_data
            )
        
        # Predict sequentially for each month
        for month in range(1, projection_months + 1):
            # Get model-predicted values for this month
            month_state = self._get_month_state(
                model_predictions, current_state, month, projection_months
            )
            
            # Update current_state with model predictions
            current_state.update(month_state)
            
            # Predict each output lever using updated state
            for lever_name in output_lever_names:
                if lever_name in ['total_revenue', 'total_members', 'retention_rate']:
                    # Model-predicted levers: use month_state values
                    predicted_value = month_state.get(lever_name)
                    confidence = max(0.5, 0.85 - ((month - 1) * 0.05))
                    
                elif lever_name in ridge_models:
                    # Non-model levers: use Ridge regression with current state
                    predicted_value, confidence = self._predict_non_model_lever_for_month(
                        lever_name,
                        ridge_models[lever_name],
                        current_state,
                        month
                    )
                    # Update current_state with this prediction for next month
                    current_state[lever_name] = predicted_value
                
                else:
                    logger.warning(f"Unknown output lever: {lever_name}")
                    continue
                
                lever_predictions_by_month[lever_name].append({
                    'month': month,
                    'predicted_value': float(predicted_value),
                    'confidence_score': float(confidence)
                })
        
        # Build final predicted_levers list
        predicted_levers = []
        for lever_name in output_lever_names:
            monthly_predictions = lever_predictions_by_month[lever_name]
            
            if not monthly_predictions:
                continue
            
            # Calculate summary values from monthly predictions
            monthly_values = [mp['predicted_value'] for mp in monthly_predictions]
            monthly_confidences = [mp['confidence_score'] for mp in monthly_predictions]
            
            if lever_name == 'total_revenue':
                # For revenue: sum across all months
                predicted_value = sum(monthly_values)
            else:
                # For other levers: average across all months
                predicted_value = np.mean(monthly_values)
            
            confidence = np.mean(monthly_confidences)
            value_range = [float(min(monthly_values)), float(max(monthly_values))]
            
            predicted_levers.append({
                'lever_name': lever_name,
                'predicted_value': float(predicted_value),
                'confidence_score': float(confidence),
                'value_range': value_range,
                'monthly_predictions': monthly_predictions
            })
        
        return predicted_levers

    def _get_month_state(
        self,
        model_predictions: np.ndarray,
        current_state: Dict,
        month: int,
        projection_months: int
    ) -> Dict:
        """
        Get lever values for a specific month from model predictions
        
        Args:
            model_predictions: Model output [rev_m1, rev_m2, rev_m3, members_m3, retention_m3]
            current_state: Current state of all levers
            month: Month number (1-based)
            projection_months: Total projection months
            
        Returns:
            Dict with total_revenue, total_members, retention_rate for this month
        """
        month_state = {}
        initial_members = current_state['total_members']
        initial_retention = current_state['retention_rate']
        predicted_members_m3 = float(model_predictions[3])
        predicted_retention_m3 = float(model_predictions[4])
        
        # Revenue
        if month <= 3:
            month_state['total_revenue'] = float(model_predictions[month - 1])
        else:
            # Extrapolate revenue beyond month 3
            last_revenue = float(model_predictions[2])
            first_revenue = float(model_predictions[0])
            growth_rate = (last_revenue - first_revenue) / (3 * first_revenue)
            # Calculate from last known value
            extrapolated_revenue = last_revenue
            for _ in range(3, month):
                extrapolated_revenue *= (1 + growth_rate)
            month_state['total_revenue'] = extrapolated_revenue
        
        # Total Members
        if month <= 3:
            # Interpolate from current to month 3 prediction
            progress = month / 3
            month_state['total_members'] = initial_members + (predicted_members_m3 - initial_members) * progress
        else:
            # Extrapolate beyond month 3
            growth = predicted_members_m3 - initial_members
            monthly_growth = growth / 3
            month_state['total_members'] = predicted_members_m3 + (month - 3) * monthly_growth
        
        # Retention Rate
        if month <= 2:
            # Use current retention for months 1-2
            month_state['retention_rate'] = initial_retention
        elif month == 3:
            # Use model prediction for month 3
            month_state['retention_rate'] = predicted_retention_m3
        else:
            # Extrapolate beyond month 3
            retention_change = predicted_retention_m3 - initial_retention
            monthly_change = retention_change / 3
            extrapolated_retention = predicted_retention_m3 + (month - 3) * monthly_change
            month_state['retention_rate'] = np.clip(extrapolated_retention, 0.5, 1.0)
        
        return month_state

    def _train_ridge_model_for_lever(
        self,
        lever_name: str,
        historical_data
    ) -> Dict:
        """
        Train a Ridge regression model for a specific lever
        
        Args:
            lever_name: Name of the lever to predict
            historical_data: Historical data for training
            
        Returns:
            Dict with 'model', 'feature_cols', and 'success' flag
        """
        result = {'model': None, 'feature_cols': [], 'success': False}
        
        if historical_data is None or len(historical_data) < 5 or lever_name not in historical_data.columns:
            return result
        
        try:
            # Prepare features: other levers (excluding target)
            available_levers = [col for col in historical_data.columns 
                               if col in self.lever_constraints.keys()]
            feature_cols = [col for col in available_levers if col != lever_name]
            
            if len(feature_cols) < 2:
                return result
            
            # Prepare training data
            X_hist = historical_data[feature_cols].dropna()
            y_hist = historical_data.loc[X_hist.index, lever_name]
            
            if len(X_hist) < 3:
                return result
            
            # Train Ridge regression
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_hist, y_hist)
            
            result['model'] = ridge
            result['feature_cols'] = feature_cols
            result['success'] = True
            
            logger.debug(f"Trained Ridge model for {lever_name} with features: {feature_cols}")
            
        except Exception as e:
            logger.warning(f"Failed to train Ridge model for {lever_name}: {e}")
        
        return result

    def _predict_non_model_lever_for_month(
        self,
        lever_name: str,
        ridge_model_info: Dict,
        current_state: Dict,
        month: int
    ) -> tuple:
        """
        Predict a non-model lever for a single month using trained Ridge model
        
        Args:
            lever_name: Name of the lever
            ridge_model_info: Dict with trained model and feature columns
            current_state: Current state of all levers (evolving each month)
            month: Month number
            
        Returns:
            (predicted_value, confidence_score)
        """
        if not ridge_model_info['success']:
            # Fallback: use current state value or historical average
            if lever_name in current_state:
                return float(current_state[lever_name]), max(0.5, 0.65 - ((month - 1) * 0.05))
            else:
                bounds = self.lever_constraints.get(lever_name, (0, 1))
                return (bounds[0] + bounds[1]) / 2, 0.50
        
        try:
            model = ridge_model_info['model']
            feature_cols = ridge_model_info['feature_cols']
            
            # Create feature vector from current_state
            X_pred = np.array([[current_state.get(col, 0) for col in feature_cols]])
            predicted_value = float(model.predict(X_pred)[0])
            
            # Ensure within bounds
            if lever_name in self.lever_constraints:
                bounds = self.lever_constraints[lever_name]
                predicted_value = np.clip(predicted_value, bounds[0], bounds[1])
            
            confidence = max(0.5, 0.70 - ((month - 1) * 0.05))
            
            return predicted_value, confidence
            
        except Exception as e:
            logger.warning(f"Failed to predict {lever_name} for month {month}: {e}")
            # Fallback
            if lever_name in current_state:
                return float(current_state[lever_name]), 0.50
            else:
                bounds = self.lever_constraints.get(lever_name, (0, 1))
                return (bounds[0] + bounds[1]) / 2, 0.50

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

    def _predict_revenue_monthly(
        self,
        model_predictions: np.ndarray,
        projection_months: int
    ) -> List[Dict]:
        """
        Predict monthly revenue using actual model outputs
        
        Args:
            model_predictions: Model output array [rev_m1, rev_m2, rev_m3, members_m3, retention_m3]
            projection_months: Number of months to predict
            
        Returns:
            List of monthly predictions
        """
        monthly_predictions = []
        
        # Use model predictions for months 1-3
        for i in range(min(projection_months, 3)):
            revenue = float(model_predictions[i])
            confidence = 0.85 - (i * 0.05)
            monthly_predictions.append({
                'month': i + 1,
                'predicted_value': revenue,
                'confidence_score': confidence
            })
        
        # Extrapolate if projection_months > 3
        if projection_months > 3:
            last_revenue = monthly_predictions[-1]['predicted_value']
            first_revenue = monthly_predictions[0]['predicted_value']
            # Calculate growth rate from actual model predictions
            growth_rate = (last_revenue - first_revenue) / (3 * first_revenue)
            
            for i in range(3, projection_months):
                revenue = last_revenue * (1 + growth_rate)
                monthly_predictions.append({
                    'month': i + 1,
                    'predicted_value': revenue,
                    'confidence_score': max(0.5, 0.80 - (i * 0.05))
                })
                last_revenue = revenue
        
        return monthly_predictions

    def _predict_total_members_monthly(
        self,
        model_predictions: np.ndarray,
        complete_levers: Dict,
        projection_months: int
    ) -> List[Dict]:
        """
        Predict monthly total_members using actual model output
        
        Args:
            model_predictions: Model output array [rev_m1, rev_m2, rev_m3, members_m3, retention_m3]
            complete_levers: Complete lever values including current total_members
            projection_months: Number of months to predict
            
        Returns:
            List of monthly predictions
        """
        monthly_predictions = []
        current_members = complete_levers['total_members']
        predicted_members_m3 = float(model_predictions[3])
        
        # Interpolate for months 1-3
        for i in range(min(projection_months, 3)):
            # Linear interpolation from current to month 3 prediction
            progress = (i + 1) / 3
            members = current_members + (predicted_members_m3 - current_members) * progress
            confidence = 0.85 - (i * 0.05)
            monthly_predictions.append({
                'month': i + 1,
                'predicted_value': float(members),
                'confidence_score': confidence
            })
        
        # Extrapolate if projection_months > 3
        if projection_months > 3:
            # Calculate growth trend from months 1-3
            growth = predicted_members_m3 - current_members
            monthly_growth = growth / 3
            
            last_members = monthly_predictions[-1]['predicted_value']
            for i in range(3, projection_months):
                members = last_members + monthly_growth
                monthly_predictions.append({
                    'month': i + 1,
                    'predicted_value': float(members),
                    'confidence_score': max(0.5, 0.80 - (i * 0.05))
                })
                last_members = members
        
        return monthly_predictions

    def _predict_retention_rate_monthly(
        self,
        model_predictions: np.ndarray,
        complete_levers: Dict,
        projection_months: int
    ) -> List[Dict]:
        """
        Predict monthly retention_rate using actual model output
        
        Args:
            model_predictions: Model output array [rev_m1, rev_m2, rev_m3, members_m3, retention_m3]
            complete_levers: Complete lever values including current retention_rate
            projection_months: Number of months to predict
            
        Returns:
            List of monthly predictions
        """
        monthly_predictions = []
        current_retention = complete_levers['retention_rate']
        predicted_retention_m3 = float(model_predictions[4])
        
        # Interpolate for months 1-3
        for i in range(min(projection_months, 3)):
            if i < 2:
                # Use current retention for months 1-2
                retention = current_retention
                confidence = 0.85 - (i * 0.05)
            else:
                # Use model prediction for month 3
                retention = predicted_retention_m3
                confidence = 0.75
            
            monthly_predictions.append({
                'month': i + 1,
                'predicted_value': float(retention),
                'confidence_score': confidence
            })
        
        # Extrapolate if projection_months > 3
        if projection_months > 3:
            # Calculate trend from current to month 3
            retention_change = predicted_retention_m3 - current_retention
            monthly_change = retention_change / 3
            
            last_retention = monthly_predictions[-1]['predicted_value']
            for i in range(3, projection_months):
                retention = last_retention + monthly_change
                # Ensure within bounds
                retention = np.clip(retention, 0.5, 1.0)
                monthly_predictions.append({
                    'month': i + 1,
                    'predicted_value': float(retention),
                    'confidence_score': max(0.5, 0.70 - ((i-3) * 0.05))
                })
                last_retention = retention
        
        return monthly_predictions

    def _predict_non_model_lever_monthly(
        self,
        lever_name: str,
        input_levers: Dict,
        complete_levers: Dict,
        historical_data,
        projection_months: int
    ) -> List[Dict]:
        """
        Predict non-model levers using Ridge regression on historical data
        
        Args:
            lever_name: Name of the lever to predict
            input_levers: User-provided input levers
            complete_levers: Complete lever values
            historical_data: Historical data for training
            projection_months: Number of months to predict
            
        Returns:
            List of monthly predictions
        """
        monthly_predictions = []
        
        # Try to use Ridge regression on historical data
        if historical_data is not None and len(historical_data) > 5 and lever_name in historical_data.columns:
            try:
                # Prepare features from historical data
                feature_cols = [col for col in historical_data.columns 
                               if col in self.lever_constraints.keys() and col != lever_name]
                
                if len(feature_cols) >= 2:
                    X_hist = historical_data[feature_cols].dropna()
                    y_hist = historical_data.loc[X_hist.index, lever_name]
                    
                    if len(X_hist) >= 3:
                        # Train Ridge regression
                        ridge = Ridge(alpha=1.0)
                        ridge.fit(X_hist, y_hist)
                        
                        # For each month, use complete levers to predict
                        for i in range(projection_months):
                            # Create feature vector from complete levers
                            X_pred = np.array([[complete_levers.get(col, X_hist[col].mean()) 
                                              for col in feature_cols]])
                            predicted_value = float(ridge.predict(X_pred)[0])
                            
                            # Ensure within bounds
                            if lever_name in self.lever_constraints:
                                bounds = self.lever_constraints[lever_name]
                                predicted_value = np.clip(predicted_value, bounds[0], bounds[1])
                            
                            confidence = max(0.5, 0.70 - (i * 0.05))
                            monthly_predictions.append({
                                'month': i + 1,
                                'predicted_value': predicted_value,
                                'confidence_score': confidence
                            })
                        
                        return monthly_predictions
            except Exception as e:
                logger.warning(f"Ridge regression failed for {lever_name}: {e}")
        
        # Fallback: Use historical average and apply minimal change
        if historical_data is not None and len(historical_data) > 0 and lever_name in historical_data.columns:
            baseline = float(historical_data[lever_name].mean())
        else:
            # Use middle of constraint range
            bounds = self.lever_constraints.get(lever_name, (0, 1))
            baseline = (bounds[0] + bounds[1]) / 2
        
        # Generate monthly predictions with minimal variation
        for i in range(projection_months):
            predicted_value = baseline
            
            # Ensure within bounds
            if lever_name in self.lever_constraints:
                bounds = self.lever_constraints[lever_name]
                predicted_value = np.clip(predicted_value, bounds[0], bounds[1])
            
            confidence = max(0.5, 0.65 - (i * 0.05))
            monthly_predictions.append({
                'month': i + 1,
                'predicted_value': predicted_value,
                'confidence_score': confidence
            })
        
        return monthly_predictions

    def _predict_lever_evolution(
        self,
        lever_name: str,
        initial_value: float,
        initial_confidence: float,
        projection_months: int,
        historical_data,
        input_levers: Dict
    ) -> List[Dict]:
        """
        Predict how a lever evolves over multiple months
        
        Args:
            lever_name: Name of the lever
            initial_value: Predicted value for month 1
            initial_confidence: Confidence score for month 1
            projection_months: Number of months to predict
            historical_data: Historical data for trend analysis
            input_levers: Input levers that may influence evolution
            
        Returns:
            List of monthly predictions with month, predicted_value, confidence_score
        """
        monthly_predictions = []
        current_value = initial_value
        
        # Calculate historical trend if available
        historical_trend = 0.0
        if historical_data is not None and len(historical_data) > 3 and lever_name in historical_data.columns:
            recent_values = historical_data[lever_name].tail(6)
            if len(recent_values) >= 2:
                # Simple linear trend
                historical_trend = (recent_values.iloc[-1] - recent_values.iloc[0]) / len(recent_values)
        
        # Define evolution patterns based on lever type
        if lever_name == 'retention_rate':
            # Retention tends to be stable or slowly improve/decline
            monthly_change_rate = historical_trend if abs(historical_trend) > 0.001 else 0.002  # +0.2% per month default
        
        elif lever_name == 'avg_ticket_price':
            # Price is strategic, generally stable month-to-month
            monthly_change_rate = 0.0  # No month-to-month change unless strategic
        
        elif lever_name == 'class_attendance_rate':
            # Attendance can vary with seasons, use historical pattern
            monthly_change_rate = historical_trend if abs(historical_trend) > 0.001 else 0.0
        
        elif lever_name == 'new_members':
            # New members can have growth trend
            growth_rate = historical_trend if abs(historical_trend) > 0.1 else 0.5  # +0.5 per month default
            monthly_change_rate = growth_rate
        
        elif lever_name == 'staff_utilization_rate':
            # Staff utilization gradually improves toward optimal
            optimal_utilization = 0.85
            monthly_change_rate = (optimal_utilization - current_value) * 0.1  # 10% of gap per month
        
        elif lever_name == 'upsell_rate':
            # Upsell can gradually improve with training
            monthly_change_rate = 0.005  # +0.5% per month
        
        elif lever_name == 'total_classes_held':
            # Classes tend to be stable or gradually increase
            monthly_change_rate = historical_trend if abs(historical_trend) > 0.1 else 1.0  # +1 class per month
        
        elif lever_name == 'total_members':
            # Total members = retention effect + new members
            retention = input_levers.get('retention_rate', 0.75)
            new_members_monthly = input_levers.get('new_members', 20)
            # Simplified: net change = new members - churned members
            churn_rate = 1 - retention
            churned = current_value * churn_rate
            monthly_change_rate = new_members_monthly - churned
        
        else:
            monthly_change_rate = 0.0
        
        # Generate monthly predictions
        for i in range(projection_months):
            # Update value based on change rate
            if i > 0:
                current_value += monthly_change_rate
            
            # Ensure within bounds
            if lever_name in self.lever_constraints:
                bounds = self.lever_constraints[lever_name]
                current_value = np.clip(current_value, bounds[0], bounds[1])
            
            # Decrease confidence over time
            confidence = max(0.5, initial_confidence - (i * 0.05))
            
            monthly_predictions.append({
                'month': i + 1,
                'predicted_value': float(current_value),
                'confidence_score': float(confidence)
            })
        
        return monthly_predictions

    def _generate_lever_insights(
        self,
        explanation: Dict,
        input_levers: Dict[str, float],
        output_lever_names: List[str],
        predicted_levers: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate lever-specific insights from SHAP explanations
        
        Args:
            explanation: SHAP explanation with all targets
            input_levers: User-provided input levers
            output_lever_names: Levers being predicted
            predicted_levers: Predicted lever results
            
        Returns:
            Dictionary mapping output levers to their driving input levers
        """
        lever_insights = {}
        
        # Get feature-to-lever mapping from explainability service
        feature_lever_map = self.explainability_service.explain_feature_to_lever_mapping()
        
        # For each output lever, identify key drivers
        for output_lever in output_lever_names:
            # Map output lever to relevant model targets
            relevant_targets = self._map_lever_to_targets(output_lever)
            
            # Aggregate SHAP contributions for this output lever
            drivers = []
            for target_name in relevant_targets:
                if target_name in explanation.get('targets', {}):
                    target_exp = explanation['targets'][target_name]
                    
                    # Extract top contributions related to input levers
                    for contrib in target_exp.get('top_5_drivers', []):
                        feature = contrib['feature']
                        
                        # Check if feature relates to an input lever
                        if self._feature_relates_to_input_levers(feature, input_levers):
                            drivers.append({
                                'input_lever': self._extract_lever_from_feature(feature),
                                'feature': feature,
                                'contribution': contrib['shap_value'],
                                'via_target': target_name
                            })
            
            # Remove duplicates and keep top drivers
            seen = set()
            unique_drivers = []
            for driver in drivers:
                key = (driver['input_lever'], driver['feature'])
                if key not in seen:
                    seen.add(key)
                    unique_drivers.append(driver)
            
            # Summarize drivers
            lever_insights[output_lever] = {
                'predicted_value': next(
                    (pl['predicted_value'] for pl in predicted_levers if pl['lever_name'] == output_lever),
                    None
                ),
                'key_drivers': unique_drivers[:5],  # Top 5 unique drivers
                'explanation': self._generate_lever_explanation_text(output_lever, unique_drivers[:5])
            }
        
        return lever_insights

    def _map_lever_to_targets(self, output_lever: str) -> List[str]:
        """
        Map an output lever name to relevant model target names
        
        Args:
            output_lever: Name of the output lever being predicted
            
        Returns:
            List of model target names that are relevant for this lever
        """
        # Map levers to model outputs
        lever_target_map = {
            'total_revenue': ['revenue_month_1', 'revenue_month_2', 'revenue_month_3'],
            'total_members': ['member_count_month_3', 'revenue_month_1', 'revenue_month_2', 'revenue_month_3'],
            'retention_rate': ['retention_rate_month_3', 'member_count_month_3'],
            'avg_ticket_price': ['revenue_month_1', 'revenue_month_2', 'revenue_month_3'],
            'class_attendance_rate': ['revenue_month_1', 'revenue_month_2', 'revenue_month_3'],
            'new_members': ['member_count_month_3', 'revenue_month_1'],
            'staff_utilization_rate': ['revenue_month_1', 'revenue_month_2', 'revenue_month_3'],
            'upsell_rate': ['revenue_month_1', 'revenue_month_2', 'revenue_month_3'],
            'total_classes_held': ['revenue_month_1', 'revenue_month_2', 'revenue_month_3']
        }
        
        # Return relevant targets, default to all revenue targets if not found
        return lever_target_map.get(output_lever, ['revenue_month_1', 'revenue_month_2', 'revenue_month_3'])

    def _feature_relates_to_input_levers(self, feature: str, input_levers: Dict[str, float]) -> bool:
        """
        Check if a feature is derived from or related to input levers
        
        Args:
            feature: Feature name from SHAP analysis
            input_levers: Dictionary of input lever names
            
        Returns:
            True if feature relates to any input lever
        """
        # Direct match: feature name is an input lever
        if feature in input_levers:
            return True
        
        # Check if feature contains any input lever name
        for lever_name in input_levers.keys():
            # Handle interaction features like 'retention_x_ticket'
            if lever_name in feature:
                return True
            
            # Handle abbreviated names in features
            lever_abbrev = lever_name.replace('_', '').lower()
            feature_clean = feature.replace('_', '').lower()
            if lever_abbrev in feature_clean:
                return True
        
        return False

    def _extract_lever_from_feature(self, feature: str) -> str:
        """
        Extract the base lever name from a feature name
        
        Args:
            feature: Feature name (may be lever name or derived feature)
            
        Returns:
            Base lever name
        """
        # List of known lever names
        known_levers = [
            'retention_rate', 'avg_ticket_price', 'class_attendance_rate',
            'new_members', 'staff_utilization_rate', 'upsell_rate',
            'total_classes_held', 'total_members'
        ]
        
        # Check if feature is a known lever
        if feature in known_levers:
            return feature
        
        # Check if feature contains a known lever
        for lever in known_levers:
            if lever in feature:
                return lever
        
        # Check interaction features (e.g., 'retention_x_ticket' -> 'retention_rate')
        for lever in known_levers:
            lever_parts = lever.split('_')
            if any(part in feature for part in lever_parts if len(part) > 3):
                return lever
        
        # Default: return the feature name itself
        return feature

    def _generate_lever_explanation_text(self, output_lever: str, drivers: List[Dict]) -> str:
        """
        Generate human-readable explanation of how input levers influence output lever
        
        Args:
            output_lever: The output lever being explained
            drivers: List of driver dictionaries with input_lever, feature, contribution
            
        Returns:
            Human-readable explanation text
        """
        if not drivers:
            return f"No significant input lever influences detected for {output_lever}"
        
        # Create explanation based on top drivers
        explanations = []
        for driver in drivers[:3]:  # Top 3 drivers
            input_lever = driver.get('input_lever', 'unknown')
            contribution = driver.get('contribution', 0)
            
            # Determine direction and magnitude
            direction = "increases" if contribution > 0 else "decreases"
            magnitude = abs(contribution)
            
            if magnitude > 1000:
                impact_desc = "strongly"
            elif magnitude > 100:
                impact_desc = "moderately"
            else:
                impact_desc = "slightly"
            
            explanations.append(f"{input_lever} {impact_desc} {direction}")
        
        return f"{output_lever} prediction: " + ", ".join(explanations)

    def _map_output_lever_to_target_metric(self, output_lever: str) -> Optional[str]:
        """
        Map an output lever to the appropriate target metric for product analyzer
        
        Args:
            output_lever: Name of the output lever being predicted
            
        Returns:
            Target metric name for product_service_analyzer, or None if not applicable
        """
        # Map output levers to analyzer-compatible metrics
        lever_to_metric_map = {
            'total_revenue': 'total_revenue',
            'retention_rate': 'retention_rate',
            'avg_ticket_price': 'avg_ticket_price',
            'class_attendance_rate': 'class_attendance_rate',
            'new_members': 'new_members',
            'upsell_rate': 'upsell_rate',
            'total_members': 'total_members',
            'staff_utilization_rate': 'total_revenue',  # Default to revenue
            'total_classes_held': 'total_revenue'  # Default to revenue
        }
        
        return lever_to_metric_map.get(output_lever)

    def _consolidate_product_recommendations(self, all_recommendations: Dict) -> Dict:
        """
        Consolidate product recommendations from multiple output levers
        
        Args:
            all_recommendations: Dict mapping output_lever -> recommendations
            
        Returns:
            Consolidated recommendations with promote/demote lists and per-lever details
        """
        if not all_recommendations:
            return {'promote': [], 'demote': [], 'by_output_lever': {}}
        
        # Track products across all recommendations
        product_scores = {}  # product_key -> {'promote_count': int, 'total_correlation': float, 'levers': []}
        
        # Aggregate across all output levers
        for output_lever, recs in all_recommendations.items():
            for product in recs.get('promote', []):
                key = product['product_key']
                if key not in product_scores:
                    product_scores[key] = {
                        'product': product['product'],
                        'product_key': key,
                        'category': product['category'],
                        'promote_count': 0,
                        'total_correlation': 0,
                        'avg_revenue': product.get('avg_revenue', 0),
                        'relevant_for_levers': []
                    }
                product_scores[key]['promote_count'] += 1
                product_scores[key]['total_correlation'] += product['correlation']
                product_scores[key]['relevant_for_levers'].append({
                    'lever': output_lever,
                    'correlation': product['correlation'],
                    'reasoning': product.get('reasoning', '')
                })
        
        # Sort by number of levers it's relevant for, then by average correlation
        consolidated_promote = sorted(
            product_scores.values(),
            key=lambda x: (x['promote_count'], x['total_correlation']),
            reverse=True
        )[:5]  # Top 5 overall
        
        # Add aggregated reasoning
        for product in consolidated_promote:
            levers_str = ', '.join([item['lever'] for item in product['relevant_for_levers']])
            avg_corr = product['total_correlation'] / product['promote_count']
            product['consolidated_reasoning'] = (
                f"Relevant for {product['promote_count']} predicted lever(s): {levers_str}. "
                f"Average correlation: {avg_corr:.2f}"
            )
        
        # Get demote products (products that appear in demote for any lever)
        demote_products = {}
        for output_lever, recs in all_recommendations.items():
            for product in recs.get('demote', []):
                key = product['product_key']
                if key not in demote_products:
                    demote_products[key] = product
        
        consolidated_demote = list(demote_products.values())[:3]
        
        return {
            'promote': consolidated_promote,
            'demote': consolidated_demote,
            'by_output_lever': all_recommendations  # Keep detailed breakdown
        }

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

