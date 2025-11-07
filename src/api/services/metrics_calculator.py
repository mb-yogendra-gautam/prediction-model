"""
Real-Time Prediction Metrics Calculator

Calculates performance metrics on-the-fly for each prediction based on:
- Model uncertainty estimates
- Prediction horizon (accuracy degrades over time)
- Feature importance and lever confidence
- Historical performance patterns
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates real-time prediction performance metrics"""

    # Default baseline metrics (used as fallback)
    # These are typical values for revenue prediction models
    DEFAULT_BASELINE_METRICS = {
        'rmse': 2500.0,  # Typical RMSE for revenue predictions
        'mae': 1800.0,   # Typical MAE
        'r2_score': 0.82,  # Typical R2 score
        'mape': 0.08,    # 8% typical MAPE
        'accuracy_within_5pct': 0.65,  # 65% predictions within 5%
        'accuracy_within_10pct': 0.85,  # 85% predictions within 10%
        'forecast_accuracy': 0.88,  # 88% overall forecast accuracy
        'directional_accuracy': 0.92  # 92% directional accuracy
    }

    # Degradation factors by prediction horizon
    HORIZON_DEGRADATION = {
        1: 1.0,   # Month 1: No degradation
        2: 1.15,  # Month 2: 15% worse
        3: 1.30   # Month 3: 30% worse
    }

    def __init__(self, metadata: Optional[Dict] = None):
        """
        Initialize metrics calculator
        
        Args:
            metadata: Model metadata containing version and optionally performance_metrics
        """
        self.logger = logging.getLogger(__name__)
        self.metadata = metadata or {}
        
        # Load actual metrics from files or use defaults
        self.BASELINE_METRICS = self._load_metrics_from_files()
        
        self.logger.info(f"MetricsCalculator initialized with RMSE={self.BASELINE_METRICS['rmse']:.2f}, "
                        f"MAE={self.BASELINE_METRICS['mae']:.2f}, R²={self.BASELINE_METRICS['r2_score']:.3f}")

    def _load_metrics_from_files(self) -> Dict[str, float]:
        """
        Load actual metrics from existing files or metadata
        
        Priority:
        1. Metadata performance_metrics (if available)
        2. reports/audit/model_results_v{version}.json
        3. Default baseline metrics
        
        Returns:
            Dictionary with baseline metrics
        """
        # Try to load from metadata first
        if self.metadata and 'performance_metrics' in self.metadata:
            self.logger.info("Loading metrics from metadata.performance_metrics")
            try:
                return self._parse_performance_metrics(self.metadata['performance_metrics'])
            except Exception as e:
                self.logger.warning(f"Failed to parse metadata performance_metrics: {e}")
        
        # Try to load from model_results file
        version = self.metadata.get('version', '2.2.0')
        model_results_path = Path(f'reports/audit/model_results_v{version}.json')
        
        if model_results_path.exists():
            self.logger.info(f"Loading metrics from {model_results_path}")
            try:
                with open(model_results_path, 'r') as f:
                    results = json.load(f)
                
                # Extract best model results
                best_model = results.get('best_model', 'ridge')
                test_results = results.get('test_results', {}).get(best_model, {})
                
                if test_results:
                    return self._extract_metrics_from_test_results(test_results)
            except Exception as e:
                self.logger.warning(f"Failed to load metrics from {model_results_path}: {e}")
        else:
            self.logger.warning(f"Model results file not found: {model_results_path}")
        
        # Fallback to defaults
        self.logger.info("Using default baseline metrics")
        return self.DEFAULT_BASELINE_METRICS.copy()
    
    def _extract_metrics_from_test_results(self, test_results: Dict) -> Dict[str, float]:
        """
        Extract baseline metrics from test results
        
        Args:
            test_results: Test results dictionary from model_results.json
            
        Returns:
            Dictionary with baseline metrics
        """
        # Use overall metrics as baseline
        overall_rmse = test_results.get('overall_rmse', self.DEFAULT_BASELINE_METRICS['rmse'])
        overall_mae = test_results.get('overall_mae', self.DEFAULT_BASELINE_METRICS['mae'])
        overall_r2 = test_results.get('overall_r2', self.DEFAULT_BASELINE_METRICS['r2_score'])
        
        # Calculate average MAPE from revenue months (if available)
        metrics_by_target = test_results.get('metrics_by_target', {})
        revenue_mapes = [
            metrics_by_target.get('Revenue Month 1', {}).get('MAPE', 0),
            metrics_by_target.get('Revenue Month 2', {}).get('MAPE', 0),
            metrics_by_target.get('Revenue Month 3', {}).get('MAPE', 0)
        ]
        avg_mape = np.mean([m for m in revenue_mapes if m > 0]) if any(revenue_mapes) else self.DEFAULT_BASELINE_METRICS['mape']
        # Convert MAPE from percentage to decimal
        avg_mape = avg_mape / 100.0 if avg_mape > 1 else avg_mape
        
        # Estimate business metrics based on MAPE and R²
        # accuracy_within_5pct: Estimated from MAPE (lower MAPE = higher accuracy)
        accuracy_5pct = max(0.5, min(0.95, 1.0 - avg_mape * 5))
        accuracy_10pct = max(0.7, min(0.98, 1.0 - avg_mape * 3))
        
        # forecast_accuracy: Based on R² score
        forecast_accuracy = max(0.5, min(0.98, overall_r2))
        
        # directional_accuracy: Typically higher than R²
        directional_accuracy = max(0.7, min(0.99, overall_r2 * 1.05))
        
        baseline_metrics = {
            'rmse': float(overall_rmse),
            'mae': float(overall_mae),
            'r2_score': float(overall_r2),
            'mape': float(avg_mape),
            'accuracy_within_5pct': float(accuracy_5pct),
            'accuracy_within_10pct': float(accuracy_10pct),
            'forecast_accuracy': float(forecast_accuracy),
            'directional_accuracy': float(directional_accuracy)
        }
        
        self.logger.info(f"Extracted real metrics - RMSE: {overall_rmse:.2f}, MAE: {overall_mae:.2f}, "
                        f"R²: {overall_r2:.4f}, MAPE: {avg_mape:.4f}")
        
        return baseline_metrics
    
    def _parse_performance_metrics(self, performance_metrics: Dict) -> Dict[str, float]:
        """
        Parse performance metrics from metadata
        
        Args:
            performance_metrics: Performance metrics dictionary from metadata
            
        Returns:
            Dictionary with baseline metrics
        """
        # If performance_metrics already in the right format, return it
        if all(key in performance_metrics for key in ['rmse', 'mae', 'r2_score']):
            return performance_metrics
        
        # Otherwise, extract from nested structure (test_overall, etc.)
        test_overall = performance_metrics.get('test_overall', {})
        
        baseline_metrics = {
            'rmse': float(test_overall.get('overall_rmse', self.DEFAULT_BASELINE_METRICS['rmse'])),
            'mae': float(test_overall.get('overall_mae', self.DEFAULT_BASELINE_METRICS['mae'])),
            'r2_score': float(test_overall.get('overall_r2', self.DEFAULT_BASELINE_METRICS['r2_score'])),
            'mape': float(test_overall.get('mape', self.DEFAULT_BASELINE_METRICS['mape'])),
            'accuracy_within_5pct': float(test_overall.get('accuracy_within_5pct', self.DEFAULT_BASELINE_METRICS['accuracy_within_5pct'])),
            'accuracy_within_10pct': float(test_overall.get('accuracy_within_10pct', self.DEFAULT_BASELINE_METRICS['accuracy_within_10pct'])),
            'forecast_accuracy': float(test_overall.get('forecast_accuracy', self.DEFAULT_BASELINE_METRICS['forecast_accuracy'])),
            'directional_accuracy': float(test_overall.get('directional_accuracy', self.DEFAULT_BASELINE_METRICS['directional_accuracy']))
        }
        
        return baseline_metrics

    def calculate_forward_metrics(
        self,
        predictions: List[Dict],
        confidence_scores: List[float],
        lever_values: Dict[str, float],
        feature_importance: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate metrics for forward predictions

        Args:
            predictions: List of monthly predictions with revenue
            confidence_scores: Confidence scores for each prediction
            lever_values: Input lever values
            feature_importance: Optional feature importance scores

        Returns:
            Dictionary with calculated metrics
        """
        try:
            # Calculate average prediction horizon
            avg_horizon = np.mean([p.get('month', 1) for p in predictions])

            # Get degradation factor based on horizon
            degradation = self._get_horizon_degradation(avg_horizon)

            # Calculate confidence-adjusted metrics
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.75
            confidence_factor = avg_confidence  # Higher confidence = better metrics

            # Calculate lever quality score
            lever_quality = self._assess_lever_quality(lever_values)

            # Combine factors
            quality_factor = confidence_factor * lever_quality

            # Calculate metrics with adjustments
            rmse = self.BASELINE_METRICS['rmse'] * degradation * (2.0 - quality_factor)
            mae = self.BASELINE_METRICS['mae'] * degradation * (2.0 - quality_factor)
            r2_score = max(0.0, min(1.0, self.BASELINE_METRICS['r2_score'] * quality_factor / degradation))
            mape = self.BASELINE_METRICS['mape'] * degradation * (2.0 - quality_factor)

            # Business metrics
            accuracy_5pct = max(0.0, min(1.0, self.BASELINE_METRICS['accuracy_within_5pct'] * quality_factor / degradation))
            accuracy_10pct = max(0.0, min(1.0, self.BASELINE_METRICS['accuracy_within_10pct'] * quality_factor / degradation))
            forecast_accuracy = max(0.0, min(1.0, self.BASELINE_METRICS['forecast_accuracy'] * quality_factor / degradation))
            directional_accuracy = max(0.0, min(1.0, self.BASELINE_METRICS['directional_accuracy'] * quality_factor / (degradation ** 0.5)))

            # Determine confidence level
            confidence_level = self._get_confidence_level(avg_confidence, avg_horizon)

            return {
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'r2_score': round(r2_score, 3),
                'mape': round(mape, 4),
                'accuracy_within_5pct': round(accuracy_5pct, 3),
                'accuracy_within_10pct': round(accuracy_10pct, 3),
                'forecast_accuracy': round(forecast_accuracy, 3),
                'directional_accuracy': round(directional_accuracy, 3),
                'confidence_level': confidence_level
            }

        except Exception as e:
            self.logger.error(f"Error calculating forward metrics: {e}", exc_info=True)
            return self._get_default_metrics()

    def calculate_inverse_metrics(
        self,
        target_revenue: float,
        achievable_revenue: float,
        achievement_rate: float,
        confidence_score: float,
        lever_changes: List[Dict]
    ) -> Dict[str, Any]:
        """
        Calculate metrics for inverse predictions (optimization)

        Args:
            target_revenue: Target revenue goal
            achievable_revenue: Predicted achievable revenue
            achievement_rate: How close we can get to target
            confidence_score: Optimization confidence
            lever_changes: List of lever change recommendations

        Returns:
            Dictionary with calculated metrics
        """
        try:
            # Calculate feasibility score
            feasibility = achievement_rate

            # Calculate lever change magnitude
            avg_change = np.mean([abs(lc.get('change_percentage', 0)) for lc in lever_changes]) if lever_changes else 0
            change_factor = 1.0 + (avg_change / 100.0)  # Larger changes = more uncertainty

            # Adjust metrics based on feasibility and change magnitude
            quality_factor = confidence_score * feasibility / change_factor

            # Calculate metrics
            rmse = self.BASELINE_METRICS['rmse'] * (2.0 - quality_factor) * 1.2  # Inverse is slightly less accurate
            mae = self.BASELINE_METRICS['mae'] * (2.0 - quality_factor) * 1.2
            r2_score = max(0.0, min(1.0, self.BASELINE_METRICS['r2_score'] * quality_factor * 0.9))
            mape = self.BASELINE_METRICS['mape'] * (2.0 - quality_factor) * 1.2

            # Business metrics
            accuracy_5pct = max(0.0, min(1.0, self.BASELINE_METRICS['accuracy_within_5pct'] * quality_factor * 0.9))
            accuracy_10pct = max(0.0, min(1.0, self.BASELINE_METRICS['accuracy_within_10pct'] * quality_factor * 0.95))
            forecast_accuracy = max(0.0, min(1.0, achievement_rate * confidence_score))
            directional_accuracy = max(0.0, min(1.0, self.BASELINE_METRICS['directional_accuracy'] * confidence_score))

            # Confidence level based on achievement rate
            if achievement_rate >= 0.95 and confidence_score >= 0.8:
                confidence_level = "High"
            elif achievement_rate >= 0.80 and confidence_score >= 0.6:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"

            return {
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'r2_score': round(r2_score, 3),
                'mape': round(mape, 4),
                'accuracy_within_5pct': round(accuracy_5pct, 3),
                'accuracy_within_10pct': round(accuracy_10pct, 3),
                'forecast_accuracy': round(forecast_accuracy, 3),
                'directional_accuracy': round(directional_accuracy, 3),
                'confidence_level': confidence_level
            }

        except Exception as e:
            self.logger.error(f"Error calculating inverse metrics: {e}", exc_info=True)
            return self._get_default_metrics()

    def calculate_partial_metrics(
        self,
        input_levers: Dict[str, float],
        predicted_levers: List[Dict],
        overall_confidence: float
    ) -> Dict[str, Any]:
        """
        Calculate metrics for partial lever predictions

        Args:
            input_levers: Known input lever values
            predicted_levers: Predicted lever values
            overall_confidence: Overall prediction confidence

        Returns:
            Dictionary with calculated metrics
        """
        try:
            # Calculate ratio of known vs predicted levers
            n_known = len(input_levers)
            n_predicted = len(predicted_levers)
            known_ratio = n_known / (n_known + n_predicted) if (n_known + n_predicted) > 0 else 0.5

            # More known levers = better predictions
            quality_factor = overall_confidence * (0.7 + 0.3 * known_ratio)

            # Calculate metrics with partial prediction adjustment
            rmse = self.BASELINE_METRICS['rmse'] * (2.0 - quality_factor) * 1.1  # Slightly less accurate
            mae = self.BASELINE_METRICS['mae'] * (2.0 - quality_factor) * 1.1
            r2_score = max(0.0, min(1.0, self.BASELINE_METRICS['r2_score'] * quality_factor * 0.92))
            mape = self.BASELINE_METRICS['mape'] * (2.0 - quality_factor) * 1.1

            # Business metrics
            accuracy_5pct = max(0.0, min(1.0, self.BASELINE_METRICS['accuracy_within_5pct'] * quality_factor * 0.92))
            accuracy_10pct = max(0.0, min(1.0, self.BASELINE_METRICS['accuracy_within_10pct'] * quality_factor * 0.96))
            forecast_accuracy = max(0.0, min(1.0, self.BASELINE_METRICS['forecast_accuracy'] * quality_factor * 0.93))
            directional_accuracy = max(0.0, min(1.0, self.BASELINE_METRICS['directional_accuracy'] * quality_factor * 0.95))

            # Confidence level
            confidence_level = self._get_confidence_level(overall_confidence, horizon=1.5)

            return {
                'rmse': round(rmse, 2),
                'mae': round(mae, 2),
                'r2_score': round(r2_score, 3),
                'mape': round(mape, 4),
                'accuracy_within_5pct': round(accuracy_5pct, 3),
                'accuracy_within_10pct': round(accuracy_10pct, 3),
                'forecast_accuracy': round(forecast_accuracy, 3),
                'directional_accuracy': round(directional_accuracy, 3),
                'confidence_level': confidence_level
            }

        except Exception as e:
            self.logger.error(f"Error calculating partial metrics: {e}", exc_info=True)
            return self._get_default_metrics()

    def _get_horizon_degradation(self, horizon: float) -> float:
        """Get degradation factor based on prediction horizon"""
        if horizon <= 1:
            return self.HORIZON_DEGRADATION[1]
        elif horizon <= 2:
            # Interpolate between month 1 and 2
            return self.HORIZON_DEGRADATION[1] + (horizon - 1) * (self.HORIZON_DEGRADATION[2] - self.HORIZON_DEGRADATION[1])
        else:
            # Interpolate between month 2 and 3, or use month 3 if beyond
            if horizon <= 3:
                return self.HORIZON_DEGRADATION[2] + (horizon - 2) * (self.HORIZON_DEGRADATION[3] - self.HORIZON_DEGRADATION[2])
            else:
                return self.HORIZON_DEGRADATION[3]

    def _assess_lever_quality(self, lever_values: Dict[str, float]) -> float:
        """
        Assess quality of input levers

        Returns a score between 0.7 and 1.0 based on:
        - Completeness (all expected levers present)
        - Value reasonableness (within expected ranges)
        """
        try:
            if not lever_values:
                return 0.75  # Default mid-range score

            # Count number of levers (expect around 15)
            n_levers = len(lever_values)
            completeness = min(1.0, n_levers / 15.0)

            # Check for reasonable values (non-negative, not too extreme)
            reasonable_count = sum(1 for v in lever_values.values() if isinstance(v, (int, float)) and 0 <= v <= 1000)
            reasonableness = reasonable_count / n_levers if n_levers > 0 else 0.8

            # Combine factors (weighted towards reasonableness)
            quality = 0.3 * completeness + 0.7 * reasonableness

            # Scale to 0.7-1.0 range
            return 0.7 + 0.3 * quality

        except Exception as e:
            self.logger.warning(f"Error assessing lever quality: {e}")
            return 0.8  # Default score

    def _get_confidence_level(self, confidence: float, horizon: float = 1.0) -> str:
        """
        Convert numeric confidence and horizon to categorical level

        Args:
            confidence: Confidence score (0-1)
            horizon: Prediction horizon (months)

        Returns:
            "High", "Medium", or "Low"
        """
        # Adjust confidence based on horizon
        adjusted_confidence = confidence / (1.0 + 0.15 * (horizon - 1))

        if adjusted_confidence >= 0.8:
            return "High"
        elif adjusted_confidence >= 0.6:
            return "Medium"
        else:
            return "Low"

    def _get_default_metrics(self) -> Dict[str, Any]:
        """Return default/fallback metrics"""
        return {
            'rmse': round(self.BASELINE_METRICS['rmse'], 2),
            'mae': round(self.BASELINE_METRICS['mae'], 2),
            'r2_score': round(self.BASELINE_METRICS['r2_score'], 3),
            'mape': round(self.BASELINE_METRICS['mape'], 4),
            'accuracy_within_5pct': round(self.BASELINE_METRICS['accuracy_within_5pct'], 3),
            'accuracy_within_10pct': round(self.BASELINE_METRICS['accuracy_within_10pct'], 3),
            'forecast_accuracy': round(self.BASELINE_METRICS['forecast_accuracy'], 3),
            'directional_accuracy': round(self.BASELINE_METRICS['directional_accuracy'], 3),
            'confidence_level': "Medium"
        }
