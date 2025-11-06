"""
Advanced Optimization Module for Inverse Prediction
Provides multiple optimization strategies, sensitivity analysis, and feasibility assessment
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import minimize, differential_evolution, Bounds
from scipy.stats import norm
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class LeverOptimizer:
    """
    Advanced optimizer for finding optimal lever values to achieve target revenue
    
    Features:
    - Multiple optimization algorithms (L-BFGS-B, SLSQP, Differential Evolution)
    - Multi-objective optimization (revenue, retention, growth)
    - Sensitivity analysis
    - Feasibility assessment
    - Uncertainty quantification
    """
    
    def __init__(
        self,
        model,
        scaler,
        feature_service,
        lever_constraints: Dict[str, Tuple[float, float]]
    ):
        """
        Initialize optimizer
        
        Args:
            model: Trained prediction model
            scaler: Feature scaler
            feature_service: Feature engineering service
            lever_constraints: Dictionary of lever constraints {lever_name: (min, max)}
        """
        self.model = model
        self.scaler = scaler
        self.feature_service = feature_service
        self.lever_constraints = lever_constraints
        
        # Lever names in fixed order
        self.lever_names = [
            'retention_rate',
            'avg_ticket_price',
            'class_attendance_rate',
            'new_members',
            'staff_utilization_rate',
            'upsell_rate',
            'total_classes_held',
            'total_members'
        ]
        
        logger.info("LeverOptimizer initialized")
    
    def optimize(
        self,
        target_revenue: float,
        current_state: Dict[str, float],
        historical_data,
        constraints: Optional[Dict] = None,
        target_months: int = 3,
        method: str = 'auto',
        objectives: List[str] = ['revenue']
    ) -> Dict[str, Any]:
        """
        Main optimization function
        
        Args:
            target_revenue: Target revenue to achieve
            current_state: Current lever values
            historical_data: Historical data for context
            constraints: Custom constraints dictionary
            target_months: Target time horizon (1-12 months)
            method: Optimization method ('auto', 'lbfgs', 'slsqp', 'de', 'ensemble')
            objectives: List of objectives to optimize ['revenue', 'retention', 'growth']
        
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting optimization: target=${target_revenue:.2f}, method={method}, objectives={objectives}")
        
        # Setup bounds
        bounds = self._setup_bounds(current_state, constraints)
        
        # Initial guess
        x0 = self._get_initial_guess(current_state)
        
        # Store context for objective function
        self._historical_data = historical_data
        self._target_months = target_months
        self._target_revenue = target_revenue
        self._objectives = objectives
        self._current_state = current_state
        
        # Choose optimization method
        if method == 'auto':
            # Try multiple methods and pick best
            result = self._optimize_ensemble(x0, bounds)
        elif method == 'lbfgs':
            result = self._optimize_lbfgs(x0, bounds)
        elif method == 'slsqp':
            result = self._optimize_slsqp(x0, bounds)
        elif method == 'de':
            result = self._optimize_differential_evolution(bounds)
        elif method == 'ensemble':
            result = self._optimize_ensemble(x0, bounds)
        else:
            logger.warning(f"Unknown method {method}, using auto")
            result = self._optimize_ensemble(x0, bounds)
        
        # Extract optimized levers
        optimized_levers = self._array_to_levers(result['x'])
        
        # Calculate achieved revenue
        achieved_revenue = self._predict_revenue(result['x'])
        
        # Perform sensitivity analysis
        sensitivity = self._sensitivity_analysis(result['x'])
        
        # Assess feasibility
        feasibility = self._assess_feasibility(current_state, optimized_levers)
        
        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(result['x'])
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            result['success'],
            achieved_revenue,
            target_revenue,
            feasibility,
            uncertainty
        )
        
        logger.info(f"Optimization complete: achieved=${achieved_revenue:.2f}, confidence={confidence:.2f}")
        
        return {
            'optimized_levers': optimized_levers,
            'achieved_revenue': achieved_revenue,
            'target_revenue': target_revenue,
            'achievement_rate': min(achieved_revenue / target_revenue, 1.0) if target_revenue > 0 else 0,
            'optimization_method': result.get('method', method),
            'success': result['success'],
            'iterations': result.get('iterations', 0),
            'function_evaluations': result.get('nfev', 0),
            'confidence_score': confidence,
            'sensitivity_analysis': sensitivity,
            'feasibility_assessment': feasibility,
            'uncertainty': uncertainty
        }
    
    def _objective_single(self, x: np.ndarray) -> float:
        """Single objective: minimize squared error to target revenue"""
        predicted_revenue = self._predict_revenue(x)
        error = (predicted_revenue - self._target_revenue) ** 2
        
        # Add penalty for large changes from current state
        x_current = self._get_initial_guess(self._current_state)
        change_penalty = 0.0001 * np.sum((x - x_current) ** 2)
        
        return error + change_penalty
    
    def _objective_multi(self, x: np.ndarray) -> float:
        """
        Multi-objective function combining revenue, retention, and growth
        Uses weighted sum approach
        """
        levers = self._array_to_levers(x)
        
        # Objective 1: Revenue error (primary)
        predicted_revenue = self._predict_revenue(x)
        revenue_error = ((predicted_revenue - self._target_revenue) / self._target_revenue) ** 2
        
        # Objective 2: Maximize retention (secondary)
        retention_gain = -(levers['retention_rate'] - self._current_state['retention_rate'])
        
        # Objective 3: Maximize member growth (tertiary)
        member_growth = -(levers['total_members'] - self._current_state['total_members'])
        
        # Objective 4: Minimize cost of changes
        x_current = self._get_initial_guess(self._current_state)
        normalized_change = np.sum(np.abs((x - x_current) / x_current)) / len(x)
        
        # Weighted combination
        weights = {
            'revenue': 1.0,
            'retention': 0.3 if 'retention' in self._objectives else 0.0,
            'growth': 0.2 if 'growth' in self._objectives else 0.0,
            'cost': 0.1
        }
        
        objective = (
            weights['revenue'] * revenue_error +
            weights['retention'] * retention_gain +
            weights['growth'] * member_growth * 0.001 +
            weights['cost'] * normalized_change
        )
        
        return objective
    
    def _predict_revenue(self, x: np.ndarray) -> float:
        """Predict revenue from lever values"""
        levers = self._array_to_levers(x)
        features = self.feature_service.engineer_features_from_levers(levers, self._historical_data)
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)[0]
        
        # Sum revenue for target months (max 3)
        revenue = sum(predictions[:min(self._target_months, 3)])
        
        # Extrapolate if needed
        if self._target_months > 3:
            avg_monthly = revenue / 3
            additional_revenue = avg_monthly * (self._target_months - 3)
            revenue += additional_revenue
        
        return float(revenue)
    
    def _optimize_lbfgs(self, x0: np.ndarray, bounds: List[Tuple]) -> Dict:
        """Optimize using L-BFGS-B algorithm (good for smooth functions)"""
        objective = self._objective_multi if len(self._objectives) > 1 else self._objective_single
        
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-4, 'maxfun': 200}
        )
        
        return {
            'x': result.x,
            'success': result.success,
            'method': 'L-BFGS-B',
            'iterations': result.nit,
            'nfev': result.nfev,
            'message': result.message
        }
    
    def _optimize_slsqp(self, x0: np.ndarray, bounds: List[Tuple]) -> Dict:
        """Optimize using SLSQP algorithm (handles constraints well)"""
        objective = self._objective_multi if len(self._objectives) > 1 else self._objective_single
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-4}
        )
        
        return {
            'x': result.x,
            'success': result.success,
            'method': 'SLSQP',
            'iterations': result.nit,
            'nfev': result.nfev,
            'message': result.message
        }
    
    def _optimize_differential_evolution(self, bounds: List[Tuple]) -> Dict:
        """
        Optimize using Differential Evolution (global optimization, good for non-convex)
        """
        objective = self._objective_multi if len(self._objectives) > 1 else self._objective_single
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=30,
            popsize=10,
            tol=1e-3,
            seed=42,
            workers=1,
            atol=1e-2
        )
        
        return {
            'x': result.x,
            'success': result.success,
            'method': 'Differential Evolution',
            'iterations': result.nit,
            'nfev': result.nfev,
            'message': result.message
        }
    
    def _optimize_ensemble(self, x0: np.ndarray, bounds: List[Tuple]) -> Dict:
        """
        Ensemble optimization: try multiple methods and pick best
        """
        results = []
        
        # Try L-BFGS-B
        try:
            r1 = self._optimize_lbfgs(x0, bounds)
            r1['objective_value'] = self._objective_single(r1['x'])
            results.append(r1)
        except Exception as e:
            logger.warning(f"L-BFGS-B failed: {e}")
        
        # Try SLSQP
        try:
            r2 = self._optimize_slsqp(x0, bounds)
            r2['objective_value'] = self._objective_single(r2['x'])
            results.append(r2)
        except Exception as e:
            logger.warning(f"SLSQP failed: {e}")
        
        # Try Differential Evolution (best for global search)
        try:
            r3 = self._optimize_differential_evolution(bounds)
            r3['objective_value'] = self._objective_single(r3['x'])
            results.append(r3)
        except Exception as e:
            logger.warning(f"DE failed: {e}")
        
        if not results:
            raise RuntimeError("All optimization methods failed")
        
        # Pick best result (lowest objective value)
        best_result = min(results, key=lambda r: r['objective_value'])
        best_result['method'] = f"Ensemble (best: {best_result['method']})"
        
        return best_result
    
    def _sensitivity_analysis(self, x_optimal: np.ndarray) -> Dict[str, float]:
        """
        Perform sensitivity analysis: how much does revenue change with each lever?
        
        Returns:
            Dictionary of {lever_name: sensitivity_score}
        """
        sensitivity = {}
        base_revenue = self._predict_revenue(x_optimal)
        
        for i, lever_name in enumerate(self.lever_names):
            # Perturb lever by 1%
            x_perturbed = x_optimal.copy()
            delta = x_optimal[i] * 0.01
            if delta == 0:
                delta = 0.01
            
            x_perturbed[i] += delta
            
            # Check bounds
            bounds = self.lever_constraints[lever_name]
            x_perturbed[i] = np.clip(x_perturbed[i], bounds[0], bounds[1])
            
            # Calculate revenue change
            perturbed_revenue = self._predict_revenue(x_perturbed)
            revenue_change = (perturbed_revenue - base_revenue) / base_revenue * 100
            
            # Sensitivity = % revenue change / % lever change
            sensitivity[lever_name] = float(revenue_change / 1.0)  # 1% lever change
        
        return sensitivity
    
    def _assess_feasibility(
        self,
        current_state: Dict[str, float],
        optimized_levers: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Assess how feasible the recommended changes are
        
        Returns:
            Dictionary with feasibility score and breakdown
        """
        lever_feasibility = {}
        
        # Define feasibility thresholds for each lever
        feasibility_thresholds = {
            'retention_rate': {
                'easy': 0.02,      # <2% change is easy
                'moderate': 0.05,  # 2-5% change is moderate
                'hard': 0.10       # >10% change is very hard
            },
            'avg_ticket_price': {
                'easy': 5.0,       # <$5 change is easy
                'moderate': 20.0,  # $5-20 change is moderate
                'hard': 50.0       # >$50 change is very hard
            },
            'class_attendance_rate': {
                'easy': 0.03,
                'moderate': 0.07,
                'hard': 0.15
            },
            'new_members': {
                'easy': 5,
                'moderate': 15,
                'hard': 30
            },
            'staff_utilization_rate': {
                'easy': 0.05,
                'moderate': 0.10,
                'hard': 0.20
            },
            'upsell_rate': {
                'easy': 0.03,
                'moderate': 0.08,
                'hard': 0.15
            },
            'total_classes_held': {
                'easy': 10,
                'moderate': 30,
                'hard': 60
            },
            'total_members': {
                'easy': 10,
                'moderate': 30,
                'hard': 60
            }
        }
        
        total_score = 0.0
        count = 0
        
        for lever_name in self.lever_names:
            current_value = current_state[lever_name]
            optimized_value = optimized_levers[lever_name]
            change_abs = abs(optimized_value - current_value)
            
            thresholds = feasibility_thresholds.get(lever_name, {
                'easy': float('inf'),
                'moderate': float('inf'),
                'hard': float('inf')
            })
            
            # Assign feasibility score
            if change_abs <= thresholds['easy']:
                score = 1.0
                difficulty = 'easy'
            elif change_abs <= thresholds['moderate']:
                score = 0.7
                difficulty = 'moderate'
            elif change_abs <= thresholds['hard']:
                score = 0.4
                difficulty = 'hard'
            else:
                score = 0.2
                difficulty = 'very_hard'
            
            lever_feasibility[lever_name] = {
                'score': score,
                'difficulty': difficulty,
                'change_abs': float(change_abs)
            }
            
            total_score += score
            count += 1
        
        overall_score = total_score / count if count > 0 else 0.0
        
        # Determine overall difficulty
        if overall_score >= 0.8:
            overall_difficulty = 'easy'
        elif overall_score >= 0.6:
            overall_difficulty = 'moderate'
        elif overall_score >= 0.4:
            overall_difficulty = 'hard'
        else:
            overall_difficulty = 'very_hard'
        
        return {
            'overall_score': overall_score,
            'overall_difficulty': overall_difficulty,
            'lever_details': lever_feasibility
        }
    
    def _calculate_uncertainty(self, x_optimal: np.ndarray) -> Dict[str, float]:
        """
        Estimate prediction uncertainty using bootstrap-style sampling
        
        Returns:
            Dictionary with uncertainty estimates
        """
        # Make multiple predictions with small perturbations
        n_samples = 20
        revenues = []
        
        for _ in range(n_samples):
            # Add small random noise to simulate uncertainty
            noise = np.random.normal(0, 0.01, size=x_optimal.shape)
            x_perturbed = x_optimal + noise
            
            # Ensure bounds
            for i, lever_name in enumerate(self.lever_names):
                bounds = self.lever_constraints[lever_name]
                x_perturbed[i] = np.clip(x_perturbed[i], bounds[0], bounds[1])
            
            revenue = self._predict_revenue(x_perturbed)
            revenues.append(revenue)
        
        revenues = np.array(revenues)
        
        return {
            'mean_revenue': float(np.mean(revenues)),
            'std_revenue': float(np.std(revenues)),
            'confidence_interval_95': [
                float(np.percentile(revenues, 2.5)),
                float(np.percentile(revenues, 97.5))
            ],
            'coefficient_of_variation': float(np.std(revenues) / np.mean(revenues) if np.mean(revenues) > 0 else 0)
        }
    
    def _calculate_confidence(
        self,
        optimization_success: bool,
        achieved_revenue: float,
        target_revenue: float,
        feasibility: Dict,
        uncertainty: Dict
    ) -> float:
        """
        Calculate overall confidence score for optimization
        
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from optimization success
        confidence = 0.7 if optimization_success else 0.4
        
        # Adjust for revenue achievement
        achievement_rate = achieved_revenue / target_revenue if target_revenue > 0 else 0
        if achievement_rate >= 0.95:
            confidence += 0.15
        elif achievement_rate >= 0.85:
            confidence += 0.10
        elif achievement_rate >= 0.75:
            confidence += 0.05
        else:
            confidence -= 0.10
        
        # Adjust for feasibility
        feasibility_score = feasibility['overall_score']
        confidence = confidence * (0.7 + 0.3 * feasibility_score)
        
        # Adjust for uncertainty
        cv = uncertainty['coefficient_of_variation']
        if cv < 0.05:  # Low uncertainty
            confidence += 0.05
        elif cv > 0.15:  # High uncertainty
            confidence -= 0.10
        
        # Clamp to [0, 1]
        return np.clip(confidence, 0.0, 1.0)
    
    def _setup_bounds(
        self,
        current_state: Dict[str, float],
        constraints: Optional[Dict]
    ) -> List[Tuple[float, float]]:
        """Setup optimization bounds from constraints"""
        bounds = []
        
        for lever_name in self.lever_names:
            base_bounds = self.lever_constraints[lever_name]
            
            # Apply custom constraints if provided
            if constraints and lever_name == 'retention_rate':
                max_increase = constraints.get('max_retention_increase', 0.05)
                lower = current_state[lever_name]
                upper = min(current_state[lever_name] + max_increase, base_bounds[1])
                bounds.append((lower, upper))
            
            elif constraints and lever_name == 'avg_ticket_price':
                max_increase = constraints.get('max_ticket_increase', 20.0)
                lower = current_state[lever_name]
                upper = current_state[lever_name] + max_increase
                bounds.append((lower, upper))
            
            elif constraints and lever_name == 'new_members':
                max_increase = constraints.get('max_new_members_increase', 10)
                lower = current_state[lever_name]
                upper = current_state[lever_name] + max_increase
                bounds.append((lower, upper))
            
            else:
                bounds.append(base_bounds)
        
        return bounds
    
    def _get_initial_guess(self, current_state: Dict[str, float]) -> np.ndarray:
        """Convert current state to array for optimization"""
        return np.array([current_state[lever] for lever in self.lever_names])
    
    def _array_to_levers(self, x: np.ndarray) -> Dict[str, float]:
        """Convert optimization array to lever dictionary"""
        levers = {}
        for i, lever_name in enumerate(self.lever_names):
            value = x[i]
            # Convert to int for discrete levers
            if lever_name in ['new_members', 'total_classes_held', 'total_members']:
                levers[lever_name] = int(round(value))
            else:
                levers[lever_name] = float(value)
        return levers


class ScenarioComparator:
    """Compare multiple optimization scenarios"""
    
    def __init__(self, optimizer: LeverOptimizer):
        self.optimizer = optimizer
    
    def compare_scenarios(
        self,
        target_revenue: float,
        current_state: Dict[str, float],
        historical_data,
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare multiple optimization scenarios with different constraints/objectives
        
        Args:
            target_revenue: Target revenue
            current_state: Current state
            historical_data: Historical data
            scenarios: List of scenario configurations, each with:
                - name: Scenario name
                - constraints: Constraints dict
                - objectives: List of objectives
                - method: Optimization method
        
        Returns:
            Comparison results with recommendations
        """
        results = []
        
        for scenario in scenarios:
            logger.info(f"Evaluating scenario: {scenario['name']}")
            
            result = self.optimizer.optimize(
                target_revenue=target_revenue,
                current_state=current_state,
                historical_data=historical_data,
                constraints=scenario.get('constraints'),
                method=scenario.get('method', 'auto'),
                objectives=scenario.get('objectives', ['revenue'])
            )
            
            result['scenario_name'] = scenario['name']
            results.append(result)
        
        # Rank scenarios
        for result in results:
            score = self._calculate_scenario_score(result)
            result['overall_score'] = score
        
        results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return {
            'scenarios': results,
            'recommended_scenario': results[0]['scenario_name'] if results else None,
            'comparison_summary': self._create_comparison_summary(results)
        }
    
    def _calculate_scenario_score(self, result: Dict) -> float:
        """Calculate overall score for a scenario"""
        score = (
            result['achievement_rate'] * 0.4 +
            result['confidence_score'] * 0.3 +
            result['feasibility_assessment']['overall_score'] * 0.3
        )
        return score
    
    def _create_comparison_summary(self, results: List[Dict]) -> str:
        """Create human-readable comparison summary"""
        if not results:
            return "No scenarios evaluated"
        
        summary_lines = []
        for i, result in enumerate(results[:3]):  # Top 3
            summary_lines.append(
                f"{i+1}. {result['scenario_name']}: "
                f"${result['achieved_revenue']:.0f} "
                f"({result['achievement_rate']*100:.1f}% of target), "
                f"Confidence: {result['confidence_score']:.2f}, "
                f"Feasibility: {result['feasibility_assessment']['overall_difficulty']}"
            )
        
        return "\n".join(summary_lines)

