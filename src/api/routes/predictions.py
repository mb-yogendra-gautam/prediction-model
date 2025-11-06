"""
API Routes for Predictions
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from src.api.schemas.requests import (
    ForwardPredictionRequest,
    InversePredictionRequest,
    PartialPredictionRequest
)
from src.api.schemas.responses import (
    ForwardPredictionResponse,
    InversePredictionResponse,
    PartialPredictionResponse
)
from src.api.schemas.lever_metadata import get_all_levers, LEVER_REGISTRY
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Global state for services (will be injected)
_prediction_service = None  # Kept for backward compatibility
_model_cache_manager = None


def set_prediction_service(service):
    """Set the prediction service instance (backward compatibility)"""
    global _prediction_service
    _prediction_service = service


def set_model_cache_manager(cache_manager):
    """Set the model cache manager instance"""
    global _model_cache_manager
    _model_cache_manager = cache_manager


def get_prediction_service():
    """Dependency to get prediction service (backward compatibility)"""
    if _prediction_service is None:
        raise HTTPException(status_code=500, detail="Prediction service not initialized")
    return _prediction_service


def get_model_cache_manager():
    """Dependency to get model cache manager"""
    if _model_cache_manager is None:
        raise HTTPException(status_code=500, detail="Model cache manager not initialized")
    return _model_cache_manager


@router.post("/forward", response_model=ForwardPredictionResponse)
async def forward_prediction(
    request: ForwardPredictionRequest,
    include_ai_insights: bool = Query(False, description="Generate AI-powered business insights using LangChain and OpenAI")
):
    """
    Forward prediction: Given lever values, predict future revenue

    **Input levers:**
    - retention_rate: Member retention rate (0.5-1.0)
    - avg_ticket_price: Average monthly ticket price ($50-$500)
    - class_attendance_rate: Class attendance rate (0.4-1.0)
    - new_members: New members per month (0-100)
    - staff_utilization_rate: Staff utilization rate (0.6-1.0)
    - upsell_rate: Upsell rate (0.0-0.5)
    - total_classes_held: Total classes per month (50-500)
    - total_members: Current total member count (min 50)

    **Model Selection:**
    - model_type: 'xgboost', 'lightgbm', 'neural_network', 'ridge' (default: 'ridge')
    - model_version: '2.2.0', '2.3.0' (default: '2.2.0')

    **Query Parameters:**
    - include_ai_insights: Set to true to get AI-generated business insights (requires OpenAI API key)

    **Returns:**
    - Monthly revenue predictions for 1-3 months
    - Member count projections
    - Confidence scores
    - AI insights (if requested)
    """
    try:
        # Get model parameters (use defaults if not specified)
        model_type = request.model_type or "ridge"
        model_version = request.model_version or "2.2.0"
        
        # Get prediction service for specified model
        cache_manager = get_model_cache_manager()
        service = cache_manager.get_prediction_service(model_type, model_version)

        # Convert request to dict
        request_dict = {
            'studio_id': request.studio_id,
            'levers': request.levers.dict(),
            'projection_months': request.projection_months
        }

        result = service.predict_forward(request_dict, include_ai_insights=include_ai_insights)
        
        # Add model information to response
        result['model_info'] = {
            'model_type': model_type,
            'model_version': model_version
        }
        
        return result
        
    except ValueError as e:
        logger.error(f"Invalid model selection: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
    except Exception as e:
        logger.error(f"Forward prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/inverse", response_model=InversePredictionResponse)
async def inverse_prediction(
    request: InversePredictionRequest,
    include_ai_insights: bool = Query(False, description="Generate AI-powered business insights using LangChain and OpenAI")
):
    """
    Inverse prediction: Given target revenue, find optimal lever values

    **Process:**
    1. Takes target revenue and current business state
    2. Uses optimization to find best lever adjustments
    3. Returns recommended lever values within constraints
    4. Provides prioritized action plan

    **Model Selection:**
    - model_type: 'xgboost', 'lightgbm', 'neural_network', 'ridge' (default: 'ridge')
    - model_version: '2.2.0', '2.3.0' (default: '2.2.0')

    **Query Parameters:**
    - include_ai_insights: Set to true to get AI-generated strategic insights (requires OpenAI API key)

    **Returns:**
    - Optimized lever values
    - Achievable revenue (may differ from target)
    - Detailed lever changes with priorities
    - Action plan with timeline
    - AI insights (if requested)
    """
    try:
        # Get model parameters (use defaults if not specified)
        model_type = request.model_type or "ridge"
        model_version = request.model_version or "2.2.0"
        
        # Get prediction service for specified model
        cache_manager = get_model_cache_manager()
        service = cache_manager.get_prediction_service(model_type, model_version)

        # Convert request to dict
        request_dict = {
            'studio_id': request.studio_id,
            'target_revenue': request.target_revenue,
            'current_state': request.current_state.dict(),
            'constraints': request.constraints.dict() if request.constraints else {},
            'target_months': request.target_months
        }

        result = service.predict_inverse(request_dict, include_ai_insights=include_ai_insights)
        
        # Add model information to response
        result['model_info'] = {
            'model_type': model_type,
            'model_version': model_version
        }
        
        return result
        
    except ValueError as e:
        logger.error(f"Invalid model selection: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
    except Exception as e:
        logger.error(f"Inverse prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.post("/partial", response_model=PartialPredictionResponse)
async def partial_lever_prediction(
    request: PartialPredictionRequest,
    include_ai_insights: bool = Query(False, description="Generate AI-powered business insights using LangChain and OpenAI")
):
    """
    Partial lever prediction: Given subset of levers, predict remaining levers

    **Use cases:**
    - "If I know retention and price, what attendance rate should I expect?"
    - "Given my current members and classes, what revenue can I achieve?"
    - "What new member count do I need to hit my revenue target?"

    **Input:**
    - input_levers: Dictionary of known lever values
    - output_levers: List of lever names to predict

    **Supported output levers:**
    - All 8 primary levers (retention_rate, avg_ticket_price, etc.)
    - total_revenue (special case)

    **Model Selection:**
    - model_type: 'xgboost', 'lightgbm', 'neural_network', 'ridge' (default: 'ridge')
    - model_version: '2.2.0', '2.3.0' (default: '2.2.0')

    **Query Parameters:**
    - include_ai_insights: Set to true to get AI-generated insights about predicted levers (requires OpenAI API key)

    **Returns:**
    - Predicted values for requested levers
    - Confidence scores for each prediction
    - Value ranges (min, max)
    - AI insights (if requested)
    """
    try:
        # Get model parameters (use defaults if not specified)
        model_type = request.model_type or "ridge"
        model_version = request.model_version or "2.2.0"
        
        # Get prediction service for specified model
        cache_manager = get_model_cache_manager()
        service = cache_manager.get_prediction_service(model_type, model_version)

        # Convert request to dict
        request_dict = {
            'studio_id': request.studio_id,
            'input_levers': request.input_levers,
            'output_levers': request.output_levers
        }

        result = service.predict_partial_levers(request_dict, include_ai_insights=include_ai_insights)
        
        # Add model information to response
        result['model_info'] = {
            'model_type': model_type,
            'model_version': model_version
        }
        
        return result
        
    except ValueError as e:
        logger.error(f"Invalid model selection: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
    except Exception as e:
        logger.error(f"Partial prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Partial prediction failed: {str(e)}")


@router.get("/levers")
async def get_business_levers(
    include_details: bool = Query(False, description="Include feasibility thresholds and action templates")
):
    """
    Get all available business levers with their metadata

    Returns comprehensive information about all 8 business levers including:
    - Name and display name
    - Data type (float or integer)
    - Constraints (min/max values)
    - Description and unit of measurement
    - Default value and priority
    - Optional: Feasibility thresholds and action templates

    **Query Parameters:**
    - include_details: Set to true to include implementation guidance (feasibility thresholds, action templates)

    **Use Cases:**
    - Dynamic form generation in frontend applications
    - Validation of user inputs
    - Understanding lever constraints and priorities
    - Implementation planning with feasibility guidance

    **Returns:**
    - levers: Array of lever metadata objects sorted by priority
    - count: Total number of levers (always 8)
    - version: Metadata schema version
    """
    try:
        levers = get_all_levers(include_details=include_details)

        return {
            "levers": levers,
            "count": len(levers),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Error retrieving levers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve levers: {str(e)}")


@router.get("/models")
async def get_available_models():
    """
    Get list of available models with metadata and capabilities
    
    **Returns:**
    - List of available models with:
      - model_type: Type of model (e.g., 'ridge', 'xgboost')
      - version: Model version (e.g., '2.2.0', '2.3.0')
      - is_default: Whether this is the default model
      - targets: List of prediction targets
      - n_features: Number of input features
      - prediction_horizons: Supported prediction horizons
      - training_date: When the model was trained
      - performance: Model performance metrics (if available)
    - default_model: The default model configuration
    - cache_stats: Cache statistics
    
    **Use Cases:**
    - Discover available models for predictions
    - Check model capabilities and performance
    - Select optimal model for specific use case
    """
    try:
        cache_manager = get_model_cache_manager()
        
        # Get available models from registry
        available_models = cache_manager.registry.list_available_models()
        
        # Define target names for each model version
        v2_2_targets = [
            'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
            'member_count_month_3', 'retention_rate_month_3'
        ]
        
        v2_3_targets = [
            'revenue_day_1', 'revenue_day_3', 'revenue_day_7', 'attendance_day_7',
            'revenue_week_1', 'revenue_week_2', 'revenue_week_4', 'attendance_week_1',
            'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
            'member_count_month_1', 'member_count_month_3', 'retention_rate_month_3'
        ]
        
        # Format models with additional information
        models_list = []
        for model in available_models:
            model_info = {
                'model_type': model['model_type'],
                'version': model['version'],
                'is_default': (model['model_type'] == 'ridge' and model['version'] == '2.2.0'),
                'targets': v2_2_targets if model['version'] == '2.2.0' else v2_3_targets,
                'n_features': model['n_features'],
                'n_targets': model['n_targets'],
                'prediction_horizons': model['prediction_horizons'],
                'training_date': model.get('training_date'),
                'algorithm': model.get('algorithm') or model.get('best_model')
            }
            
            # Add performance metrics for v2.3.0 models if available
            if model['version'] == '2.3.0':
                try:
                    import json
                    from pathlib import Path
                    results_path = Path('reports/audit') / f"model_results_v{model['version']}_{model['model_type']}.json"
                    if results_path.exists():
                        with open(results_path, 'r') as f:
                            results = json.load(f)
                            if 'test_results' in results:
                                model_info['performance'] = {
                                    'r2': results['test_results'].get('overall_r2'),
                                    'rmse': results['test_results'].get('overall_rmse'),
                                    'mae': results['test_results'].get('overall_mae')
                                }
                            if 'business_metrics' in results:
                                model_info['business_metrics'] = results['business_metrics']
                except Exception as e:
                    logger.warning(f"Could not load performance metrics for {model['model_type']} v{model['version']}: {e}")
            
            models_list.append(model_info)
        
        # Get cache statistics
        cache_stats = cache_manager.get_cache_stats()
        
        return {
            'models': models_list,
            'count': len(models_list),
            'default_model': {
                'model_type': 'ridge',
                'version': '2.2.0'
            },
            'cache_stats': cache_stats
        }
        
    except Exception as e:
        logger.error(f"Error retrieving models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve models: {str(e)}")


@router.post("/inverse/compare-scenarios")
async def compare_optimization_scenarios(request: dict):
    """
    Compare multiple optimization scenarios with different constraints and objectives
    
    **Use case:**
    Compare different strategies to achieve revenue target:
    - Conservative: Minimal changes, focus on easy wins
    - Balanced: Mix of difficulty, optimize for revenue only
    - Aggressive: Allow larger changes, multi-objective optimization
    - Growth-focused: Prioritize member growth and retention
    
    **Input:**
    - target_revenue: Target revenue goal
    - current_state: Current lever values
    - scenarios: List of scenario configurations with name, constraints, objectives, method
    
    **Returns:**
    - Results for each scenario
    - Recommended scenario based on overall score
    - Comparison summary
    
    **Example:**
    ```json
    {
      "studio_id": "STU001",
      "target_revenue": 35000,
      "current_state": {...},
      "scenarios": [
        {
          "name": "Conservative",
          "constraints": {"max_retention_increase": 0.03, "max_ticket_increase": 10},
          "objectives": ["revenue"],
          "method": "auto"
        },
        {
          "name": "Aggressive",
          "constraints": {"max_retention_increase": 0.10, "max_ticket_increase": 50},
          "objectives": ["revenue", "retention", "growth"],
          "method": "ensemble"
        }
      ]
    }
    ```
    """
    try:
        service = get_prediction_service()
        
        # Validate required fields
        if 'studio_id' not in request or 'target_revenue' not in request:
            raise HTTPException(status_code=400, detail="Missing required fields: studio_id, target_revenue")
        
        if 'current_state' not in request or 'scenarios' not in request:
            raise HTTPException(status_code=400, detail="Missing required fields: current_state, scenarios")
        
        studio_id = request['studio_id']
        target_revenue = request['target_revenue']
        current_state = request['current_state']
        scenarios = request['scenarios']
        
        # Get historical data
        historical_data = service.historical_data_service.get_studio_history(studio_id, n_months=12)
        
        # Run scenario comparison
        result = service.scenario_comparator.compare_scenarios(
            target_revenue=target_revenue,
            current_state=current_state,
            historical_data=historical_data,
            scenarios=scenarios
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Scenario comparison error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scenario comparison failed: {str(e)}")
