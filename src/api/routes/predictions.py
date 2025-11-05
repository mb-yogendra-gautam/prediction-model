"""
API Routes for Predictions
"""

from fastapi import APIRouter, HTTPException, Depends
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
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Global state for service (will be injected)
_prediction_service = None


def set_prediction_service(service):
    """Set the prediction service instance"""
    global _prediction_service
    _prediction_service = service


def get_prediction_service():
    """Dependency to get prediction service"""
    if _prediction_service is None:
        raise HTTPException(status_code=500, detail="Prediction service not initialized")
    return _prediction_service


@router.post("/forward", response_model=ForwardPredictionResponse)
async def forward_prediction(request: ForwardPredictionRequest):
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
    
    **Returns:**
    - Monthly revenue predictions for 1-3 months
    - Member count projections
    - Confidence scores
    """
    try:
        service = get_prediction_service()
        
        # Convert request to dict
        request_dict = {
            'studio_id': request.studio_id,
            'levers': request.levers.dict(),
            'projection_months': request.projection_months
        }
        
        result = service.predict_forward(request_dict)
        return result
        
    except Exception as e:
        logger.error(f"Forward prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/inverse", response_model=InversePredictionResponse)
async def inverse_prediction(request: InversePredictionRequest):
    """
    Inverse prediction: Given target revenue, find optimal lever values
    
    **Process:**
    1. Takes target revenue and current business state
    2. Uses optimization to find best lever adjustments
    3. Returns recommended lever values within constraints
    4. Provides prioritized action plan
    
    **Returns:**
    - Optimized lever values
    - Achievable revenue (may differ from target)
    - Detailed lever changes with priorities
    - Action plan with timeline
    """
    try:
        service = get_prediction_service()
        
        # Convert request to dict
        request_dict = {
            'studio_id': request.studio_id,
            'target_revenue': request.target_revenue,
            'current_state': request.current_state.dict(),
            'constraints': request.constraints.dict() if request.constraints else {},
            'target_months': request.target_months
        }
        
        result = service.predict_inverse(request_dict)
        return result
        
    except Exception as e:
        logger.error(f"Inverse prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.post("/partial", response_model=PartialPredictionResponse)
async def partial_lever_prediction(request: PartialPredictionRequest):
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
    
    **Returns:**
    - Predicted values for requested levers
    - Confidence scores for each prediction
    - Value ranges (min, max)
    """
    try:
        service = get_prediction_service()
        
        # Convert request to dict
        request_dict = {
            'studio_id': request.studio_id,
            'input_levers': request.input_levers,
            'output_levers': request.output_levers
        }
        
        result = service.predict_partial_levers(request_dict)
        return result
        
    except Exception as e:
        logger.error(f"Partial prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Partial prediction failed: {str(e)}")

