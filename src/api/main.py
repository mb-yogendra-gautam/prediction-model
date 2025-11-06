"""
FastAPI Main Application for Studio Revenue Simulator
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from datetime import datetime

from src.models.model_registry import ModelRegistry
from src.api.services.prediction_service import PredictionService
from src.api.services.feature_service import FeatureService
from src.api.services.historical_data_service import HistoricalDataService
from src.api.services.explainability_service import ExplainabilityService
from src.api.services.counterfactual_service import CounterfactualService
from src.api.services.ai_insights_service import AIInsightsService
from src.api.routes import predictions
from src.api.schemas.responses import HealthCheckResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Studio Revenue Simulator API")
    logger.info("=" * 60)
    
    try:
        # Load model
        logger.info("Loading model artifacts...")
        registry = ModelRegistry(base_dir="data/models")
        model_artifacts = registry.load_model(version="2.2.0")
        app_state['model_artifacts'] = model_artifacts
        app_state['registry'] = registry
        
        logger.info(f"✓ Model version: {model_artifacts['version']}")
        logger.info(f"✓ Model type: {model_artifacts['metadata'].get('best_model', 'unknown')}")
        logger.info(f"✓ Number of features: {model_artifacts['metadata'].get('n_features', 'unknown')}")
        
        # Initialize services
        logger.info("Initializing services...")
        
        # Historical data service
        historical_service = HistoricalDataService()
        app_state['historical_service'] = historical_service
        logger.info("✓ Historical data service initialized")
        
        # Feature service
        feature_service = FeatureService(
            selected_features=model_artifacts['selected_features']
        )
        app_state['feature_service'] = feature_service
        logger.info("✓ Feature service initialized")

        # Explainability service
        explainability_service = ExplainabilityService(
            model=model_artifacts['model'],
            scaler=model_artifacts['scaler'],
            feature_names=model_artifacts['selected_features'],
            background_data=model_artifacts.get('shap_background')
        )
        app_state['explainability_service'] = explainability_service
        logger.info("✓ Explainability service initialized")

        # Counterfactual service
        counterfactual_service = CounterfactualService(
            model=model_artifacts['model'],
            scaler=model_artifacts['scaler'],
            feature_service=feature_service,
            historical_data_service=historical_service
        )
        app_state['counterfactual_service'] = counterfactual_service
        logger.info("✓ Counterfactual service initialized")

        # AI Insights service (LangChain + OpenAI)
        ai_insights_service = AIInsightsService(
            model_name="gpt-4o",  # Latest GPT-4 optimized model - faster and more cost-effective
            temperature=0.7,
            max_tokens=1000
        )
        app_state['ai_insights_service'] = ai_insights_service
        logger.info("✓ AI Insights service initialized")

        # Prediction service
        prediction_service = PredictionService(
            model=model_artifacts['model'],
            scaler=model_artifacts['scaler'],
            feature_service=feature_service,
            historical_data_service=historical_service,
            metadata=model_artifacts['metadata'],
            explainability_service=explainability_service,
            counterfactual_service=counterfactual_service,
            ai_insights_service=ai_insights_service
        )
        app_state['prediction_service'] = prediction_service

        # Set prediction service in routes
        predictions.set_prediction_service(prediction_service)
        logger.info("✓ Prediction service initialized")
        
        logger.info("=" * 60)
        logger.info("API startup complete! Server ready to accept requests.")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Studio Revenue Simulator API...")
    logger.info("Cleanup complete")


# Create FastAPI app
app = FastAPI(
    title="Studio Revenue Simulator API",
    description="Predict fitness studio revenue and optimize business levers using ML models",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    predictions.router,
    prefix="/api/v1/predict",
    tags=["Predictions"]
)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - Basic health check"""
    return {
        "service": "Studio Revenue Simulator API",
        "status": "healthy",
        "version": "1.0.0",
        "message": "Welcome! Visit /docs for API documentation"
    }


@app.get("/api/v1/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Detailed health check endpoint
    
    Returns:
    - Service status
    - Loaded model information
    - Available model versions
    """
    try:
        model_artifacts = app_state.get('model_artifacts', {})
        metadata = model_artifacts.get('metadata', {})
        registry = app_state.get('registry')
        
        # Get available versions
        available_versions = []
        if registry:
            available_versions = [v['version'] for v in registry.list_available_versions()]
        
        return {
            "status": "healthy",
            "model_version": metadata.get('version', 'unknown'),
            "model_type": metadata.get('best_model', 'unknown'),
            "n_features": metadata.get('n_features', 0),
            "available_versions": available_versions,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "degraded",
            "model_version": "unknown",
            "model_type": "unknown",
            "n_features": 0,
            "available_versions": [],
            "timestamp": datetime.now().isoformat()
        }


@app.get("/api/v1/studios", tags=["Studios"])
async def list_studios():
    """
    List all available studios in the system
    
    Returns:
    - List of studio IDs with basic information
    """
    try:
        historical_service = app_state.get('historical_service')
        if not historical_service:
            return {"studios": [], "count": 0}
        
        studio_ids = historical_service.get_all_studio_ids()
        
        # Get summary for each studio
        studios = []
        for studio_id in studio_ids[:20]:  # Limit to first 20 for performance
            summary = historical_service.get_studio_summary(studio_id)
            if summary:
                studios.append(summary)
        
        return {
            "studios": studios,
            "count": len(studio_ids),
            "showing": len(studios)
        }
    except Exception as e:
        logger.error(f"Error listing studios: {e}")
        return {"studios": [], "count": 0, "error": str(e)}


@app.get("/api/v1/studios/{studio_id}", tags=["Studios"])
async def get_studio_details(studio_id: str):
    """
    Get detailed information for a specific studio
    
    Args:
        studio_id: Unique studio identifier
        
    Returns:
    - Studio summary with historical statistics
    """
    try:
        historical_service = app_state.get('historical_service')
        if not historical_service:
            return {"error": "Historical data service not available"}
        
        summary = historical_service.get_studio_summary(studio_id)
        
        if not summary:
            return {"error": f"Studio {studio_id} not found"}
        
        return summary
    except Exception as e:
        logger.error(f"Error getting studio details: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

