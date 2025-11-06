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
from src.api.services.model_cache_service import ModelCacheManager
from src.api.routes import predictions
from src.api.routes import model_metadata
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
        # Initialize model registry
        logger.info("Initializing model registry...")
        registry = ModelRegistry(base_dir="data/models")
        app_state['registry'] = registry
        logger.info("✓ Model registry initialized")
        
        # Initialize services (shared across all models)
        logger.info("Initializing shared services...")
        
        # Historical data service
        historical_service = HistoricalDataService()
        app_state['historical_service'] = historical_service
        logger.info("✓ Historical data service initialized")

        # AI Insights service (LangChain + OpenAI)
        ai_insights_service = AIInsightsService(
            model_name="gpt-4o",  # Latest GPT-4 optimized model - faster and more cost-effective
            temperature=0.7,
            max_tokens=1000
        )
        app_state['ai_insights_service'] = ai_insights_service
        logger.info("✓ AI Insights service initialized")
        
        # Initialize Model Cache Manager
        logger.info("Initializing model cache manager...")
        model_cache_manager = ModelCacheManager(
            registry=registry,
            historical_data_service=historical_service,
            ai_insights_service=ai_insights_service,
            max_cache_size=5
        )
        app_state['model_cache_manager'] = model_cache_manager
        logger.info("✓ Model cache manager initialized")
        
        # Preload all latest models for fast initial responses
        logger.info("Preloading latest models...")
        
        # Ridge v2.2.0
        logger.info("  - Preloading Ridge v2.2.0...")
        model_cache_manager.preload_model("ridge", "2.2.0")
        logger.info("    ✓ Ridge v2.2.0 preloaded")

        # Ridge v2.3.0
        logger.info("  - Preloading Ridge v2.3.0...")
        model_cache_manager.preload_model("ridge", "2.3.0")
        logger.info("    ✓ Ridge v2.3.0 preloaded")

        # XGBoost v2.3.0
        logger.info("  - Preloading XGBoost v2.3.0...")
        model_cache_manager.preload_model("xgboost", "2.3.0")
        logger.info("    ✓ XGBoost v2.3.0 preloaded")
        
        # LightGBM v2.3.0
        logger.info("  - Preloading LightGBM v2.3.0...")
        model_cache_manager.preload_model("lightgbm", "2.3.0")
        logger.info("    ✓ LightGBM v2.3.0 preloaded")
        
        # Neural Network v2.3.0
        logger.info("  - Preloading Neural Network v2.3.0...")
        model_cache_manager.preload_model("neural_network", "2.3.0")
        logger.info("    ✓ Neural Network v2.3.0 preloaded")
        
        logger.info("✓ All 5 models preloaded successfully")
        
        # Get default prediction service for backward compatibility
        prediction_service = model_cache_manager.get_prediction_service("ridge", "2.2.0")
        app_state['prediction_service'] = prediction_service

        # Set services in routes
        predictions.set_prediction_service(prediction_service)  # Backward compatibility
        predictions.set_model_cache_manager(model_cache_manager)
        model_metadata.set_model_cache_manager(model_cache_manager)
        logger.info("✓ Prediction services initialized")
        
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

app.include_router(
    model_metadata.router,
    prefix="/api/v1/models",
    tags=["Model Metadata"]
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
    - Default model information
    - Available model versions
    - Cache statistics
    """
    try:
        cache_manager = app_state.get('model_cache_manager')
        registry = app_state.get('registry')
        
        # Get cache statistics
        cache_stats = cache_manager.get_cache_stats() if cache_manager else {}
        
        # Get available models
        available_models = []
        if registry:
            available_models = registry.list_available_models()
        
        return {
            "status": "healthy",
            "model_version": "2.2.0",  # Default model version
            "model_type": "ridge",  # Default model type
            "n_features": 15,  # Ridge v2.2.0 features
            "available_models": len(available_models),
            "cached_models": cache_stats.get('cached_models', 0),
            "cache_hit_rate": cache_stats.get('hit_rate_percent', 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "degraded",
            "model_version": "unknown",
            "model_type": "unknown",
            "n_features": 0,
            "available_models": 0,
            "cached_models": 0,
            "cache_hit_rate": 0,
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

