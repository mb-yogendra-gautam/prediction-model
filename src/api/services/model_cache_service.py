"""
Model Cache Manager for Dynamic Model Selection

Provides LRU caching for multiple models with thread-safe loading
and cache statistics.
"""

import logging
from typing import Dict, Any, Optional
from collections import OrderedDict
from threading import Lock
from datetime import datetime

from src.models.model_registry import ModelRegistry
from src.api.services.prediction_service import PredictionService
from src.api.services.feature_service import FeatureService
from src.api.services.historical_data_service import HistoricalDataService
from src.api.services.explainability_service import ExplainabilityService
from src.api.services.counterfactual_service import CounterfactualService
from src.api.services.ai_insights_service import AIInsightsService

logger = logging.getLogger(__name__)


class ModelCacheManager:
    """
    Manages multiple models with LRU caching
    
    Features:
    - Thread-safe model loading
    - LRU eviction policy
    - Lazy loading on first request
    - Cache statistics
    """
    
    def __init__(
        self, 
        registry: ModelRegistry,
        historical_data_service: HistoricalDataService,
        ai_insights_service: AIInsightsService,
        max_cache_size: int = 5
    ):
        """
        Initialize Model Cache Manager
        
        Args:
            registry: ModelRegistry instance
            historical_data_service: Historical data service
            ai_insights_service: AI insights service
            max_cache_size: Maximum number of models to cache (default: 5)
        """
        self.registry = registry
        self.historical_data_service = historical_data_service
        self.ai_insights_service = ai_insights_service
        self.max_cache_size = max_cache_size
        
        # Cache storage (model_key -> {artifacts, service, metadata})
        self._cache = OrderedDict()
        self._lock = Lock()
        
        # Statistics
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'models_loaded': 0,
            'models_evicted': 0
        }
        
        logger.info(f"ModelCacheManager initialized with max cache size: {max_cache_size}")
    
    def get_or_load_model(self, model_type: str, version: str) -> Dict[str, Any]:
        """
        Get model artifacts from cache or load from disk
        
        Args:
            model_type: Model type ('ridge', 'xgboost', 'lightgbm', 'neural_network')
            version: Model version (e.g., '2.2.0', '2.3.0')
        
        Returns:
            Dictionary with model artifacts
        """
        model_key = self.registry.get_model_key(model_type, version)
        
        with self._lock:
            # Check if model is in cache
            if model_key in self._cache:
                logger.info(f"Cache HIT for {model_key}")
                self._stats['cache_hits'] += 1
                # Move to end (most recently used)
                self._cache.move_to_end(model_key)
                return self._cache[model_key]['artifacts']
            
            logger.info(f"Cache MISS for {model_key}")
            self._stats['cache_misses'] += 1
            
            # Load model from disk
            artifacts = self._load_model_artifacts(model_type, version)
            
            # Evict oldest if cache is full
            if len(self._cache) >= self.max_cache_size:
                self._evict_oldest()
            
            # Add to cache
            self._cache[model_key] = {
                'artifacts': artifacts,
                'service': None,  # Lazy load service
                'loaded_at': datetime.now().isoformat(),
                'model_type': model_type,
                'version': version
            }
            
            self._stats['models_loaded'] += 1
            
            return artifacts
    
    def get_prediction_service(
        self, 
        model_type: str, 
        version: str
    ) -> PredictionService:
        """
        Get PredictionService for specified model
        
        Args:
            model_type: Model type ('ridge', 'xgboost', 'lightgbm', 'neural_network')
            version: Model version (e.g., '2.2.0', '2.3.0')
        
        Returns:
            PredictionService instance
        """
        model_key = self.registry.get_model_key(model_type, version)
        
        with self._lock:
            # Get or load model artifacts
            if model_key not in self._cache:
                # This will load and cache the artifacts
                self.get_or_load_model(model_type, version)
            
            # Check if service already exists
            if self._cache[model_key]['service'] is not None:
                logger.info(f"Returning cached service for {model_key}")
                return self._cache[model_key]['service']
            
            # Create new service
            logger.info(f"Creating new PredictionService for {model_key}")
            artifacts = self._cache[model_key]['artifacts']
            service = self._create_prediction_service(artifacts, model_type, version)
            
            # Cache the service
            self._cache[model_key]['service'] = service
            
            return service
    
    def _load_model_artifacts(self, model_type: str, version: str) -> Dict[str, Any]:
        """Load model artifacts from disk"""
        logger.info(f"Loading {model_type} v{version} from disk...")
        
        try:
            # Handle v2.2.0 (Ridge) models
            if version == "2.2.0" and model_type == "ridge":
                return self.registry.load_model(version=version)

            # Handle v2.3.0 models
            elif version == "2.3.0" and model_type in ['xgboost', 'lightgbm', 'neural_network', 'ridge']:
                return self.registry.load_v2_3_model(model_type=model_type, version=version)
            
            else:
                raise ValueError(f"Unsupported model type/version: {model_type} v{version}")
        
        except Exception as e:
            logger.error(f"Failed to load {model_type} v{version}: {e}")
            raise
    
    def _create_prediction_service(
        self, 
        artifacts: Dict[str, Any],
        model_type: str,
        version: str
    ) -> PredictionService:
        """Create PredictionService from model artifacts"""
        
        # Create feature service
        feature_service = FeatureService(
            selected_features=artifacts['selected_features']
        )
        
        # Create explainability service
        explainability_service = ExplainabilityService(
            model=artifacts['model'],
            scaler=artifacts['scaler'],
            feature_names=artifacts['selected_features'],
            background_data=artifacts.get('shap_background')
        )
        
        # Create counterfactual service
        counterfactual_service = CounterfactualService(
            model=artifacts['model'],
            scaler=artifacts['scaler'],
            feature_service=feature_service,
            historical_data_service=self.historical_data_service
        )
        
        # Create prediction service
        prediction_service = PredictionService(
            model=artifacts['model'],
            scaler=artifacts['scaler'],
            feature_service=feature_service,
            historical_data_service=self.historical_data_service,
            metadata=artifacts['metadata'],
            explainability_service=explainability_service,
            counterfactual_service=counterfactual_service,
            ai_insights_service=self.ai_insights_service
        )
        
        # Add model type and version to service
        prediction_service.model_type = model_type
        prediction_service.model_version = version
        
        return prediction_service
    
    def _evict_oldest(self):
        """Evict least recently used model"""
        if not self._cache:
            return
        
        # Remove first item (oldest)
        oldest_key = next(iter(self._cache))
        evicted = self._cache.pop(oldest_key)
        
        logger.info(f"Evicted model from cache: {oldest_key}")
        logger.info(f"  Loaded at: {evicted['loaded_at']}")
        
        self._stats['models_evicted'] += 1
    
    def clear_cache(self):
        """Clear all cached models"""
        with self._lock:
            num_cleared = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared {num_cleared} models from cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._stats['cache_hits'] + self._stats['cache_misses']
            hit_rate = (self._stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'cache_hits': self._stats['cache_hits'],
                'cache_misses': self._stats['cache_misses'],
                'total_requests': total_requests,
                'hit_rate_percent': round(hit_rate, 2),
                'models_loaded': self._stats['models_loaded'],
                'models_evicted': self._stats['models_evicted'],
                'current_cache_size': len(self._cache),
                'max_cache_size': self.max_cache_size,
                'cached_models': [
                    {
                        'model_key': key,
                        'model_type': data['model_type'],
                        'version': data['version'],
                        'loaded_at': data['loaded_at']
                    }
                    for key, data in self._cache.items()
                ]
            }
    
    def validate_model_request(self, model_type: Optional[str], version: Optional[str]) -> tuple:
        """
        Validate and normalize model request parameters
        
        Args:
            model_type: Requested model type (or None for default)
            version: Requested version (or None for default)
        
        Returns:
            Tuple of (validated_model_type, validated_version)
        
        Raises:
            ValueError: If model type/version combination is invalid
        """
        # Default to ridge v2.2.0
        if model_type is None:
            model_type = "ridge"
        if version is None:
            version = "2.2.0"
        
        # Normalize model type
        model_type = model_type.lower()
        
        # Validate combinations
        valid_combinations = {
            ('ridge', '2.2.0'),
            ('ridge', '2.0.0'),
            ('ridge', '2.1.0'),
            ('ridge', '2.3.0'),
            ('xgboost', '2.3.0'),
            ('lightgbm', '2.3.0'),
            ('neural_network', '2.3.0')
        }
        
        if (model_type, version) not in valid_combinations:
            available = "\n  - ".join([f"{mt} v{v}" for mt, v in valid_combinations])
            raise ValueError(
                f"Invalid model type/version combination: {model_type} v{version}\n"
                f"Available models:\n  - {available}"
            )
        
        return model_type, version
    
    def list_available_models(self) -> list:
        """
        List all available models
        
        Returns:
            List of available models with metadata
        """
        return self.registry.list_available_models()
    
    def preload_model(self, model_type: str, version: str):
        """
        Preload a model into cache
        
        Args:
            model_type: Model type to preload
            version: Model version to preload
        """
        logger.info(f"Preloading {model_type} v{version}...")
        self.get_or_load_model(model_type, version)
        logger.info(f"Successfully preloaded {model_type} v{version}")
