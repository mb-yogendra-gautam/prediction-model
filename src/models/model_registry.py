"""
Model Registry for Loading and Managing Trained Models
"""

import joblib
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Centralized model registry for loading model artifacts"""

    def __init__(self, base_dir: str = "data/models"):
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise ValueError(f"Model directory not found: {base_dir}")

    def load_model(self, version: str = "2.2.0") -> Dict[str, Any]:
        """
        Load model artifacts for a specific version
        
        Args:
            version: Model version to load (default: 2.2.0)
            
        Returns:
            Dictionary containing model artifacts:
                - model: The trained model
                - scaler: Feature scaler
                - selected_features: List of selected feature names
                - feature_selector: Feature selector object
                - metadata: Model metadata
        """
        logger.info(f"Loading model version {version}")

        try:
            # Load model artifacts
            model_path = self.base_dir / f'best_model_v{version}.pkl'
            scaler_path = self.base_dir / f'scaler_v{version}.pkl'
            features_path = self.base_dir / f'selected_features_v{version}.pkl'
            selector_path = self.base_dir / f'feature_selector_v{version}.pkl'
            metadata_path = self.base_dir / f'metadata_v{version}.json'

            # Check if all files exist
            for path in [model_path, scaler_path, features_path, selector_path, metadata_path]:
                if not path.exists():
                    raise FileNotFoundError(f"Missing model artifact: {path}")

            # Load artifacts
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            selected_features = joblib.load(features_path)
            feature_selector = joblib.load(selector_path)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            logger.info(f"Successfully loaded model version {version}")
            logger.info(f"Model type: {metadata.get('best_model', 'unknown')}")
            logger.info(f"Number of features: {metadata.get('n_features', 'unknown')}")

            return {
                'model': model,
                'scaler': scaler,
                'selected_features': selected_features,
                'feature_selector': feature_selector,
                'metadata': metadata,
                'version': version
            }

        except Exception as e:
            logger.error(f"Error loading model version {version}: {e}")
            raise

    def list_available_versions(self):
        """List all available model versions"""
        versions = []
        for metadata_file in self.base_dir.glob('metadata_v*.json'):
            version = metadata_file.stem.replace('metadata_v', '')
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            versions.append({
                'version': version,
                'training_date': metadata.get('training_date'),
                'best_model': metadata.get('best_model'),
                'n_features': metadata.get('n_features')
            })
        return sorted(versions, key=lambda x: x['version'], reverse=True)

