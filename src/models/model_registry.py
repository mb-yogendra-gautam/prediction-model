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
                - shap_background: Background data for SHAP (optional)
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
            shap_background_path = self.base_dir / f'shap_background_v{version}.pkl'

            # Check if all required files exist
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

            # Load SHAP background data if available
            shap_background = None
            if shap_background_path.exists():
                shap_background = joblib.load(shap_background_path)
                logger.info(f"Loaded SHAP background data: {shap_background.shape}")
            else:
                logger.warning(f"SHAP background data not found at {shap_background_path}")

            logger.info(f"Successfully loaded model version {version}")
            logger.info(f"Model type: {metadata.get('best_model', 'unknown')}")
            logger.info(f"Number of features: {metadata.get('n_features', 'unknown')}")

            return {
                'model': model,
                'scaler': scaler,
                'selected_features': selected_features,
                'feature_selector': feature_selector,
                'shap_background': shap_background,
                'metadata': metadata,
                'version': version
            }

        except Exception as e:
            logger.error(f"Error loading model version {version}: {e}")
            raise

    def load_product_correlations(self, version: str = "2.2.0") -> Optional[Dict[str, Any]]:
        """
        Load pre-computed product correlation artifacts for a specific version

        Args:
            version: Model version to load (default: 2.2.0)

        Returns:
            Dictionary containing product correlation artifacts:
                - product_lever_correlations: Correlations between products and levers
                - product_revenue_correlations: Correlations between products and revenue
                - product_statistics: Statistical summaries for each product
                - correlation_matrix: Full correlation matrix
            Returns None if file doesn't exist
        """
        logger.info(f"Loading product correlations version {version}")

        try:
            correlation_path = self.base_dir / f'product_correlations_v{version}.pkl'

            if not correlation_path.exists():
                logger.warning(f"Product correlation file not found: {correlation_path}")
                return None

            # Load correlation artifacts
            artifacts = joblib.load(correlation_path)

            logger.info(f"Successfully loaded product correlations version {version}")
            logger.info(f"Products analyzed: {len(artifacts.get('product_lever_correlations', {}))}")

            return artifacts

        except Exception as e:
            logger.error(f"Error loading product correlations version {version}: {e}")
            return None

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

