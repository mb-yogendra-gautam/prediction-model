"""
Script to run the Studio Revenue Simulator API
"""

import uvicorn
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_development():
    """Run API in development mode with auto-reload"""
    logger.info("Starting API in DEVELOPMENT mode...")
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


def run_production():
    """Run API in production mode with multiple workers"""
    logger.info("Starting API in PRODUCTION mode...")
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info"
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "development"
    
    if mode == "production":
        run_production()
    else:
        run_development()

