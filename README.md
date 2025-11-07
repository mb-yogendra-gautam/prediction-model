# Studio Revenue Simulator

A machine learning-powered business intelligence platform that enables fitness studio owners to predict revenue and optimize business levers for growth.

## Overview

The Studio Revenue Simulator uses ensemble machine learning models to provide:

- **Forward Prediction**: Predict future revenue based on operational lever adjustments
- **Inverse Prediction**: Determine optimal lever values to achieve target revenue goals
- **AI-Powered Insights**: Generate actionable recommendations based on predicted scenarios

## Features

- **Predictive Analytics**: Ensemble ML models (XGBoost + LightGBM + Random Forest) for accurate revenue forecasting
- **Scenario Planning**: Test "what-if" scenarios by adjusting business levers
- **Optimization Engine**: Find optimal lever combinations to reach revenue targets
- **RESTful API**: Production-ready FastAPI backend with automatic documentation
- **Real-time Caching**: Redis-powered caching for fast predictions
- **Model Versioning**: MLflow integration for experiment tracking and model management
- **AI Insights**: OpenAI-powered recommendations for business actions

## Project Structure

```
studio-revenue-simulator/
├── data/                       # Data storage
│   ├── raw/                    # Raw data files
│   ├── processed/              # Processed/engineered data
│   └── models/                 # Trained model artifacts
├── src/                        # Source code
│   ├── data/                   # Data generation & preprocessing
│   ├── features/               # Feature engineering
│   ├── models/                 # ML model implementations
│   ├── api/                    # FastAPI application
│   │   ├── routes/             # API endpoints
│   │   ├── schemas/            # Request/response schemas
│   │   └── services/           # Business logic
│   ├── database/               # Database models & connections
│   └── utils/                  # Utility functions
├── training/                   # Model training scripts
├── tests/                      # Unit & integration tests
├── notebooks/                  # Jupyter notebooks
├── config/                     # Configuration files
└── logs/                       # Application logs
```

## Technology Stack

### Backend

- **FastAPI**: Modern, high-performance web framework
- **Python 3.10+**: Primary programming language
- **Pydantic**: Data validation and settings management

### Machine Learning

- **scikit-learn**: ML utilities and preprocessing
- **XGBoost**: Gradient boosting for predictions
- **LightGBM**: Fast gradient boosting framework
- **MLflow**: Experiment tracking and model registry

### Database & Caching

- **PostgreSQL**: Primary database with TimescaleDB extension
- **Redis**: Caching layer for fast predictions

### AI & Analytics

- **OpenAI GPT-4**: AI-powered insights and recommendations
- **pandas & numpy**: Data manipulation and analysis

## Prerequisites

- Python 3.10 or higher
- PostgreSQL 15+ (with TimescaleDB extension)
- Redis 7+
- Git

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd prediction-model
```

### 2. Create Virtual Environment (You can use uv as well)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Update database credentials, API keys, etc.
```

### 5. Generate Sample Data

```bash
# Generate 5 years of synthetic studio data
python -m src.data.data_generator
```

### 6. Train Models

```bash
# Train the ensemble prediction model
python training/train_forward_model.py
```

### 7. Run API Server

```bash
# Development mode (auto-reload)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode (multiple workers)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 8. Access API Documentation

Open your browser and navigate to:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Usage

### Forward Prediction

Predict future revenue based on lever adjustments:

```bash
curl -X POST http://localhost:8000/api/v1/predict/forward \
  -H "Content-Type: application/json" \
  -d '{
    "studio_id": "studio-123",
    "base_month": "2024-01-01",
    "projection_months": 3,
    "levers": {
      "retention_rate": 0.75,
      "avg_ticket_price": 150.0,
      "class_attendance_rate": 0.70,
      "new_members_monthly": 25,
      "staff_utilization_rate": 0.85,
      "upsell_rate": 0.25,
      "total_classes_held": 120,
      "current_member_count": 250
    }
  }'
```

### Inverse Prediction

Find optimal levers to achieve target revenue:

```bash
curl -X POST http://localhost:8000/api/v1/predict/inverse \
  -H "Content-Type: application/json" \
  -d '{
    "studio_id": "studio-123",
    "base_month": "2024-01-01",
    "target_revenue": 50000.0,
    "current_state": {
      "retention_rate": 0.70,
      "avg_ticket_price": 140.0,
      "class_attendance_rate": 0.65,
      "new_members_monthly": 20,
      "staff_utilization_rate": 0.80,
      "upsell_rate": 0.20,
      "total_classes_held": 100,
      "current_member_count": 250
    }
  }'
```

## Business Levers

The platform supports the following adjustable business levers:

1. **Retention Rate** (0.5 - 1.0): Member retention percentage
2. **Average Ticket Price** ($50 - $500): Monthly membership cost
3. **Class Attendance Rate** (0.4 - 1.0): Percentage of classes attended
4. **New Members Monthly** (0 - 100): New member acquisitions per month
5. **Staff Utilization Rate** (0.6 - 1.0): Percentage of staff capacity used
6. **Upsell Rate** (0.0 - 0.5): Percentage of members purchasing add-ons
7. **Total Classes Held** (50 - 500): Number of classes offered per month

## Model Performance

The ensemble model achieves:

- **RMSE**: < $3,000 for monthly revenue predictions
- **MAPE**: < 8% mean absolute percentage error
- **R² Score**: > 0.85 goodness of fit
- **Directional Accuracy**: > 90% for growth/decline predictions

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Formatting

```bash
# Format code
black src/ training/ tests/

# Check formatting
black --check src/ training/ tests/
```

### Linting

```bash
# Lint code
flake8 src/ training/ tests/
```

## Database Setup

### PostgreSQL with TimescaleDB

1. Install PostgreSQL and TimescaleDB extension
2. Create database:

```sql
CREATE DATABASE studio_simulator;
\c studio_simulator
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

3. Run migrations (schema in comprehensive_development_prompt.md)

## MLflow Tracking

Start MLflow tracking server:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Access MLflow UI at http://localhost:5000

## Configuration

### Environment Variables

Key environment variables in `.env`:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/studio_simulator
REDIS_URL=redis://localhost:6379/0
MLFLOW_TRACKING_URI=http://localhost:5000
OPENAI_API_KEY=your_api_key_here
MODEL_VERSION=1.0.0
LOG_LEVEL=INFO
```

### Model Configuration

Adjust model hyperparameters in `config/model_config.yaml`

## Monitoring

### Prediction Logging

All predictions are logged to the database for:

- Model performance tracking
- Prediction accuracy analysis
- Business insights

### Performance Metrics

Monitor API performance:

- Forward prediction latency: < 200ms (p95)
- Inverse optimization latency: < 1000ms (p95)
- Cache hit rate: > 60%

## Troubleshooting

### Common Issues

**Issue**: Module not found errors

```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Issue**: Database connection errors

```bash
# Solution: Check PostgreSQL is running
# Verify DATABASE_URL in .env
```

**Issue**: Redis connection errors

```bash
# Solution: Start Redis server
redis-server
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write tests
5. Submit a pull request

## License

[Add your license here]

## Contact

[Add contact information here]

## Acknowledgments

- Built for Mindbody Studios
- Uses state-of-the-art ML ensemble techniques
- Powered by FastAPI and modern Python stack
