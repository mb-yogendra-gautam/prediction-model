"""
Business Metrics Calculator

Computes business-specific KPIs and performance indicators
for revenue prediction model evaluation and audit purposes.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_business_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate business-specific metrics for model evaluation

    Args:
        y_true: Actual values (n_samples, 5 targets)
        y_pred: Predicted values (n_samples, 5 targets)

    Returns:
        Dictionary of business metrics
    """
    logger.info("Calculating business metrics")

    # Revenue predictions (first 3 targets)
    revenue_true = y_true[:, :3]
    revenue_pred = y_pred[:, :3]

    # Calculate percentage errors for revenue
    pct_error = np.abs((revenue_pred - revenue_true) / revenue_true) * 100

    # 1. Within 5% accuracy rate
    within_5pct = np.mean(pct_error <= 5.0) * 100

    # 2. Within 10% accuracy rate
    within_10pct = np.mean(pct_error <= 10.0) * 100

    # 3. Revenue forecast accuracy (3-month cumulative)
    cumulative_true = np.sum(revenue_true, axis=1)
    cumulative_pred = np.sum(revenue_pred, axis=1)
    forecast_accuracy = (1 - np.mean(np.abs(
        (cumulative_pred - cumulative_true) / cumulative_true
    ))) * 100

    # 4. Directional accuracy (did we predict growth/decline correctly?)
    if revenue_true.shape[1] > 1:
        direction_true = np.sign(np.diff(revenue_true, axis=1))
        direction_pred = np.sign(np.diff(revenue_pred, axis=1))
        directional_accuracy = np.mean(direction_true == direction_pred) * 100
    else:
        directional_accuracy = None

    # 5. Business impact score (heavier penalty for large errors)
    penalty = np.where(pct_error > 10, 2.0, 1.0)
    business_impact_score = 100 - np.mean(pct_error * penalty)

    # 6. Mean absolute percentage error by horizon
    mape_month_1 = np.mean(np.abs((revenue_pred[:, 0] - revenue_true[:, 0]) / revenue_true[:, 0])) * 100
    mape_month_2 = np.mean(np.abs((revenue_pred[:, 1] - revenue_true[:, 1]) / revenue_true[:, 1])) * 100
    mape_month_3 = np.mean(np.abs((revenue_pred[:, 2] - revenue_true[:, 2]) / revenue_true[:, 2])) * 100

    # 7. Member prediction accuracy (4th target)
    member_true = y_true[:, 3]
    member_pred = y_pred[:, 3]
    member_mape = np.mean(np.abs((member_pred - member_true) / member_true)) * 100

    # 8. Retention prediction accuracy (5th target)
    retention_true = y_true[:, 4]
    retention_pred = y_pred[:, 4]
    retention_mae = np.mean(np.abs(retention_pred - retention_true))

    # 9. Revenue trend consistency (sequential predictions)
    if len(revenue_true) > 1:
        true_trends = np.sign(np.diff(revenue_true[:, 0]))  # Month 1 trends
        pred_trends = np.sign(np.diff(revenue_pred[:, 0]))
        trend_consistency = np.mean(true_trends == pred_trends) * 100
    else:
        trend_consistency = None

    # 10. Large error rate (errors > 15%)
    large_error_rate = np.mean(pct_error > 15.0) * 100

    metrics = {
        'within_5_percent': round(within_5pct, 2),
        'within_10_percent': round(within_10pct, 2),
        'forecast_accuracy': round(forecast_accuracy, 2),
        'directional_accuracy': round(directional_accuracy, 2) if directional_accuracy is not None else None,
        'business_impact_score': round(business_impact_score, 2),
        'mape_month_1': round(mape_month_1, 2),
        'mape_month_2': round(mape_month_2, 2),
        'mape_month_3': round(mape_month_3, 2),
        'member_prediction_mape': round(member_mape, 2),
        'retention_prediction_mae': round(retention_mae, 4),
        'revenue_trend_consistency': round(trend_consistency, 2) if trend_consistency is not None else None,
        'large_error_rate': round(large_error_rate, 2)
    }

    logger.info("Business metrics calculated successfully")
    for key, value in metrics.items():
        if value is not None:
            logger.info(f"  {key}: {value}")

    return metrics


def calculate_revenue_breakdown_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict]:
    """
    Calculate detailed metrics for each revenue horizon

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary with metrics for each horizon
    """
    revenue_true = y_true[:, :3]
    revenue_pred = y_pred[:, :3]

    horizons = ['Month_1', 'Month_2', 'Month_3']
    breakdown = {}

    for i, horizon in enumerate(horizons):
        true = revenue_true[:, i]
        pred = revenue_pred[:, i]

        # Calculate various error metrics
        abs_errors = np.abs(pred - true)
        pct_errors = np.abs((pred - true) / true) * 100

        breakdown[horizon] = {
            'mean_abs_error': round(np.mean(abs_errors), 2),
            'median_abs_error': round(np.median(abs_errors), 2),
            'max_abs_error': round(np.max(abs_errors), 2),
            'mean_pct_error': round(np.mean(pct_errors), 2),
            'median_pct_error': round(np.median(pct_errors), 2),
            'within_5pct': round(np.mean(pct_errors <= 5.0) * 100, 2),
            'within_10pct': round(np.mean(pct_errors <= 10.0) * 100, 2),
            'rmse': round(np.sqrt(np.mean((pred - true) ** 2)), 2)
        }

    return breakdown


def calculate_quarterly_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate quarterly aggregate metrics

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Quarterly metrics dictionary
    """
    # Sum revenue across 3 months for quarterly forecast
    quarterly_true = np.sum(y_true[:, :3], axis=1)
    quarterly_pred = np.sum(y_pred[:, :3], axis=1)

    # Quarterly MAPE
    quarterly_mape = np.mean(np.abs((quarterly_pred - quarterly_true) / quarterly_true)) * 100

    # Quarterly total error
    total_error = np.sum(np.abs(quarterly_pred - quarterly_true))

    # Quarterly accuracy (within 5%)
    quarterly_pct_errors = np.abs((quarterly_pred - quarterly_true) / quarterly_true) * 100
    quarterly_accuracy_5pct = np.mean(quarterly_pct_errors <= 5.0) * 100
    quarterly_accuracy_10pct = np.mean(quarterly_pct_errors <= 10.0) * 100

    return {
        'quarterly_mape': round(quarterly_mape, 2),
        'total_quarterly_error': round(total_error, 2),
        'quarterly_within_5pct': round(quarterly_accuracy_5pct, 2),
        'quarterly_within_10pct': round(quarterly_accuracy_10pct, 2)
    }


def generate_business_summary(metrics: Dict) -> str:
    """
    Generate human-readable summary of business metrics

    Args:
        metrics: Business metrics dictionary

    Returns:
        Formatted summary string
    """
    summary = []
    summary.append("\n" + "="*70)
    summary.append("BUSINESS METRICS SUMMARY")
    summary.append("="*70)

    summary.append("\n1. ACCURACY METRICS:")
    summary.append(f"   - Predictions within ±5% of actual:  {metrics['within_5_percent']:.1f}%")
    summary.append(f"   - Predictions within ±10% of actual: {metrics['within_10_percent']:.1f}%")
    summary.append(f"   - Large errors (>15%):               {metrics['large_error_rate']:.1f}%")

    summary.append("\n2. FORECAST ACCURACY:")
    summary.append(f"   - 3-Month cumulative forecast:       {metrics['forecast_accuracy']:.1f}%")
    if metrics['directional_accuracy'] is not None:
        summary.append(f"   - Directional accuracy:              {metrics['directional_accuracy']:.1f}%")
    if metrics['revenue_trend_consistency'] is not None:
        summary.append(f"   - Revenue trend consistency:         {metrics['revenue_trend_consistency']:.1f}%")

    summary.append("\n3. REVENUE PREDICTION BY HORIZON:")
    summary.append(f"   - Month 1 MAPE: {metrics['mape_month_1']:.2f}%")
    summary.append(f"   - Month 2 MAPE: {metrics['mape_month_2']:.2f}%")
    summary.append(f"   - Month 3 MAPE: {metrics['mape_month_3']:.2f}%")

    summary.append("\n4. OTHER PREDICTIONS:")
    summary.append(f"   - Member count MAPE:      {metrics['member_prediction_mape']:.2f}%")
    summary.append(f"   - Retention rate MAE:     {metrics['retention_prediction_mae']:.4f}")

    summary.append("\n5. BUSINESS IMPACT:")
    summary.append(f"   - Business Impact Score:  {metrics['business_impact_score']:.1f}/100")

    # Interpretation
    summary.append("\n6. INTERPRETATION:")
    if metrics['within_5_percent'] >= 70:
        summary.append("   [OK] EXCELLENT: >70% predictions within ±5%")
    elif metrics['within_5_percent'] >= 50:
        summary.append("   [OK] GOOD: 50-70% predictions within ±5%")
    else:
        summary.append("   [!] NEEDS IMPROVEMENT: <50% predictions within ±5%")

    if metrics['business_impact_score'] >= 80:
        summary.append("   [OK] HIGH business value - Low prediction errors")
    elif metrics['business_impact_score'] >= 60:
        summary.append("   [OK] MODERATE business value - Acceptable errors")
    else:
        summary.append("   [!] CAUTION - High prediction errors impacting business decisions")

    summary.append("\n" + "="*70 + "\n")

    return "\n".join(summary)


def save_business_metrics_report(metrics: Dict, breakdown: Dict, quarterly: Dict,
                                 output_path: str):
    """
    Save business metrics to structured CSV report

    Args:
        metrics: Main business metrics
        breakdown: Revenue breakdown by horizon
        quarterly: Quarterly metrics
        output_path: Path to save CSV
    """
    # Flatten all metrics into single dictionary
    report_data = {
        **{f'overall_{k}': v for k, v in metrics.items() if v is not None},
        **{f'quarterly_{k}': v for k, v in quarterly.items()},
    }

    # Add breakdown metrics
    for horizon, horizon_metrics in breakdown.items():
        for metric_name, value in horizon_metrics.items():
            report_data[f'{horizon}_{metric_name}'] = value

    # Convert to dataframe
    df = pd.DataFrame([report_data])
    df.to_csv(output_path, index=False)

    logger.info(f"Business metrics report saved to {output_path}")


if __name__ == "__main__":
    """Test business metrics calculation"""
    print("\nBusiness Metrics Calculator - Test Module")
    print("="*70)

    # Load test data
    try:
        import joblib
        from pathlib import Path

        df = pd.read_csv('data/processed/studio_data_engineered.csv')
        test_df = df[df['split'] == 'test']

        target_cols = [
            'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
            'member_count_month_3', 'retention_rate_month_3'
        ]

        y_true = test_df[target_cols].values

        # Load model and make predictions
        feature_names = joblib.load('data/models/features_v1.0.0.pkl')
        scaler = joblib.load('data/models/scaler_v1.0.0.pkl')
        models = joblib.load('data/models/ensemble_models_v1.0.0.pkl')
        weights = joblib.load('data/models/weights_v1.0.0.pkl')

        X = test_df[feature_names].values
        X_scaled = scaler.transform(X)

        # Make ensemble predictions
        predictions = []
        for model_name, model_list in models.items():
            if model_name == 'random_forest':
                pred = model_list.predict(X_scaled)
            else:
                pred = np.column_stack([m.predict(X_scaled) for m in model_list])
            predictions.append(pred * weights[model_name])

        y_pred = np.sum(predictions, axis=0)

        # Calculate metrics
        print("\nCalculating business metrics...")
        metrics = calculate_business_metrics(y_true, y_pred)
        breakdown = calculate_revenue_breakdown_metrics(y_true, y_pred)
        quarterly = calculate_quarterly_metrics(y_true, y_pred)

        # Print summary
        summary = generate_business_summary(metrics)
        print(summary)

        # Save report
        audit_dir = Path('reports/audit')
        audit_dir.mkdir(parents=True, exist_ok=True)
        save_business_metrics_report(
            metrics, breakdown, quarterly,
            audit_dir / 'business_metrics_v1.0.0.csv'
        )

        print(f"\n[OK] Business metrics report saved to {audit_dir / 'business_metrics_v1.0.0.csv'}")

    except Exception as e:
        print(f"\n[ERROR] Could not run test: {e}")
        print("This is expected if models haven't been trained yet.")
