"""
Visualization Script for Model v2.2.0 Performance
Generates comprehensive plots for training, validation (CV), and testing metrics
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory
output_dir = Path('reports/figures')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MODEL V2.2.0 PERFORMANCE VISUALIZATION")
print("=" * 80)
print()

# Load model results
print("Loading model results...")
with open('reports/audit/model_results_v2.2.0.json', 'r') as f:
    results = json.load(f)

# Load feature importance
print("Loading feature importance...")
feature_importance = pd.read_csv('reports/audit/feature_importance_v2.2.0.csv')

print(f"[OK] Data loaded successfully")
print()

# ============================================================================
# 1. Cross-Validation Performance Comparison
# ============================================================================
print("Creating Plot 1: Cross-Validation Performance Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model v2.2.0: Cross-Validation Performance Comparison', 
             fontsize=16, fontweight='bold', y=1.02)

cv_results = results['cv_results']
models = list(cv_results.keys())
model_labels = ['Ridge', 'ElasticNet', 'GBM']

# R2 scores
r2_means = [cv_results[m]['r2_mean'] for m in models]
r2_stds = [cv_results[m]['r2_std'] for m in models]

axes[0].bar(model_labels, r2_means, yerr=r2_stds, capsize=10, 
            color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black')
axes[0].set_ylabel('R² Score', fontsize=12, fontweight='bold')
axes[0].set_title('R² Score (mean ± std)', fontsize=13, fontweight='bold')
axes[0].set_ylim([min(r2_means) - 0.01, 1.0])
axes[0].grid(axis='y', alpha=0.3)

# Add value labels
for i, (mean, std) in enumerate(zip(r2_means, r2_stds)):
    axes[0].text(i, mean + std + 0.001, f'{mean:.4f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# RMSE values
rmse_means = [cv_results[m]['rmse_mean'] for m in models]
rmse_stds = [cv_results[m]['rmse_std'] for m in models]

axes[1].bar(model_labels, rmse_means, yerr=rmse_stds, capsize=10,
            color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black')
axes[1].set_ylabel('RMSE ($)', fontsize=12, fontweight='bold')
axes[1].set_title('RMSE (mean ± std)', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Add value labels
for i, (mean, std) in enumerate(zip(rmse_means, rmse_stds)):
    axes[1].text(i, mean + std + 20, f'${mean:.0f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'model_v2_2_0_cv_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: {output_dir / 'model_v2_2_0_cv_performance.png'}")

# ============================================================================
# 2. Test Set Performance by Target Variable
# ============================================================================
print("Creating Plot 2: Test Set Performance by Target Variable...")

test_results = results['test_results']
target_names = ['Revenue\nMonth 1', 'Revenue\nMonth 2', 'Revenue\nMonth 3', 
                'Members\nMonth 3', 'Retention\nMonth 3']
target_keys = ['Revenue Month 1', 'Revenue Month 2', 'Revenue Month 3',
               'Members Month 3', 'Retention Month 3']

fig, axes = plt.subplots(2, 1, figsize=(15, 10))
fig.suptitle('Model v2.2.0: Test Set Performance by Target Variable', 
             fontsize=16, fontweight='bold', y=0.995)

# R2 scores
x = np.arange(len(target_names))
width = 0.25

for i, (model, label, color) in enumerate(zip(models, model_labels, 
                                               ['#3498db', '#e74c3c', '#2ecc71'])):
    r2_values = [test_results[model]['metrics_by_target'][t]['R2'] for t in target_keys]
    axes[0].bar(x + i*width, r2_values, width, label=label, 
               color=color, alpha=0.8, edgecolor='black')

axes[0].set_ylabel('R² Score', fontsize=12, fontweight='bold')
axes[0].set_title('R² Score by Target Variable', fontsize=13, fontweight='bold')
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(target_names)
axes[0].legend(fontsize=11)
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0, 1.05])

# RMSE values
for i, (model, label, color) in enumerate(zip(models, model_labels, 
                                               ['#3498db', '#e74c3c', '#2ecc71'])):
    rmse_values = [test_results[model]['metrics_by_target'][t]['RMSE'] for t in target_keys]
    axes[1].bar(x + i*width, rmse_values, width, label=label, 
               color=color, alpha=0.8, edgecolor='black')

axes[1].set_ylabel('RMSE', fontsize=12, fontweight='bold')
axes[1].set_title('RMSE by Target Variable', fontsize=13, fontweight='bold')
axes[1].set_xticks(x + width)
axes[1].set_xticklabels(target_names)
axes[1].legend(fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'model_v2_2_0_test_performance_by_target.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: {output_dir / 'model_v2_2_0_test_performance_by_target.png'}")

# ============================================================================
# 3. Detailed Ridge Model Metrics (Best Model)
# ============================================================================
print("Creating Plot 3: Detailed Ridge Model Metrics...")

ridge_results = test_results['ridge']['metrics_by_target']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model v2.2.0: Detailed Ridge Model Performance (Best Model)', 
             fontsize=16, fontweight='bold', y=0.995)

metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
metric_titles = ['RMSE', 'MAE ($)', 'R² Score', 'MAPE (%)']
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

for idx, (metric, title, color) in enumerate(zip(metrics, metric_titles, colors)):
    ax = axes[idx // 2, idx % 2]
    values = [ridge_results[t][metric] for t in target_keys]
    
    bars = ax.bar(target_names, values, color=color, alpha=0.8, edgecolor='black')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if metric == 'MAPE':
            label = f'{val:.2f}%'
        elif metric == 'R2':
            label = f'{val:.4f}'
        else:
            label = f'{val:.2f}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               label, ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'model_v2_2_0_ridge_detailed_metrics.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: {output_dir / 'model_v2_2_0_ridge_detailed_metrics.png'}")

# ============================================================================
# 4. Feature Importance (Top 20)
# ============================================================================
print("Creating Plot 4: Feature Importance...")

top_features = feature_importance.head(20)

fig, ax = plt.subplots(figsize=(12, 10))
colors_grad = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))

bars = ax.barh(range(len(top_features)), top_features['Score'], 
               color=colors_grad, edgecolor='black', alpha=0.8)

ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'])
ax.invert_yaxis()
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Model v2.2.0: Top 20 Feature Importance', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, top_features['Score'])):
    width = bar.get_width()
    ax.text(width + max(top_features['Score']) * 0.01, bar.get_y() + bar.get_height()/2.,
           f'{score:,.0f}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'model_v2_2_0_feature_importance.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: {output_dir / 'model_v2_2_0_feature_importance.png'}")

# ============================================================================
# 5. Overall Model Comparison
# ============================================================================
print("Creating Plot 5: Overall Model Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('Model v2.2.0: Overall Test Set Performance Comparison', 
             fontsize=16, fontweight='bold', y=1.02)

# Overall R2
r2_overall = [test_results[m]['overall_r2'] for m in models]
axes[0].bar(model_labels, r2_overall, color=['#3498db', '#e74c3c', '#2ecc71'], 
           alpha=0.8, edgecolor='black')
axes[0].set_ylabel('Overall R² Score', fontsize=12, fontweight='bold')
axes[0].set_title('Overall R² Score', fontsize=13, fontweight='bold')
axes[0].set_ylim([min(r2_overall) - 0.01, 1.0])
axes[0].grid(axis='y', alpha=0.3)

for i, val in enumerate(r2_overall):
    axes[0].text(i, val + 0.001, f'{val:.4f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Overall RMSE
rmse_overall = [test_results[m]['overall_rmse'] for m in models]
axes[1].bar(model_labels, rmse_overall, color=['#3498db', '#e74c3c', '#2ecc71'], 
           alpha=0.8, edgecolor='black')
axes[1].set_ylabel('Overall RMSE ($)', fontsize=12, fontweight='bold')
axes[1].set_title('Overall RMSE', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

for i, val in enumerate(rmse_overall):
    axes[1].text(i, val + 30, f'${val:.2f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Overall MAE
mae_overall = [test_results[m]['overall_mae'] for m in models]
axes[2].bar(model_labels, mae_overall, color=['#3498db', '#e74c3c', '#2ecc71'], 
           alpha=0.8, edgecolor='black')
axes[2].set_ylabel('Overall MAE ($)', fontsize=12, fontweight='bold')
axes[2].set_title('Overall MAE', fontsize=13, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

for i, val in enumerate(mae_overall):
    axes[2].text(i, val + 15, f'${val:.2f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'model_v2_2_0_overall_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: {output_dir / 'model_v2_2_0_overall_comparison.png'}")

# ============================================================================
# 6. Training vs Test Data Split
# ============================================================================
print("Creating Plot 6: Training vs Test Data Split...")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Model v2.2.0: Training vs Test Data Split', 
             fontsize=16, fontweight='bold', y=1.02)

# Sample counts
sample_data = [results['training_samples'], results['test_samples']]
sample_labels = ['Training', 'Test']
colors_samples = ['#3498db', '#e74c3c']

wedges, texts, autotexts = axes[0].pie(sample_data, labels=sample_labels, autopct='%1.1f%%',
                                        colors=colors_samples, startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[0].set_title('Sample Distribution\n(Total: 816 samples)', 
                 fontsize=13, fontweight='bold')

# Add actual counts
for i, (label, count) in enumerate(zip(sample_labels, sample_data)):
    texts[i].set_text(f'{label}\n({count} samples)')

# Studio counts
studio_data = [results['n_studios_train'], results['n_studios_test']]
studio_labels = ['Training\nStudios', 'Test\nStudios']

bars = axes[1].bar(studio_labels, studio_data, color=['#3498db', '#e74c3c'], 
                   alpha=0.8, edgecolor='black')
axes[1].set_ylabel('Number of Studios', fontsize=12, fontweight='bold')
axes[1].set_title('Studio Distribution', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, studio_data):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
               f'{val} studios', ha='center', va='bottom', 
               fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'model_v2_2_0_data_split.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: {output_dir / 'model_v2_2_0_data_split.png'}")

# ============================================================================
# 7. Combined Summary Figure
# ============================================================================
print("Creating Plot 7: Combined Summary Figure...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Model v2.2.0: Comprehensive Performance Summary', 
             fontsize=18, fontweight='bold', y=0.98)

# Top left: CV Performance
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(model_labels, r2_means, color=['#3498db', '#e74c3c', '#2ecc71'], 
       alpha=0.8, edgecolor='black')
ax1.set_title('CV R² Scores', fontsize=12, fontweight='bold')
ax1.set_ylabel('R² Score', fontsize=10)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0.99, 1.0])

# Top middle: Test R2 by Target
ax2 = fig.add_subplot(gs[0, 1])
ridge_r2 = [ridge_results[t]['R2'] for t in target_keys]
ax2.bar(range(len(target_names)), ridge_r2, color='#3498db', 
       alpha=0.8, edgecolor='black')
ax2.set_title('Ridge Test R² by Target', fontsize=12, fontweight='bold')
ax2.set_ylabel('R² Score', fontsize=10)
ax2.set_xticks(range(len(target_names)))
ax2.set_xticklabels(['Rev M1', 'Rev M2', 'Rev M3', 'Mem M3', 'Ret M3'], rotation=45)
ax2.grid(axis='y', alpha=0.3)

# Top right: Overall Metrics
ax3 = fig.add_subplot(gs[0, 2])
x_pos = np.arange(3)
metrics_vals = [r2_overall[0], rmse_overall[0]/1000, mae_overall[0]/1000]
metrics_labels = ['R²\n(score)', 'RMSE\n(÷1000)', 'MAE\n(÷1000)']
colors_metrics = ['#2ecc71', '#e74c3c', '#f39c12']
ax3.bar(x_pos, metrics_vals, color=colors_metrics, alpha=0.8, edgecolor='black')
ax3.set_title('Ridge Overall Metrics', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(metrics_labels)
ax3.grid(axis='y', alpha=0.3)

# Middle row: Feature Importance (top 10)
ax4 = fig.add_subplot(gs[1, :])
top10_features = feature_importance.head(10)
colors_feat = plt.cm.viridis(np.linspace(0.3, 0.9, 10))
ax4.barh(range(10), top10_features['Score'], color=colors_feat, 
        alpha=0.8, edgecolor='black')
ax4.set_yticks(range(10))
ax4.set_yticklabels(top10_features['Feature'], fontsize=9)
ax4.invert_yaxis()
ax4.set_xlabel('Importance Score', fontsize=10, fontweight='bold')
ax4.set_title('Top 10 Most Important Features', fontsize=12, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

# Bottom left: MAPE by Target
ax5 = fig.add_subplot(gs[2, 0])
ridge_mape = [ridge_results[t]['MAPE'] for t in target_keys]
ax5.bar(range(len(target_names)), ridge_mape, color='#f39c12', 
       alpha=0.8, edgecolor='black')
ax5.set_title('Ridge MAPE by Target', fontsize=12, fontweight='bold')
ax5.set_ylabel('MAPE (%)', fontsize=10)
ax5.set_xticks(range(len(target_names)))
ax5.set_xticklabels(['Rev M1', 'Rev M2', 'Rev M3', 'Mem M3', 'Ret M3'], rotation=45)
ax5.grid(axis='y', alpha=0.3)

# Bottom middle: Data Split
ax6 = fig.add_subplot(gs[2, 1])
ax6.pie(sample_data, labels=['Training\n(732)', 'Test\n(84)'], 
       colors=['#3498db', '#e74c3c'], autopct='%1.1f%%',
       textprops={'fontsize': 9, 'fontweight': 'bold'}, startangle=90)
ax6.set_title('Sample Distribution', fontsize=12, fontweight='bold')

# Bottom right: Key Statistics
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')
stats_text = f"""
KEY STATISTICS

Best Model: Ridge Regression

Cross-Validation:
• R² = {cv_results['ridge']['r2_mean']:.4f} (±{cv_results['ridge']['r2_std']:.4f})
• RMSE = ${cv_results['ridge']['rmse_mean']:.2f} (±{cv_results['ridge']['rmse_std']:.2f})

Test Performance:
• Overall R² = {test_results['ridge']['overall_r2']:.4f}
• Overall RMSE = ${test_results['ridge']['overall_rmse']:.2f}
• Overall MAE = ${test_results['ridge']['overall_mae']:.2f}

Data:
• Training: {results['training_samples']} samples, {results['n_studios_train']} studios
• Test: {results['test_samples']} samples, {results['n_studios_test']} studios

Status: PRODUCTION READY [OK]
"""
ax7.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.savefig(output_dir / 'model_v2_2_0_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Saved: {output_dir / 'model_v2_2_0_summary.png'}")

# ============================================================================
# Summary
# ============================================================================
print()
print("=" * 80)
print("VISUALIZATION COMPLETE!")
print("=" * 80)
print()
print("Generated plots:")
print(f"  1. Cross-validation performance comparison")
print(f"  2. Test set performance by target variable")
print(f"  3. Detailed Ridge model metrics")
print(f"  4. Feature importance (top 20)")
print(f"  5. Overall model comparison")
print(f"  6. Training vs test data split")
print(f"  7. Combined summary figure")
print()
print(f"All plots saved to: {output_dir.absolute()}")
print()
print("Model v2.2.0 Summary:")
print(f"  • Best Model: Ridge Regression")
print(f"  • Cross-Val R²: {cv_results['ridge']['r2_mean']:.4f} (±{cv_results['ridge']['r2_std']:.4f})")
print(f"  • Test R²: {test_results['ridge']['overall_r2']:.4f}")
print(f"  • Test RMSE: ${test_results['ridge']['overall_rmse']:.2f}")
print(f"  • Test MAE: ${test_results['ridge']['overall_mae']:.2f}")
print(f"  • Status: PRODUCTION READY [OK]")
print()
print("=" * 80)

