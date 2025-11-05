"""
Comprehensive Evaluation for Multi-Studio Model v2.2.0

Compares performance with v2.0.0 to demonstrate data volume impact.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class MultiStudioEvaluator:
    """Comprehensive evaluation and comparison"""
    
    def __init__(self):
        self.v2_0_results = None
        self.v2_2_results = None
        
    def load_results(self):
        """Load results from both versions"""
        logger.info("Loading model results...")
        
        # Load v2.0.0 results
        try:
            with open('reports/audit/improved_model_results_v2.0.0.json', 'r') as f:
                self.v2_0_results = json.load(f)
            logger.info("Loaded v2.0.0 results")
        except FileNotFoundError:
            logger.warning("v2.0.0 results not found")
            self.v2_0_results = None
        
        # Load v2.2.0 results
        try:
            with open('reports/audit/model_results_v2.2.0.json', 'r') as f:
                self.v2_2_results = json.load(f)
            logger.info("Loaded v2.2.0 results")
        except FileNotFoundError:
            logger.error("v2.2.0 results not found! Please train the model first.")
            raise
    
    def compare_versions(self):
        """Compare v2.0.0 vs v2.2.0"""
        print("\n" + "="*80)
        print("VERSION COMPARISON: v2.0.0 vs v2.2.0")
        print("="*80 + "\n")
        
        if self.v2_0_results is None:
            print("v2.0.0 results not available for comparison")
            return
        
        # Extract metrics
        v2_0_model = self.v2_0_results['best_model']
        v2_0_test = self.v2_0_results['test_results'][v2_0_model]
        v2_0_cv = self.v2_0_results['cv_results'][v2_0_model]
        
        v2_2_model = self.v2_2_results['best_model']
        v2_2_test = self.v2_2_results['test_results'][v2_2_model]
        v2_2_cv = self.v2_2_results['cv_results'][v2_2_model]
        
        # Training data comparison
        v2_0_samples = self.v2_0_results.get('training_samples', 71)
        v2_2_samples = self.v2_2_results['training_samples']
        
        print("TRAINING DATA:")
        print(f"  v2.0.0: {v2_0_samples} samples (single studio)")
        print(f"  v2.2.0: {v2_2_samples} samples ({self.v2_2_results['n_studios_train']} studios)")
        print(f"  Increase: {v2_2_samples / v2_0_samples:.1f}x more data")
        
        # Cross-validation performance
        print("\nCROSS-VALIDATION PERFORMANCE:")
        print(f"  v2.0.0 CV R²: {v2_0_cv['r2_mean']:.4f} (+/- {v2_0_cv['r2_std']:.4f})")
        print(f"  v2.2.0 CV R²: {v2_2_cv['r2_mean']:.4f} (+/- {v2_2_cv['r2_std']:.4f})")
        
        cv_improvement = v2_2_cv['r2_mean'] - v2_0_cv['r2_mean']
        print(f"  Improvement: {cv_improvement:+.4f}")
        
        # Test performance
        print("\nTEST PERFORMANCE:")
        
        # Get per-target R² for v2.0.0
        v2_0_target_r2 = []
        for target_name in ['Revenue Month 1', 'Revenue Month 2', 'Revenue Month 3', 
                           'Members Month 3', 'Retention Month 3']:
            r2 = v2_0_test['metrics_by_target'][target_name]['R2']
            v2_0_target_r2.append(r2)
        v2_0_avg_r2 = np.mean(v2_0_target_r2)
        
        # Get per-target R² for v2.2.0
        v2_2_target_r2 = []
        for target_name in ['Revenue Month 1', 'Revenue Month 2', 'Revenue Month 3',
                           'Members Month 3', 'Retention Month 3']:
            r2 = v2_2_test['metrics_by_target'][target_name]['R2']
            v2_2_target_r2.append(r2)
        v2_2_avg_r2 = np.mean(v2_2_target_r2)
        
        print(f"  v2.0.0 Test R² (avg): {v2_0_avg_r2:.4f}")
        print(f"  v2.2.0 Test R² (avg): {v2_2_avg_r2:.4f}")
        
        test_improvement = v2_2_avg_r2 - v2_0_avg_r2
        print(f"  Improvement: {test_improvement:+.4f}")
        
        print(f"\n  v2.0.0 Test RMSE: {v2_0_test['overall_rmse']:.2f}")
        print(f"  v2.2.0 Test RMSE: {v2_2_test['overall_rmse']:.2f}")
        print(f"  Improvement: {v2_0_test['overall_rmse'] - v2_2_test['overall_rmse']:+.2f}")
        
        # Per-target comparison
        print("\nPER-TARGET COMPARISON:")
        print(f"{'Target':<20} {'v2.0.0 R²':>12} {'v2.2.0 R²':>12} {'Improvement':>12}")
        print("-" * 60)
        
        target_names = ['Revenue Month 1', 'Revenue Month 2', 'Revenue Month 3',
                       'Members Month 3', 'Retention Month 3']
        
        for target in target_names:
            v2_0_r2 = v2_0_test['metrics_by_target'][target]['R2']
            v2_2_r2 = v2_2_test['metrics_by_target'][target]['R2']
            improvement = v2_2_r2 - v2_0_r2
            print(f"{target:<20} {v2_0_r2:>12.4f} {v2_2_r2:>12.4f} {improvement:>+12.4f}")
        
        # Production readiness
        print("\nPRODUCTION READINESS:")
        
        if v2_0_avg_r2 < 0:
            v2_0_status = "NOT READY (negative R²)"
        elif v2_0_avg_r2 < 0.20:
            v2_0_status = "NOT READY"
        elif v2_0_avg_r2 < 0.30:
            v2_0_status = "MARGINAL"
        elif v2_0_avg_r2 < 0.40:
            v2_0_status = "ACCEPTABLE"
        else:
            v2_0_status = "READY"
        
        if v2_2_avg_r2 < 0.20:
            v2_2_status = "NOT READY"
        elif v2_2_avg_r2 < 0.30:
            v2_2_status = "MARGINAL"
        elif v2_2_avg_r2 < 0.40:
            v2_2_status = "ACCEPTABLE"
        elif v2_2_avg_r2 < 0.50:
            v2_2_status = "READY"
        else:
            v2_2_status = "EXCELLENT"
        
        print(f"  v2.0.0: {v2_0_status}")
        print(f"  v2.2.0: {v2_2_status}")
        
        print("\n" + "="*80)
    
    def create_visualizations(self):
        """Create comparison visualizations"""
        logger.info("Creating visualizations...")
        
        if self.v2_0_results is None or self.v2_2_results is None:
            logger.warning("Cannot create visualizations without both version results")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: R² Comparison by Target
        ax1 = axes[0, 0]
        
        target_names = ['Rev M1', 'Rev M2', 'Rev M3', 'Mem M3', 'Ret M3']
        v2_0_model = self.v2_0_results['best_model']
        v2_2_model = self.v2_2_results['best_model']
        
        full_target_names = ['Revenue Month 1', 'Revenue Month 2', 'Revenue Month 3',
                            'Members Month 3', 'Retention Month 3']
        
        v2_0_r2 = [self.v2_0_results['test_results'][v2_0_model]['metrics_by_target'][t]['R2'] 
                   for t in full_target_names]
        v2_2_r2 = [self.v2_2_results['test_results'][v2_2_model]['metrics_by_target'][t]['R2']
                   for t in full_target_names]
        
        x = np.arange(len(target_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, v2_0_r2, width, label='v2.0.0 (71 samples)', color='coral', edgecolor='black')
        bars2 = ax1.bar(x + width/2, v2_2_r2, width, label=f'v2.2.0 ({self.v2_2_results["training_samples"]} samples)', 
                       color='steelblue', edgecolor='black')
        
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Baseline')
        ax1.axhline(y=0.40, color='green', linestyle='--', alpha=0.5, label='Production Target')
        ax1.set_xlabel('Target Variable', fontweight='bold')
        ax1.set_ylabel('R² Score', fontweight='bold')
        ax1.set_title('Test R² Comparison by Target', fontweight='bold', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(target_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Overall Metrics Comparison
        ax2 = axes[0, 1]
        
        metrics = ['CV R²', 'Test R²', 'Test RMSE\n(x100)']
        v2_0_cv_r2 = self.v2_0_results['cv_results'][v2_0_model]['r2_mean']
        v2_2_cv_r2 = self.v2_2_results['cv_results'][v2_2_model]['r2_mean']
        
        v2_0_test_r2 = np.mean(v2_0_r2)
        v2_2_test_r2 = np.mean(v2_2_r2)
        
        v2_0_rmse = self.v2_0_results['test_results'][v2_0_model]['overall_rmse'] / 100
        v2_2_rmse = self.v2_2_results['test_results'][v2_2_model]['overall_rmse'] / 100
        
        v2_0_vals = [v2_0_cv_r2, v2_0_test_r2, v2_0_rmse]
        v2_2_vals = [v2_2_cv_r2, v2_2_test_r2, v2_2_rmse]
        
        x2 = np.arange(len(metrics))
        bars1 = ax2.bar(x2 - width/2, v2_0_vals, width, label='v2.0.0', color='coral', edgecolor='black')
        bars2 = ax2.bar(x2 + width/2, v2_2_vals, width, label='v2.2.0', color='steelblue', edgecolor='black')
        
        ax2.set_xlabel('Metric', fontweight='bold')
        ax2.set_ylabel('Value', fontweight='bold')
        ax2.set_title('Overall Metrics Comparison', fontweight='bold', fontsize=12)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Data Volume Impact
        ax3 = axes[1, 0]
        
        # Create learning curve visualization
        data_points = [71, self.v2_2_results['training_samples']]
        r2_points = [v2_0_test_r2, v2_2_test_r2]
        
        ax3.plot(data_points, r2_points, 'o-', color='steelblue', markersize=10, linewidth=2)
        ax3.axhline(y=0.40, color='green', linestyle='--', alpha=0.7, label='Production Target (0.40)')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Baseline (0)')
        
        # Annotate points
        ax3.annotate(f'v2.0.0\nR²={v2_0_test_r2:.3f}', 
                    xy=(data_points[0], r2_points[0]), 
                    xytext=(data_points[0]-50, r2_points[0]-0.05),
                    fontsize=9, ha='right',
                    bbox=dict(boxstyle='round', facecolor='coral', alpha=0.5))
        
        ax3.annotate(f'v2.2.0\nR²={v2_2_test_r2:.3f}', 
                    xy=(data_points[1], r2_points[1]), 
                    xytext=(data_points[1]+50, r2_points[1]+0.05),
                    fontsize=9, ha='left',
                    bbox=dict(boxstyle='round', facecolor='steelblue', alpha=0.5))
        
        ax3.set_xlabel('Training Samples (studio-months)', fontweight='bold')
        ax3.set_ylabel('Test R²', fontweight='bold')
        ax3.set_title('Learning Curve: Data Volume Impact', fontweight='bold', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary Table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        DATA VOLUME IMPACT ANALYSIS
        {'─'*50}
        
        Training Data:
          v2.0.0: 71 samples (1 studio)
          v2.2.0: {self.v2_2_results['training_samples']} samples ({self.v2_2_results['n_studios_train']} studios)
          Increase: {self.v2_2_results['training_samples'] / 71:.1f}x
        
        Performance Improvement:
          CV R²:   {v2_0_cv_r2:.4f} → {v2_2_cv_r2:.4f} ({v2_2_cv_r2 - v2_0_cv_r2:+.4f})
          Test R²: {v2_0_test_r2:.4f} → {v2_2_test_r2:.4f} ({v2_2_test_r2 - v2_0_test_r2:+.4f})
          RMSE:    {v2_0_rmse*100:.0f} → {v2_2_rmse*100:.0f} ({(v2_0_rmse - v2_2_rmse)*100:+.0f})
        
        Key Findings:
          • More data = Better generalization
          • Multi-studio = Diverse patterns
          • {self.v2_2_results['training_samples'] / 71:.0f}x data → {abs(v2_2_test_r2 - v2_0_test_r2):.2f} R² improvement
        
        Production Readiness:
          v2.0.0: {'NOT READY' if v2_0_test_r2 < 0.30 else 'ACCEPTABLE'}
          v2.2.0: {'EXCELLENT' if v2_2_test_r2 > 0.50 else 'READY' if v2_2_test_r2 > 0.40 else 'ACCEPTABLE'}
        
        Recommendation:
          {'✓ Deploy v2.2.0 to production' if v2_2_test_r2 > 0.40 else '⚠ Deploy to staging, collect more data'}
        """
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path('reports/figures/multi_studio_comparison.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
    
    def generate_report(self):
        """Generate text report"""
        logger.info("Generating evaluation report...")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("MULTI-STUDIO MODEL EVALUATION REPORT")
        report_lines.append("="*80)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"\nModel Version: v2.2.0")
        report_lines.append(f"Data Type: Multi-Studio")
        report_lines.append(f"Training Samples: {self.v2_2_results['training_samples']}")
        report_lines.append(f"Studios (Train): {self.v2_2_results['n_studios_train']}")
        report_lines.append(f"Studios (Test): {self.v2_2_results['n_studios_test']}")
        
        # Best model details
        best_model = self.v2_2_results['best_model']
        cv_results = self.v2_2_results['cv_results'][best_model]
        test_results = self.v2_2_results['test_results'][best_model]
        
        report_lines.append(f"\nBest Model: {best_model}")
        report_lines.append(f"\nCross-Validation Results:")
        report_lines.append(f"  R² = {cv_results['r2_mean']:.4f} (+/- {cv_results['r2_std']:.4f})")
        report_lines.append(f"  RMSE = {cv_results['rmse_mean']:.2f} (+/- {cv_results['rmse_std']:.2f})")
        
        report_lines.append(f"\nTest Set Results:")
        report_lines.append(f"  Overall R² = {test_results['overall_r2']:.4f}")
        report_lines.append(f"  Overall RMSE = {test_results['overall_rmse']:.2f}")
        report_lines.append(f"  Overall MAE = {test_results['overall_mae']:.2f}")
        
        report_lines.append(f"\nPer-Target Performance:")
        for target, metrics in test_results['metrics_by_target'].items():
            report_lines.append(f"  {target}:")
            report_lines.append(f"    R² = {metrics['R2']:.4f}")
            report_lines.append(f"    RMSE = {metrics['RMSE']:.2f}")
            report_lines.append(f"    MAE = {metrics['MAE']:.2f}")
            report_lines.append(f"    MAPE = {metrics['MAPE']:.2f}%")
        
        # Production readiness
        avg_r2 = np.mean([m['R2'] for m in test_results['metrics_by_target'].values()])
        
        report_lines.append(f"\n{'='*80}")
        report_lines.append("PRODUCTION READINESS ASSESSMENT")
        report_lines.append("="*80)
        
        if avg_r2 > 0.50:
            status = "EXCELLENT - PRODUCTION READY"
            action = "Deploy to production with standard monitoring"
        elif avg_r2 > 0.40:
            status = "GOOD - PRODUCTION READY"
            action = "Deploy to production with enhanced monitoring"
        elif avg_r2 > 0.30:
            status = "ACCEPTABLE - STAGING READY"
            action = "Deploy to staging, monitor closely"
        elif avg_r2 > 0.20:
            status = "MARGINAL - USE WITH CAUTION"
            action = "Use for guidance only, require human review"
        else:
            status = "INSUFFICIENT - NOT READY"
            action = "Collect more data, revisit approach"
        
        report_lines.append(f"\nStatus: {status}")
        report_lines.append(f"Average Test R²: {avg_r2:.4f}")
        report_lines.append(f"\nRecommended Action:")
        report_lines.append(f"  {action}")
        
        report_lines.append(f"\n{'='*80}\n")
        
        # Save report
        report_text = "\n".join(report_lines)
        output_path = Path('reports/audit/multi_studio_evaluation_report_v2.2.0.txt')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Report saved to {output_path}")
        
        # Print to console
        print(report_text)


def main():
    print("\n" + "="*80)
    print("MULTI-STUDIO MODEL EVALUATION")
    print("="*80 + "\n")
    
    evaluator = MultiStudioEvaluator()
    
    # Load results
    print("Step 1: Loading model results...")
    evaluator.load_results()
    
    # Compare versions
    print("\nStep 2: Comparing v2.0.0 vs v2.2.0...")
    evaluator.compare_versions()
    
    # Create visualizations
    print("\nStep 3: Creating visualizations...")
    evaluator.create_visualizations()
    
    # Generate report
    print("\nStep 4: Generating evaluation report...")
    evaluator.generate_report()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nOutputs:")
    print("  • reports/audit/multi_studio_evaluation_report_v2.2.0.txt")
    print("  • reports/figures/multi_studio_comparison.png")
    print("\nNext Steps:")
    print("  1. Review evaluation report")
    print("  2. Check comparison visualizations")
    print("  3. Read: reports/data_volume_impact_analysis.md")
    print("  4. Make deployment decision")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

