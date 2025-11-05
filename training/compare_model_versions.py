"""
Model Version Comparison Script

Compare performance between different model versions to validate improvements.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')


class ModelVersionComparison:
    """Compare multiple model versions"""
    
    def __init__(self, version1='1.0.0', version2='2.0.0'):
        self.version1 = version1
        self.version2 = version2
        self.metrics_v1 = None
        self.metrics_v2 = None
        
    def load_metrics(self):
        """Load metrics from both versions"""
        logger.info(f"Loading metrics for versions {self.version1} and {self.version2}...")
        
        # Load v1 metrics
        try:
            with open(f'reports/audit/metrics_v{self.version1}.json', 'r') as f:
                self.metrics_v1 = json.load(f)
            logger.info(f"✓ Loaded v{self.version1} metrics")
        except FileNotFoundError:
            logger.error(f"✗ Could not find metrics for v{self.version1}")
            self.metrics_v1 = None
        
        # Load v2 metrics
        try:
            with open(f'reports/audit/improved_model_results_v{self.version2}.json', 'r') as f:
                self.metrics_v2 = json.load(f)
            logger.info(f"✓ Loaded v{self.version2} metrics")
        except FileNotFoundError:
            logger.error(f"✗ Could not find metrics for v{self.version2}")
            self.metrics_v2 = None
        
        if self.metrics_v1 is None or self.metrics_v2 is None:
            logger.error("Cannot proceed without both metric files")
            return False
        
        return True
    
    def compare_test_performance(self):
        """Compare test set performance"""
        logger.info("Comparing test set performance...")
        
        # Extract test metrics from v1
        v1_test = self.metrics_v1['metrics_by_dataset']['test']
        
        # Extract test metrics from v2 (from best model)
        best_model_v2 = self.metrics_v2['best_model']
        v2_test = self.metrics_v2['test_results'][best_model_v2]['metrics_by_target']
        
        comparison = {}
        
        # Compare revenue targets
        revenue_targets = {
            'Revenue Month 1': 'Revenue Month 1',
            'Revenue Month 2': 'Revenue Month 2',
            'Revenue Month 3': 'Revenue Month 3'
        }
        
        for target_name, target_key in revenue_targets.items():
            v1_metrics = v1_test[target_name]
            v2_metrics = v2_test[target_name]
            
            comparison[target_name] = {
                'v1_rmse': v1_metrics['RMSE'],
                'v2_rmse': v2_metrics['RMSE'],
                'rmse_improvement': (v1_metrics['RMSE'] - v2_metrics['RMSE']) / v1_metrics['RMSE'] * 100,
                'v1_r2': v1_metrics['R2'],
                'v2_r2': v2_metrics['R2'],
                'r2_improvement': v2_metrics['R2'] - v1_metrics['R2'],
                'v1_mape': v1_metrics['MAPE'],
                'v2_mape': v2_metrics['MAPE'],
                'mape_improvement': (v1_metrics['MAPE'] - v2_metrics['MAPE']) / v1_metrics['MAPE'] * 100
            }
        
        return comparison
    
    def compare_generalization(self):
        """Compare train/test gap (overfitting indicator)"""
        logger.info("Comparing generalization (overfitting check)...")
        
        # V1: Compare train vs test
        v1_train = self.metrics_v1['metrics_by_dataset']['train']
        v1_test = self.metrics_v1['metrics_by_dataset']['test']
        
        v1_train_rmse = np.mean([v1_train[t]['RMSE'] for t in ['Revenue Month 1', 'Revenue Month 2', 'Revenue Month 3']])
        v1_test_rmse = np.mean([v1_test[t]['RMSE'] for t in ['Revenue Month 1', 'Revenue Month 2', 'Revenue Month 3']])
        v1_gap = (v1_test_rmse - v1_train_rmse) / v1_train_rmse * 100
        
        # V2: Use CV std as generalization metric
        best_model_v2 = self.metrics_v2['best_model']
        v2_cv = self.metrics_v2['cv_results'][best_model_v2]
        v2_test = self.metrics_v2['test_results'][best_model_v2]
        
        v2_cv_rmse = v2_cv['rmse_mean']
        v2_test_rmse = v2_test['overall_rmse']
        v2_gap = (v2_test_rmse - v2_cv_rmse) / v2_cv_rmse * 100
        
        generalization = {
            'v1_train_rmse': v1_train_rmse,
            'v1_test_rmse': v1_test_rmse,
            'v1_train_test_gap_pct': v1_gap,
            'v2_cv_rmse': v2_cv_rmse,
            'v2_test_rmse': v2_test_rmse,
            'v2_cv_test_gap_pct': v2_gap,
            'gap_improvement': v1_gap - v2_gap
        }
        
        return generalization
    
    def visualize_comparison(self, comparison, generalization, save_path='reports/figures/version_comparison.png'):
        """Create comparison visualizations"""
        logger.info("Creating comparison visualizations...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. RMSE Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        targets = list(comparison.keys())
        v1_rmses = [comparison[t]['v1_rmse'] for t in targets]
        v2_rmses = [comparison[t]['v2_rmse'] for t in targets]
        
        x = np.arange(len(targets))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, v1_rmses, width, label=f'v{self.version1}', color='#ff7f0e')
        bars2 = ax1.bar(x + width/2, v2_rmses, width, label=f'v{self.version2}', color='#2ca02c')
        
        ax1.set_ylabel('RMSE', fontsize=11, fontweight='bold')
        ax1.set_title('RMSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Month 1', 'Month 2', 'Month 3'])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}',
                        ha='center', va='bottom', fontsize=9)
        
        # 2. R² Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        v1_r2s = [comparison[t]['v1_r2'] for t in targets]
        v2_r2s = [comparison[t]['v2_r2'] for t in targets]
        
        bars1 = ax2.bar(x - width/2, v1_r2s, width, label=f'v{self.version1}', color='#ff7f0e')
        bars2 = ax2.bar(x + width/2, v2_r2s, width, label=f'v{self.version2}', color='#2ca02c')
        
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Baseline (0)')
        ax2.set_ylabel('R² Score', fontsize=11, fontweight='bold')
        ax2.set_title('R² Comparison (Higher is Better)', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Month 1', 'Month 2', 'Month 3'])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        # 3. Improvements
        ax3 = fig.add_subplot(gs[1, 0])
        improvements = [comparison[t]['rmse_improvement'] for t in targets]
        colors = ['#2ca02c' if imp > 0 else '#d62728' for imp in improvements]
        
        bars = ax3.bar(range(len(targets)), improvements, color=colors)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_ylabel('RMSE Improvement (%)', fontsize=11, fontweight='bold')
        ax3.set_title(f'v{self.version2} vs v{self.version1} RMSE Improvement', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(targets)))
        ax3.set_xticklabels(['Month 1', 'Month 2', 'Month 3'])
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, improvements):
            ax3.text(bar.get_x() + bar.get_width()/2., val,
                    f'{val:+.1f}%',
                    ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
        
        # 4. Generalization (Overfitting) Comparison
        ax4 = fig.add_subplot(gs[1, 1])
        
        categories = [f'v{self.version1}\n(Train-Test Gap)', 
                     f'v{self.version2}\n(CV-Test Gap)']
        gaps = [generalization['v1_train_test_gap_pct'], 
                generalization['v2_cv_test_gap_pct']]
        colors = ['#ff7f0e' if gap > 50 else '#2ca02c' for gap in gaps]
        
        bars = ax4.bar(categories, gaps, color=colors)
        ax4.set_ylabel('Performance Gap (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Generalization Gap (Lower is Better)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, gaps):
            ax4.text(bar.get_x() + bar.get_width()/2., val,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to: {save_path}")
    
    def generate_report(self, comparison, generalization, 
                       output_path='reports/audit/version_comparison_report.txt'):
        """Generate detailed comparison report"""
        logger.info("Generating comparison report...")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MODEL VERSION COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Comparing: v{self.version1} vs v{self.version2}\n\n")
            
            f.write("1. TEST SET PERFORMANCE COMPARISON\n")
            f.write("-" * 80 + "\n\n")
            
            for target_name, metrics in comparison.items():
                f.write(f"{target_name}:\n")
                f.write(f"  RMSE:  v{self.version1}={metrics['v1_rmse']:.2f}  |  "
                       f"v{self.version2}={metrics['v2_rmse']:.2f}  |  "
                       f"Improvement: {metrics['rmse_improvement']:+.1f}%\n")
                f.write(f"  R²:    v{self.version1}={metrics['v1_r2']:.4f}  |  "
                       f"v{self.version2}={metrics['v2_r2']:.4f}  |  "
                       f"Change: {metrics['r2_improvement']:+.4f}\n")
                f.write(f"  MAPE:  v{self.version1}={metrics['v1_mape']:.2f}%  |  "
                       f"v{self.version2}={metrics['v2_mape']:.2f}%  |  "
                       f"Improvement: {metrics['mape_improvement']:+.1f}%\n\n")
            
            f.write("\n2. GENERALIZATION ANALYSIS\n")
            f.write("-" * 80 + "\n\n")
            
            f.write(f"v{self.version1} (Ensemble):\n")
            f.write(f"  Train RMSE: {generalization['v1_train_rmse']:.2f}\n")
            f.write(f"  Test RMSE:  {generalization['v1_test_rmse']:.2f}\n")
            f.write(f"  Gap:        {generalization['v1_train_test_gap_pct']:.1f}%\n")
            f.write(f"  Status:     {'⚠️  SEVERE OVERFITTING' if generalization['v1_train_test_gap_pct'] > 100 else '✓ Acceptable'}\n\n")
            
            f.write(f"v{self.version2} (Regularized):\n")
            f.write(f"  CV RMSE:    {generalization['v2_cv_rmse']:.2f}\n")
            f.write(f"  Test RMSE:  {generalization['v2_test_rmse']:.2f}\n")
            f.write(f"  Gap:        {generalization['v2_cv_test_gap_pct']:.1f}%\n")
            f.write(f"  Status:     {'✓ GOOD GENERALIZATION' if generalization['v2_cv_test_gap_pct'] < 50 else '⚠️  Some overfitting'}\n\n")
            
            f.write(f"Gap Improvement: {generalization['gap_improvement']:.1f}%\n\n")
            
            f.write("\n3. SUMMARY\n")
            f.write("-" * 80 + "\n\n")
            
            # Calculate average improvements
            avg_rmse_improvement = np.mean([comparison[t]['rmse_improvement'] for t in comparison])
            avg_r2_improvement = np.mean([comparison[t]['r2_improvement'] for t in comparison])
            
            f.write(f"Average RMSE Improvement: {avg_rmse_improvement:+.1f}%\n")
            f.write(f"Average R² Improvement:   {avg_r2_improvement:+.4f}\n")
            f.write(f"Generalization Improvement: {generalization['gap_improvement']:+.1f}%\n\n")
            
            # Recommendation
            if avg_rmse_improvement > 10 and generalization['v2_cv_test_gap_pct'] < 50:
                recommendation = "✅ RECOMMENDED: Deploy v" + self.version2
                reason = "Significant performance improvement with good generalization"
            elif avg_r2_improvement > 0.2:
                recommendation = "✅ RECOMMENDED: Deploy v" + self.version2
                reason = "Better predictive power (R² improvement)"
            elif generalization['v2_cv_test_gap_pct'] < generalization['v1_train_test_gap_pct']:
                recommendation = "⚠️  CONDITIONAL: Consider v" + self.version2
                reason = "Better generalization, but monitor performance"
            else:
                recommendation = "❌ NOT RECOMMENDED: Keep v" + self.version1
                reason = "No significant improvement"
            
            f.write("4. RECOMMENDATION\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"{recommendation}\n")
            f.write(f"Reason: {reason}\n\n")
            
            f.write("="*80 + "\n")
        
        logger.info(f"Report saved to: {output_path}")
        
        return recommendation


def main():
    print("\n" + "="*80)
    print("MODEL VERSION COMPARISON")
    print("="*80 + "\n")
    
    # Initialize comparison
    comparator = ModelVersionComparison(version1='1.0.0', version2='2.0.0')
    
    # Load metrics
    if not comparator.load_metrics():
        print("\n❌ Error: Could not load metrics for both versions")
        print("\nMake sure you have:")
        print("  1. Trained the original model (v1.0.0)")
        print("  2. Trained the improved model (v2.0.0)")
        print("  3. Both metric files exist in reports/audit/")
        return
    
    # Compare performance
    print("\nComparing test set performance...")
    comparison = comparator.compare_test_performance()
    
    print("\nComparing generalization...")
    generalization = comparator.compare_generalization()
    
    # Visualize
    print("\nCreating visualizations...")
    comparator.visualize_comparison(comparison, generalization)
    
    # Generate report
    print("\nGenerating detailed report...")
    recommendation = comparator.generate_report(comparison, generalization)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    for target, metrics in comparison.items():
        print(f"\n{target}:")
        print(f"  RMSE improvement: {metrics['rmse_improvement']:+.1f}%")
        print(f"  R² change: {metrics['r2_improvement']:+.4f}")
    
    print(f"\nGeneralization:")
    print(f"  v1.0.0 gap: {generalization['v1_train_test_gap_pct']:.1f}%")
    print(f"  v2.0.0 gap: {generalization['v2_cv_test_gap_pct']:.1f}%")
    print(f"  Improvement: {generalization['gap_improvement']:+.1f}%")
    
    print(f"\n{recommendation}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

