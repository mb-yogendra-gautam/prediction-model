"""
Data Augmentation for Small Datasets

Generates synthetic training samples to increase dataset size
while preserving statistical properties.
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample
from scipy.stats import norm
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAugmenter:
    """Generate synthetic samples for small datasets"""
    
    def __init__(self, noise_level=0.02, random_state=42):
        """
        Args:
            noise_level: Standard deviation of noise as fraction of feature std
            random_state: Random seed for reproducibility
        """
        self.noise_level = noise_level
        self.random_state = random_state
        np.random.seed(random_state)
        
    def add_noise(self, X, y):
        """Add small Gaussian noise to features and targets"""
        logger.info(f"Adding Gaussian noise (level={self.noise_level})...")
        
        # Calculate noise for features
        X_noise = np.random.normal(0, self.noise_level * X.std(axis=0), X.shape)
        X_augmented = X + X_noise
        
        # Calculate noise for targets
        y_noise = np.random.normal(0, self.noise_level * y.std(axis=0), y.shape)
        y_augmented = y + y_noise
        
        # Ensure no negative values for strictly positive features
        X_augmented = np.maximum(X_augmented, 0)
        y_augmented = np.maximum(y_augmented, 0)
        
        return X_augmented, y_augmented
    
    def bootstrap_resample(self, X, y, n_samples):
        """Bootstrap resampling with replacement"""
        logger.info(f"Bootstrap resampling {n_samples} samples...")
        
        indices = np.random.choice(len(X), size=n_samples, replace=True)
        X_resampled = X[indices]
        y_resampled = y[indices]
        
        return X_resampled, y_resampled
    
    def interpolate_samples(self, X, y, n_samples):
        """Create synthetic samples by interpolating between existing ones"""
        logger.info(f"Interpolating {n_samples} synthetic samples...")
        
        X_synthetic = []
        y_synthetic = []
        
        for _ in range(n_samples):
            # Randomly select two samples
            idx1, idx2 = np.random.choice(len(X), size=2, replace=False)
            
            # Random interpolation weight
            alpha = np.random.uniform(0.3, 0.7)
            
            # Interpolate
            X_new = alpha * X[idx1] + (1 - alpha) * X[idx2]
            y_new = alpha * y[idx1] + (1 - alpha) * y[idx2]
            
            X_synthetic.append(X_new)
            y_synthetic.append(y_new)
        
        return np.array(X_synthetic), np.array(y_synthetic)
    
    def mixup(self, X, y, alpha=0.4):
        """
        Mixup data augmentation
        
        Zhang et al., 2018 - mixup: Beyond Empirical Risk Minimization
        """
        logger.info(f"Applying mixup augmentation (alpha={alpha})...")
        
        n_samples = len(X)
        X_mixed = []
        y_mixed = []
        
        for i in range(n_samples):
            # Random sample index
            j = np.random.randint(n_samples)
            
            # Random lambda from Beta distribution
            lam = np.random.beta(alpha, alpha)
            
            # Mix
            X_new = lam * X[i] + (1 - lam) * X[j]
            y_new = lam * y[i] + (1 - lam) * y[j]
            
            X_mixed.append(X_new)
            y_mixed.append(y_new)
        
        return np.array(X_mixed), np.array(y_mixed)
    
    def augment_training_data(self, X_train, y_train, augmentation_factor=1.5, 
                             methods=['noise', 'bootstrap', 'interpolate']):
        """
        Apply multiple augmentation techniques
        
        Args:
            X_train: Training features
            y_train: Training targets
            augmentation_factor: Multiply dataset size by this factor
            methods: List of augmentation methods to apply
            
        Returns:
            X_augmented, y_augmented: Augmented dataset
        """
        logger.info(f"Augmenting dataset by factor of {augmentation_factor}...")
        logger.info(f"Original size: {len(X_train)} samples")
        
        # Calculate target size
        n_original = len(X_train)
        n_target = int(n_original * augmentation_factor)
        n_synthetic = n_target - n_original
        
        # Distribute synthetic samples across methods
        n_per_method = n_synthetic // len(methods)
        
        X_augmented = [X_train]
        y_augmented = [y_train]
        
        # Apply each method
        for method in methods:
            if method == 'noise':
                # Bootstrap + noise
                X_resampled, y_resampled = self.bootstrap_resample(
                    X_train, y_train, n_per_method
                )
                X_noisy, y_noisy = self.add_noise(X_resampled, y_resampled)
                X_augmented.append(X_noisy)
                y_augmented.append(y_noisy)
                
            elif method == 'bootstrap':
                # Pure bootstrap
                X_boot, y_boot = self.bootstrap_resample(
                    X_train, y_train, n_per_method
                )
                X_augmented.append(X_boot)
                y_augmented.append(y_boot)
                
            elif method == 'interpolate':
                # Interpolation
                X_interp, y_interp = self.interpolate_samples(
                    X_train, y_train, n_per_method
                )
                X_augmented.append(X_interp)
                y_augmented.append(y_interp)
                
            elif method == 'mixup':
                # Mixup
                X_mix, y_mix = self.mixup(X_train, y_train)
                # Take subset
                indices = np.random.choice(len(X_mix), n_per_method, replace=False)
                X_augmented.append(X_mix[indices])
                y_augmented.append(y_mix[indices])
        
        # Combine all
        X_final = np.vstack(X_augmented)
        y_final = np.vstack(y_augmented)
        
        logger.info(f"Augmented size: {len(X_final)} samples")
        logger.info(f"Increase: {len(X_final) - n_original} synthetic samples")
        
        return X_final, y_final
    
    def validate_augmentation(self, X_original, X_augmented):
        """Check if augmented data maintains statistical properties"""
        logger.info("Validating augmented data...")
        
        # Compare statistics
        stats_comparison = pd.DataFrame({
            'Original_Mean': X_original.mean(axis=0),
            'Augmented_Mean': X_augmented.mean(axis=0),
            'Original_Std': X_original.std(axis=0),
            'Augmented_Std': X_augmented.std(axis=0)
        })
        
        # Calculate differences
        stats_comparison['Mean_Diff_%'] = (
            abs(stats_comparison['Original_Mean'] - stats_comparison['Augmented_Mean']) / 
            (stats_comparison['Original_Mean'] + 1e-10) * 100
        )
        stats_comparison['Std_Diff_%'] = (
            abs(stats_comparison['Original_Std'] - stats_comparison['Augmented_Std']) / 
            (stats_comparison['Original_Std'] + 1e-10) * 100
        )
        
        # Check if differences are acceptable (< 10%)
        mean_diffs = stats_comparison['Mean_Diff_%']
        std_diffs = stats_comparison['Std_Diff_%']
        
        logger.info(f"Mean difference: {mean_diffs.mean():.2f}% (max: {mean_diffs.max():.2f}%)")
        logger.info(f"Std difference: {std_diffs.mean():.2f}% (max: {std_diffs.max():.2f}%)")
        
        if mean_diffs.mean() < 10 and std_diffs.mean() < 15:
            logger.info("✓ Augmented data maintains statistical properties")
        else:
            logger.warning("⚠️  Augmented data shows significant distribution shift")
        
        return stats_comparison


def augment_dataset(input_path='data/processed/studio_data_engineered.csv',
                   output_path='data/processed/studio_data_augmented.csv',
                   augmentation_factor=1.5,
                   noise_level=0.02):
    """
    Main function to augment the dataset
    
    Args:
        input_path: Path to original data
        output_path: Path to save augmented data
        augmentation_factor: Multiply training data by this factor
        noise_level: Noise level for augmentation
    """
    logger.info("Loading data...")
    df = pd.read_csv(input_path)
    
    # Separate train/val/test
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'validation'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    logger.info(f"Original train size: {len(train_df)}")
    
    # Define features and targets
    target_cols = [
        'revenue_month_1', 'revenue_month_2', 'revenue_month_3',
        'member_count_month_3', 'retention_rate_month_3'
    ]
    exclude_cols = target_cols + ['month_year', 'split', 'year_index']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Extract train data
    X_train = train_df[feature_cols].values
    y_train = train_df[target_cols].values
    
    # Initialize augmenter
    augmenter = DataAugmenter(noise_level=noise_level)
    
    # Augment training data only (never augment validation/test!)
    X_augmented, y_augmented = augmenter.augment_training_data(
        X_train, y_train,
        augmentation_factor=augmentation_factor,
        methods=['noise', 'interpolate']  # Conservative methods
    )
    
    # Validate
    stats = augmenter.validate_augmentation(X_train, X_augmented)
    
    # Create augmented dataframe
    augmented_train_df = pd.DataFrame(X_augmented, columns=feature_cols)
    
    # Add targets
    for i, col in enumerate(target_cols):
        augmented_train_df[col] = y_augmented[:, i]
    
    # Add metadata (use first date, mark as augmented)
    augmented_train_df['month_year'] = train_df['month_year'].iloc[0]
    augmented_train_df['split'] = 'train'
    augmented_train_df['year_index'] = 0
    
    # Combine with validation and test (unchanged)
    final_df = pd.concat([augmented_train_df, val_df, test_df], ignore_index=True)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    logger.info(f"Augmented dataset saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("DATA AUGMENTATION SUMMARY")
    print("="*80)
    print(f"\nOriginal train size: {len(train_df)}")
    print(f"Augmented train size: {len(augmented_train_df)}")
    print(f"Increase: {len(augmented_train_df) - len(train_df)} samples")
    print(f"\nValidation size: {len(val_df)} (unchanged)")
    print(f"Test size: {len(test_df)} (unchanged)")
    print(f"\nTotal dataset size: {len(final_df)}")
    print(f"\nSaved to: {output_path}")
    print("="*80 + "\n")
    
    print("⚠️  IMPORTANT:")
    print("  - Only training data was augmented")
    print("  - Validation and test data remain original")
    print("  - Use this augmented data for model training")
    print("  - Monitor for overfitting with cross-validation")
    print("  - Compare results with original data")
    
    return final_df, stats


def main():
    print("\n" + "="*80)
    print("DATA AUGMENTATION - Increase Training Dataset Size")
    print("="*80 + "\n")
    
    print("Configuration:")
    print("  Augmentation factor: 1.5x (increase by 50%)")
    print("  Noise level: 2% of feature standard deviation")
    print("  Methods: Noise injection + Interpolation")
    print("\n" + "-"*80 + "\n")
    
    # Run augmentation
    final_df, stats = augment_dataset(
        input_path='data/processed/studio_data_engineered.csv',
        output_path='data/processed/studio_data_augmented.csv',
        augmentation_factor=1.5,
        noise_level=0.02
    )
    
    # Save statistics
    stats_path = 'reports/audit/augmentation_statistics.csv'
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(stats_path)
    print(f"\n✓ Augmentation statistics saved to: {stats_path}")
    
    print("\nNext Steps:")
    print("  1. Review augmentation statistics")
    print("  2. Train model using: data/processed/studio_data_augmented.csv")
    print("  3. Compare performance with original dataset")
    print("  4. Use cross-validation to detect overfitting")


if __name__ == "__main__":
    main()

