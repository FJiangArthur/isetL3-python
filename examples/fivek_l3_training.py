#!/usr/bin/env python3
"""MIT-Adobe FiveK L3 Training Example.

This script demonstrates how to:
1. Download and preprocess the MIT-Adobe FiveK dataset
2. Train L3 kernels on Bayer → RGB pairs
3. Apply trained filters to new images
4. Evaluate performance with BT.2020 color space output

Usage:
    python examples/fivek_l3_training.py --dataset_path /path/to/fivek --max_images 100
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from l3_isp import (
    L3Pipeline,
    L3Config,
    FiveKDataset,
    L3Classifier,
    L3TrainOLS,
    BT2020Converter,
)
from l3_isp.utils import (
    calculate_image_metrics,
    save_image_with_metadata,
    create_synthetic_bayer,
)


def create_config(args) -> L3Config:
    """Create L3 configuration from arguments."""
    return L3Config(
        # Core L3 parameters
        patch_size=(args.patch_size, args.patch_size),
        num_cut_points=args.num_cut_points,
        min_patches_per_class=args.min_patches,
        
        # Training settings
        training_method=args.training_method,
        ridge_alpha=args.ridge_alpha,
        
        # Color space
        output_color_space="BT.2020",
        linear_output=True,
        
        # InfiniteISP integration
        use_infinite_isp=args.use_infinite_isp,
        
        # Output settings
        debug=args.debug,
        save_intermediate=args.save_intermediate,
        output_dir=args.output_dir,
        
        # Dataset
        dataset_type="fivek",
        dataset_path=args.dataset_path,
    )


def load_dataset(args) -> tuple:
    """Load training and validation datasets."""
    print("Loading FiveK dataset...")
    
    # Training set
    train_dataset = FiveKDataset(
        dataset_path=args.dataset_path,
        subset="train",
        download=args.download,
        max_images=args.max_images,
        target_resolution=(args.image_width, args.image_height) if args.image_width > 0 else None,
        expert_choice=args.expert_choice,
    )
    
    # Validation set
    val_dataset = FiveKDataset(
        dataset_path=args.dataset_path,
        subset="val",
        download=args.download,
        max_images=args.max_val_images,
        target_resolution=(args.image_width, args.image_height) if args.image_width > 0 else None,
        expert_choice=args.expert_choice,
    )
    
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")
    
    return train_dataset, val_dataset


def create_training_data(dataset, args) -> list:
    """Create training data from dataset."""
    print("Creating training data...")
    
    training_data = []
    
    for i in tqdm(range(len(dataset))):
        try:
            raw_bayer, target_rgb, metadata = dataset[i]
            
            # Add to training data
            training_data.append((raw_bayer, target_rgb))
            
            if args.debug and i < 3:
                # Save sample images for debugging
                output_dir = Path(args.output_dir) / "debug_samples"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                save_image_with_metadata(
                    raw_bayer,
                    str(output_dir / f"sample_{i}_raw.png"),
                    metadata
                )
                save_image_with_metadata(
                    target_rgb,
                    str(output_dir / f"sample_{i}_target.png"),
                    metadata
                )
        
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue
    
    print(f"Created {len(training_data)} training samples")
    return training_data


def train_l3_pipeline(pipeline: L3Pipeline, training_data: list, validation_data: list) -> dict:
    """Train the L3 pipeline."""
    print("Training L3 pipeline...")
    
    # Train the pipeline
    results = pipeline.train(training_data, validation_data)
    
    print(f"Training completed with {results['num_kernels']} kernels")
    
    if results['validation_results']:
        val_results = results['validation_results']
        print(f"Validation PSNR: {val_results['psnr']:.2f} dB")
        print(f"Validation MSE: {val_results['mse']:.6f}")
    
    return results


def test_pipeline(pipeline: L3Pipeline, test_data: list, args) -> dict:
    """Test the trained pipeline on test data."""
    print("Testing pipeline...")
    
    output_dir = Path(args.output_dir) / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    
    for i, (raw_bayer, target_rgb) in enumerate(tqdm(test_data[:args.max_test_images])):
        try:
            # Process with L3
            processed_rgb = pipeline.process_image(raw_bayer, use_trained_kernels=True)
            
            # Calculate metrics
            metrics = calculate_image_metrics(processed_rgb, target_rgb)
            all_metrics.append(metrics)
            
            # Save results for first few images
            if i < args.save_first_n:
                # Save raw input
                save_image_with_metadata(
                    raw_bayer,
                    str(output_dir / f"test_{i:03d}_raw.png"),
                    {"type": "raw_bayer"}
                )
                
                # Save target
                save_image_with_metadata(
                    target_rgb,
                    str(output_dir / f"test_{i:03d}_target.png"),
                    {"type": "target_bt2020"}
                )
                
                # Save L3 result
                save_image_with_metadata(
                    processed_rgb,
                    str(output_dir / f"test_{i:03d}_l3.png"),
                    {"type": "l3_processed", "metrics": metrics}
                )
                
                # Create comparison plot
                create_comparison_plot(
                    raw_bayer, target_rgb, processed_rgb, 
                    str(output_dir / f"test_{i:03d}_comparison.png"),
                    metrics
                )
        
        except Exception as e:
            print(f"Error processing test image {i}: {e}")
            continue
    
    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[f"avg_{key}"] = np.mean([m[key] for m in all_metrics])
        avg_metrics[f"std_{key}"] = np.std([m[key] for m in all_metrics])
    
    print(f"\nTest Results ({len(all_metrics)} images):")
    print(f"Average PSNR: {avg_metrics['avg_psnr']:.2f} ± {avg_metrics['std_psnr']:.2f} dB")
    print(f"Average MSE: {avg_metrics['avg_mse']:.6f} ± {avg_metrics['std_mse']:.6f}")
    print(f"Average SSIM: {avg_metrics['avg_ssim']:.4f} ± {avg_metrics['std_ssim']:.4f}")
    
    return avg_metrics


def create_comparison_plot(
    raw_bayer: np.ndarray,
    target_rgb: np.ndarray,
    processed_rgb: np.ndarray,
    output_path: str,
    metrics: dict
) -> None:
    """Create comparison plot of raw, target, and processed images."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Raw Bayer (show as grayscale)
    axes[0].imshow(raw_bayer, cmap='gray')
    axes[0].set_title('Raw Bayer')
    axes[0].axis('off')
    
    # Target RGB
    target_display = np.clip(target_rgb, 0, 1)
    axes[1].imshow(target_display)
    axes[1].set_title('Target (BT.2020)')
    axes[1].axis('off')
    
    # Processed RGB
    processed_display = np.clip(processed_rgb, 0, 1)
    axes[2].imshow(processed_display)
    axes[2].set_title(f'L3 Processed\nPSNR: {metrics["psnr"]:.1f} dB')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def demonstrate_synthetic_data(args) -> None:
    """Demonstrate synthetic data generation."""
    print("\nDemonstrating synthetic data generation...")
    
    from l3_isp.data import SyntheticDataGenerator
    
    # Create synthetic data generator
    generator = SyntheticDataGenerator(
        image_size=(256, 256),
        bayer_pattern="RGGB",
        noise_std=0.02,
    )
    
    # Generate some synthetic data
    synthetic_data = []
    for i in range(10):
        raw_bayer, target_rgb = generator.generate_sample()
        synthetic_data.append((raw_bayer, target_rgb))
    
    # Train a small L3 pipeline on synthetic data
    config = L3Config(
        patch_size=(16, 16),
        num_cut_points=10,
        min_patches_per_class=20,
        training_method="OLS",
        output_color_space="BT.2020",
        debug=True,
        output_dir=args.output_dir,
    )
    
    synthetic_pipeline = L3Pipeline(config)
    synthetic_results = synthetic_pipeline.train(synthetic_data[:8], synthetic_data[8:])
    
    print(f"Synthetic training completed with {synthetic_results['num_kernels']} kernels")
    
    # Save synthetic pipeline
    synthetic_pipeline.save_model(str(Path(args.output_dir) / "synthetic_l3_model.npz"))


def main():
    """Main training and evaluation script."""
    parser = argparse.ArgumentParser(description="FiveK L3 Training")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, default="/tmp/fivek_demo",
                       help="Path to FiveK dataset")
    parser.add_argument("--download", action="store_true",
                       help="Download dataset if not present")
    parser.add_argument("--max_images", type=int, default=100,
                       help="Maximum training images")
    parser.add_argument("--max_val_images", type=int, default=20,
                       help="Maximum validation images")
    parser.add_argument("--max_test_images", type=int, default=10,
                       help="Maximum test images")
    parser.add_argument("--expert_choice", type=str, default="C", choices=["A", "B", "C", "D", "E"],
                       help="Expert retouching choice")
    
    # Image processing arguments
    parser.add_argument("--image_width", type=int, default=512,
                       help="Target image width (0 for original)")
    parser.add_argument("--image_height", type=int, default=512,
                       help="Target image height (0 for original)")
    
    # L3 training arguments
    parser.add_argument("--patch_size", type=int, default=32,
                       help="Patch size for L3 training")
    parser.add_argument("--num_cut_points", type=int, default=30,
                       help="Number of cut points for classification")
    parser.add_argument("--min_patches", type=int, default=50,
                       help="Minimum patches per class")
    parser.add_argument("--training_method", type=str, default="OLS", choices=["OLS", "Ridge"],
                       help="Training method")
    parser.add_argument("--ridge_alpha", type=float, default=1.0,
                       help="Ridge regression alpha")
    
    # Pipeline arguments
    parser.add_argument("--use_infinite_isp", action="store_true",
                       help="Use InfiniteISP preprocessing")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="output/fivek_l3",
                       help="Output directory")
    parser.add_argument("--save_first_n", type=int, default=5,
                       help="Save first N test results")
    parser.add_argument("--save_model", action="store_true",
                       help="Save trained model")
    
    # Debug arguments
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--save_intermediate", action="store_true",
                       help="Save intermediate results")
    parser.add_argument("--synthetic_demo", action="store_true",
                       help="Run synthetic data demonstration")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("FiveK L3 Training and Evaluation")
    print("=" * 40)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max training images: {args.max_images}")
    print(f"Expert choice: {args.expert_choice}")
    
    try:
        # Create configuration
        config = create_config(args)
        
        # Initialize pipeline
        pipeline = L3Pipeline(config)
        
        # Load datasets
        train_dataset, val_dataset = load_dataset(args)
        
        if len(train_dataset) == 0:
            print("\nNo training data found. Please check dataset path and download settings.")
            print("\nTo manually download FiveK dataset:")
            print("1. Visit: https://data.csail.mit.edu/graphics/fivek/")
            print("2. Download raw and retouched images")
            print("3. Extract to the specified dataset path")
            
            if args.synthetic_demo:
                demonstrate_synthetic_data(args)
            return
        
        # Create training data
        training_data = create_training_data(train_dataset, args)
        validation_data = create_training_data(val_dataset, args)
        
        if len(training_data) == 0:
            print("No valid training data created. Exiting.")
            return
        
        # Train pipeline
        training_results = train_l3_pipeline(pipeline, training_data, validation_data)
        
        # Test pipeline
        if len(validation_data) > 0:
            test_results = test_pipeline(pipeline, validation_data, args)
        
        # Save model if requested
        if args.save_model:
            model_path = Path(args.output_dir) / "fivek_l3_model.npz"
            pipeline.save_model(str(model_path))
            print(f"\nModel saved to: {model_path}")
        
        # Save configuration
        config_path = Path(args.output_dir) / "config.yaml"
        config.save_yaml(str(config_path))
        print(f"Configuration saved to: {config_path}")
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    # Run synthetic demo if requested
    if args.synthetic_demo:
        demonstrate_synthetic_data(args)


if __name__ == "__main__":
    main()
