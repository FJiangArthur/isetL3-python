#!/usr/bin/env python3
"""Quick test script to verify the L3 ISP setup works correctly.

This script tests:
1. Basic imports and package structure
2. Synthetic data generation
3. Small-scale L3 training
4. Color space conversions
5. Basic pipeline functionality

Run this after installation to verify everything is working.
"""

import sys
import numpy as np
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all main components can be imported."""
    print("Testing imports...")
    
    try:
        from l3_isp import (
            L3Pipeline,
            L3Config,
            SyntheticDataGenerator,
            BT2020Converter,
            l3_root_path,
        )
        print("✓ Core imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_synthetic_data():
    """Test synthetic data generation."""
    print("\nTesting synthetic data generation...")
    
    try:
        from l3_isp import SyntheticDataGenerator
        
        # Create generator
        generator = SyntheticDataGenerator(
            image_size=(64, 64),
            bayer_pattern="RGGB",
            noise_std=0.01,
            num_scenes=10
        )
        
        # Generate a sample
        raw_bayer, target_rgb = generator.generate_sample(0)
        
        # Check dimensions
        assert raw_bayer.shape == (64, 64), f"Raw shape: {raw_bayer.shape}"
        assert target_rgb.shape == (64, 64, 3), f"RGB shape: {target_rgb.shape}"
        
        # Check value ranges
        assert 0 <= np.min(raw_bayer) <= np.max(raw_bayer) <= 1, f"Raw range: [{np.min(raw_bayer)}, {np.max(raw_bayer)}]"
        assert 0 <= np.min(target_rgb) <= np.max(target_rgb) <= 1, f"RGB range: [{np.min(target_rgb)}, {np.max(target_rgb)}]"
        
        print(f"✓ Generated sample: raw {raw_bayer.shape}, rgb {target_rgb.shape}")
        print(f"  Raw range: [{np.min(raw_bayer):.3f}, {np.max(raw_bayer):.3f}]")
        print(f"  RGB range: [{np.min(target_rgb):.3f}, {np.max(target_rgb):.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Synthetic data test failed: {e}")
        return False

def test_color_conversion():
    """Test color space conversion."""
    print("\nTesting color conversion...")
    
    try:
        from l3_isp import BT2020Converter
        
        converter = BT2020Converter()
        
        # Create test sRGB image
        srgb_image = np.random.rand(32, 32, 3) * 0.8  # Keep in reasonable range
        
        # Convert to BT.2020 and back
        bt2020_image = converter.srgb_to_linear_bt2020(srgb_image)
        recovered_srgb = converter.linear_bt2020_to_srgb(bt2020_image)
        
        # Check shapes
        assert bt2020_image.shape == srgb_image.shape
        assert recovered_srgb.shape == srgb_image.shape
        
        # Check that conversion is reasonably close (allowing for numerical precision)
        diff = np.mean(np.abs(srgb_image - recovered_srgb))
        assert diff < 0.01, f"Round-trip error too large: {diff}"
        
        print(f"✓ Color conversion successful")
        print(f"  Round-trip error: {diff:.6f}")
        print(f"  sRGB range: [{np.min(srgb_image):.3f}, {np.max(srgb_image):.3f}]")
        print(f"  BT.2020 range: [{np.min(bt2020_image):.3f}, {np.max(bt2020_image):.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Color conversion test failed: {e}")
        return False

def test_l3_training():
    """Test basic L3 training on synthetic data."""
    print("\nTesting L3 training...")
    
    try:
        from l3_isp import L3Pipeline, L3Config, SyntheticDataGenerator
        
        # Create minimal configuration
        config = L3Config(
            patch_size=(8, 8),
            num_cut_points=5,
            min_patches_per_class=10,
            training_method="OLS",
            output_color_space="BT.2020",
            use_infinite_isp=False,  # Disable for this test
            debug=False
        )
        
        # Initialize pipeline
        pipeline = L3Pipeline(config)
        
        # Generate small synthetic dataset
        generator = SyntheticDataGenerator(
            image_size=(32, 32),
            num_scenes=20
        )
        
        training_data = []
        for i in range(15):
            raw_bayer, target_rgb = generator.generate_sample(i)
            training_data.append((raw_bayer, target_rgb))
        
        validation_data = []
        for i in range(15, 20):
            raw_bayer, target_rgb = generator.generate_sample(i)
            validation_data.append((raw_bayer, target_rgb))
        
        # Train the pipeline
        print("  Training...")
        results = pipeline.train(training_data, validation_data)
        
        # Check results
        assert "num_kernels" in results
        assert results["num_kernels"] > 0
        
        if results["validation_results"]:
            val_psnr = results["validation_results"]["psnr"]
            print(f"  Validation PSNR: {val_psnr:.2f} dB")
        
        # Test processing a new image
        test_raw, test_target = validation_data[0]
        processed = pipeline.process_image(test_raw, use_trained_kernels=True)
        
        assert processed.shape == test_target.shape
        
        print(f"✓ L3 training successful")
        print(f"  Trained {results['num_kernels']} kernels")
        print(f"  Processed image shape: {processed.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ L3 training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_utilities():
    """Test utility functions."""
    print("\nTesting utilities...")
    
    try:
        from l3_isp.utils import (
            create_bayer_pattern,
            extract_patches,
            normalize_image,
            calculate_image_metrics,
            l3_root_path
        )
        
        # Test Bayer pattern creation
        rggb_pattern = create_bayer_pattern("RGGB")
        assert rggb_pattern.shape == (2, 2)
        assert np.array_equal(rggb_pattern, [[0, 1], [1, 2]])
        
        # Test patch extraction
        test_image = np.random.rand(64, 64)
        patches = extract_patches(test_image, (8, 8), stride=16, max_patches=10)
        assert len(patches) <= 10
        assert all(patch.shape == (8, 8) for patch in patches)
        
        # Test normalization
        test_image = np.random.rand(32, 32) * 2 + 0.5  # Range [0.5, 2.5]
        normalized, params = normalize_image(test_image, method="minmax")
        assert 0 <= np.min(normalized) <= np.max(normalized) <= 1
        
        # Test metrics calculation
        img1 = np.random.rand(16, 16, 3)
        img2 = img1 + np.random.rand(16, 16, 3) * 0.1  # Add some noise
        metrics = calculate_image_metrics(img1, img2)
        assert "mse" in metrics
        assert "psnr" in metrics
        assert metrics["mse"] >= 0
        assert metrics["psnr"] > 0
        
        # Test root path
        root = l3_root_path()
        assert Path(root).exists()
        
        print("✓ Utilities test successful")
        print(f"  RGGB pattern: {rggb_pattern.tolist()}")
        print(f"  Extracted {len(patches)} patches")
        print(f"  Test PSNR: {metrics['psnr']:.2f} dB")
        print(f"  Root path: {root}")
        
        return True
        
    except Exception as e:
        print(f"✗ Utilities test failed: {e}")
        return False

def test_configuration():
    """Test configuration management."""
    print("\nTesting configuration...")
    
    try:
        from l3_isp import L3Config
        import tempfile
        import os
        
        # Create config
        config = L3Config(
            patch_size=(64, 64),
            training_method="Ridge",
            ridge_alpha=0.5,
            output_color_space="BT.2020",
            debug=True
        )
        
        # Test serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save_yaml(f.name)
            temp_file = f.name
        
        # Test deserialization
        loaded_config = L3Config.from_yaml(temp_file)
        
        # Clean up
        os.unlink(temp_file)
        
        # Check values
        assert loaded_config.patch_size == (64, 64)
        assert loaded_config.training_method == "Ridge"
        assert loaded_config.ridge_alpha == 0.5
        assert loaded_config.output_color_space == "BT.2020"
        assert loaded_config.debug == True
        
        print("✓ Configuration test successful")
        print(f"  Patch size: {loaded_config.patch_size}")
        print(f"  Training method: {loaded_config.training_method}")
        print(f"  Output color space: {loaded_config.output_color_space}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("L3 ISP Setup Verification")
    print("=" * 30)
    
    tests = [
        test_imports,
        test_utilities,
        test_configuration,
        test_synthetic_data,
        test_color_conversion,
        test_l3_training,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 30)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! L3 ISP is ready to use.")
        print("\nNext steps:")
        print("1. Try the synthetic data demo:")
        print("   python examples/fivek_l3_training.py --synthetic_demo")
        print("\n2. Download FiveK dataset and run full training:")
        print("   python examples/fivek_l3_training.py --dataset_path /path/to/fivek")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
