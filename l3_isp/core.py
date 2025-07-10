"""Core L3 ISP pipeline implementation."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import yaml
from dataclasses import dataclass, field

# Add InfiniteISP to path
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "inifinite-isp"))

try:
    from infinite_isp import InfiniteISP
except ImportError:
    print("Warning: InfiniteISP not available. Some features may be limited.")
    InfiniteISP = None

from .color import ColorSpaceConverter, BT2020Converter
from .utils import l3_root_path


@dataclass
class L3Config:
    """Configuration for L3 ISP pipeline."""
    
    # Core L3 parameters
    patch_size: Tuple[int, int] = (5, 5)
    num_cut_points: int = 30
    cut_point_range: Tuple[float, float] = (-1.7, -0.12)
    min_patches_per_class: int = 50
    
    # ISP pipeline settings
    use_infinite_isp: bool = True
    infinite_isp_config: Optional[str] = None
    
    # Color space settings
    output_color_space: str = "BT.2020"
    linear_output: bool = True
    
    # Training settings
    training_method: str = "OLS"  # OLS, Ridge, Wiener
    ridge_alpha: float = 1.0
    wiener_noise_var: float = 0.01
    
    # Data preprocessing
    black_level: float = 0.0
    white_level: float = 1.0
    normalize_input: bool = True
    
    # Rendering settings
    interpolate_missing_kernels: bool = True
    apply_gamma_correction: bool = False
    gamma_value: float = 2.2
    
    # Debug and visualization
    debug: bool = False
    save_intermediate: bool = False
    output_dir: str = "output"
    
    # Dataset specific
    dataset_type: str = "fivek"  # fivek, synthetic, iset
    dataset_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "patch_size": list(self.patch_size),  # Convert tuple to list for YAML
            "num_cut_points": self.num_cut_points,
            "cut_point_range": list(self.cut_point_range),  # Convert tuple to list
            "min_patches_per_class": self.min_patches_per_class,
            "use_infinite_isp": self.use_infinite_isp,
            "infinite_isp_config": self.infinite_isp_config,
            "output_color_space": self.output_color_space,
            "linear_output": self.linear_output,
            "training_method": self.training_method,
            "ridge_alpha": self.ridge_alpha,
            "wiener_noise_var": self.wiener_noise_var,
            "black_level": self.black_level,
            "white_level": self.white_level,
            "normalize_input": self.normalize_input,
            "interpolate_missing_kernels": self.interpolate_missing_kernels,
            "apply_gamma_correction": self.apply_gamma_correction,
            "gamma_value": self.gamma_value,
            "debug": self.debug,
            "save_intermediate": self.save_intermediate,
            "output_dir": self.output_dir,
            "dataset_type": self.dataset_type,
            "dataset_path": self.dataset_path,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "L3Config":
        """Create config from dictionary."""
        # Convert lists back to tuples
        if "patch_size" in config_dict and isinstance(config_dict["patch_size"], list):
            config_dict["patch_size"] = tuple(config_dict["patch_size"])
        if "cut_point_range" in config_dict and isinstance(config_dict["cut_point_range"], list):
            config_dict["cut_point_range"] = tuple(config_dict["cut_point_range"])
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "L3Config":
        """Load config from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def save_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class L3Pipeline:
    """Main L3 ISP pipeline class."""
    
    def __init__(self, config: Optional[L3Config] = None):
        """Initialize L3 pipeline.
        
        Args:
            config: L3 configuration. If None, uses default config.
        """
        self.config = config or L3Config()
        self.infinite_isp = None
        self.color_converter = None
        self.trained_kernels = None
        self.classifier = None
        self.trainer = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize pipeline components."""
        # Initialize color converter
        if self.config.output_color_space == "BT.2020":
            self.color_converter = BT2020Converter()
        else:
            self.color_converter = ColorSpaceConverter()
        
        # Initialize InfiniteISP if enabled
        if self.config.use_infinite_isp and InfiniteISP is not None:
            self._initialize_infinite_isp()
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def _initialize_infinite_isp(self) -> None:
        """Initialize InfiniteISP pipeline."""
        if self.config.infinite_isp_config is None:
            # Use default config
            config_path = Path(__file__).parent.parent / "external" / "inifinite-isp" / "config" / "configs.yml"
        else:
            config_path = self.config.infinite_isp_config
        
        try:
            # Initialize with dummy data path - will be updated per image
            self.infinite_isp = InfiniteISP(
                data_path=str(Path(__file__).parent.parent / "external" / "inifinite-isp" / "in_frames" / "normal"),
                config_path=str(config_path)
            )
        except Exception as e:
            print(f"Warning: Failed to initialize InfiniteISP: {e}")
            self.infinite_isp = None
    
    def process_image(
        self, 
        raw_image: np.ndarray,
        target_image: Optional[np.ndarray] = None,
        sensor_info: Optional[Dict[str, Any]] = None,
        use_trained_kernels: bool = True
    ) -> np.ndarray:
        """Process a single raw image through the L3 ISP pipeline.
        
        Args:
            raw_image: Raw Bayer image
            target_image: Target RGB image (for training)
            sensor_info: Sensor information (width, height, bit_depth, bayer_pattern)
            use_trained_kernels: Whether to use trained kernels or apply ISP directly
            
        Returns:
            Processed RGB image
        """
        if self.config.debug:
            print(f"Processing image with shape: {raw_image.shape}")
        
        # Apply InfiniteISP preprocessing if enabled
        if self.config.use_infinite_isp and self.infinite_isp is not None:
            processed_image = self._apply_infinite_isp(raw_image, sensor_info)
        else:
            processed_image = raw_image
        
        # Apply L3 processing if trained kernels are available
        if use_trained_kernels and self.trained_kernels is not None:
            processed_image = self._apply_l3_kernels(processed_image, sensor_info)
        else:
            # Fallback: simple Bayer to RGB conversion for testing
            processed_image = self._simple_bayer_to_rgb(processed_image)
        
        # Apply color space conversion
        if self.color_converter is not None:
            processed_image = self.color_converter.convert(processed_image)
        
        # Apply gamma correction if requested
        if self.config.apply_gamma_correction:
            processed_image = self._apply_gamma_correction(processed_image)
        
        return processed_image
    
    def _apply_infinite_isp(self, raw_image: np.ndarray, sensor_info: Optional[Dict[str, Any]]) -> np.ndarray:
        """Apply InfiniteISP pipeline to raw image."""
        if sensor_info is not None:
            # Update sensor info in InfiniteISP
            isp_sensor_info = [
                sensor_info.get("width", raw_image.shape[1]),
                sensor_info.get("height", raw_image.shape[0]),
                sensor_info.get("bit_depth", 12),
                sensor_info.get("bayer_pattern", "RGGB").lower(),
            ]
            self.infinite_isp.update_sensor_info(isp_sensor_info)
        
        # Set raw image directly
        self.infinite_isp.raw = raw_image
        
        # Run ISP pipeline
        self.infinite_isp.run_pipeline(visualize_output=False)
        
        # Return processed image (this is a simplified version)
        # In reality, you'd need to access the internal processed image
        return raw_image  # Placeholder
    
    def _apply_l3_kernels(self, image: np.ndarray, sensor_info: Optional[Dict[str, Any]]) -> np.ndarray:
        """Apply trained L3 kernels to image."""
        if self.trained_kernels is None or self.trainer is None:
            return self._simple_bayer_to_rgb(image)
        
        # Use the L3 renderer
        from .render import L3Render
        renderer = L3Render()
        
        try:
            return renderer.render(image, self.trainer)
        except Exception as e:
            print(f"Warning: L3 rendering failed: {e}")
            return self._simple_bayer_to_rgb(image)
    
    def _simple_bayer_to_rgb(self, bayer_image: np.ndarray) -> np.ndarray:
        """Simple Bayer to RGB conversion for testing."""
        height, width = bayer_image.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.float32)
        
        # Simple RGGB pattern demosaicing
        # R: (0,0), (0,2), (2,0), (2,2), ...
        # G: (0,1), (1,0), (0,3), (1,2), ...  
        # B: (1,1), (1,3), (3,1), (3,3), ...
        
        # Extract channels
        r_mask = np.zeros((height, width), dtype=bool)
        g_mask = np.zeros((height, width), dtype=bool)
        b_mask = np.zeros((height, width), dtype=bool)
        
        r_mask[0::2, 0::2] = True  # R pixels
        g_mask[0::2, 1::2] = True  # G pixels (first row)
        g_mask[1::2, 0::2] = True  # G pixels (second row)
        b_mask[1::2, 1::2] = True  # B pixels
        
        # Simple interpolation - just replicate values
        rgb_image[:, :, 0] = bayer_image  # Use all values for R
        rgb_image[:, :, 1] = bayer_image  # Use all values for G  
        rgb_image[:, :, 2] = bayer_image  # Use all values for B
        
        # Apply masks to get approximate demosaicing
        for y in range(height):
            for x in range(width):
                if r_mask[y, x]:
                    rgb_image[y, x, 1] *= 0.8  # Reduce G in R pixels
                    rgb_image[y, x, 2] *= 0.6  # Reduce B in R pixels
                elif b_mask[y, x]:
                    rgb_image[y, x, 0] *= 0.6  # Reduce R in B pixels
                    rgb_image[y, x, 1] *= 0.8  # Reduce G in B pixels
                else:  # G pixel
                    rgb_image[y, x, 0] *= 0.7  # Reduce R in G pixels
                    rgb_image[y, x, 2] *= 0.7  # Reduce B in G pixels
        
        return np.clip(rgb_image, 0, 1)
    
    def _apply_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction to image."""
        return np.power(image, 1.0 / self.config.gamma_value)
    
    def train(
        self,
        training_data: List[Tuple[np.ndarray, np.ndarray]],
        validation_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> Dict[str, Any]:
        """Train the L3 pipeline on provided data.
        
        Args:
            training_data: List of (raw_image, target_image) pairs
            validation_data: Optional validation data
            
        Returns:
            Training metrics and results
        """
        if self.config.debug:
            print(f"Training on {len(training_data)} image pairs")
        
        # Import training components
        from .classify import L3Classifier
        from .train import L3TrainOLS, L3TrainRidge
        
        # Initialize classifier
        self.classifier = L3Classifier(
            patch_size=self.config.patch_size,
            num_cut_points=self.config.num_cut_points,
            cut_point_range=self.config.cut_point_range
        )
        
        # Initialize trainer
        if self.config.training_method == "OLS":
            self.trainer = L3TrainOLS(classifier=self.classifier)
        elif self.config.training_method == "Ridge":
            self.trainer = L3TrainRidge(
                classifier=self.classifier,
                alpha=self.config.ridge_alpha
            )
        else:
            raise ValueError(f"Unknown training method: {self.config.training_method}")
        
        # Train the model
        training_results = self.trainer.train(training_data)
        
        # Store trained kernels
        self.trained_kernels = self.trainer.get_kernels()
        
        # Validate if validation data provided
        validation_results = None
        if validation_data is not None:
            validation_results = self.validate(validation_data)
        
        return {
            "training_results": training_results,
            "validation_results": validation_results,
            "num_kernels": len(self.trained_kernels) if self.trained_kernels else 0,
        }
    
    def validate(self, validation_data: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Validate the trained pipeline.
        
        Args:
            validation_data: List of (raw_image, target_image) pairs
            
        Returns:
            Validation metrics
        """
        if self.trained_kernels is None:
            raise ValueError("Pipeline must be trained before validation")
        
        total_mse = 0.0
        total_psnr = 0.0
        
        for raw_image, target_image in validation_data:
            processed_image = self.process_image(raw_image, use_trained_kernels=True)
            
            # Ensure compatible shapes for comparison
            if len(processed_image.shape) == 2 and len(target_image.shape) == 3:
                # Convert grayscale processed to RGB by replicating
                processed_image = np.stack([processed_image] * 3, axis=2)
            elif len(processed_image.shape) == 3 and len(target_image.shape) == 2:
                # Convert target to RGB if needed
                target_image = np.stack([target_image] * 3, axis=2)
            
            # Calculate MSE
            mse = np.mean((processed_image - target_image) ** 2)
            total_mse += mse
            
            # Calculate PSNR
            if mse > 0:
                psnr = 20 * np.log10(1.0 / np.sqrt(mse))
                total_psnr += psnr
        
        avg_mse = total_mse / len(validation_data)
        avg_psnr = total_psnr / len(validation_data)
        
        return {
            "mse": avg_mse,
            "psnr": avg_psnr,
            "num_images": len(validation_data),
        }
    
    def save_model(self, model_path: str) -> None:
        """Save trained model to file."""
        model_data = {
            "config": self.config.to_dict(),
            "trained_kernels": self.trained_kernels,
            "classifier_state": self.classifier.get_state() if self.classifier else None,
        }
        
        np.savez_compressed(model_path, **model_data)
        
        if self.config.debug:
            print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load trained model from file."""
        model_data = np.load(model_path, allow_pickle=True)
        
        # Load config
        config_dict = model_data["config"].item()
        self.config = L3Config.from_dict(config_dict)
        
        # Load trained kernels
        self.trained_kernels = model_data["trained_kernels"]
        
        # Load classifier state
        if "classifier_state" in model_data and model_data["classifier_state"] is not None:
            from .classify import L3Classifier
            self.classifier = L3Classifier.from_state(model_data["classifier_state"].item())
        
        # Reinitialize components
        self._initialize_components()
        
        if self.config.debug:
            print(f"Model loaded from {model_path}")
