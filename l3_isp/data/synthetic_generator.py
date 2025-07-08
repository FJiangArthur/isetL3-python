"""Synthetic data generation for L3 training."""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass

from .base import L3DataBase
from ..utils import create_bayer_pattern, create_synthetic_bayer
from ..color import BT2020Converter


@dataclass
class SyntheticScene:
    """Configuration for synthetic scene generation."""
    scene_type: str = "gradient"  # gradient, checkerboard, natural, noise
    num_objects: int = 3
    color_variance: float = 0.3
    texture_complexity: float = 0.5
    lighting_variation: float = 0.2


class SyntheticDataGenerator(L3DataBase):
    """Generate synthetic Bayer/RGB pairs for L3 training."""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        bayer_pattern: str = "RGGB",
        noise_std: float = 0.01,
        black_level: float = 0.0,
        white_level: float = 1.0,
        num_scenes: int = 1000,
        scene_config: Optional[SyntheticScene] = None,
        name: str = "Synthetic Data Generator"
    ):
        """Initialize synthetic data generator.
        
        Args:
            image_size: Output image size (height, width)
            bayer_pattern: Bayer pattern string
            noise_std: Standard deviation of sensor noise
            black_level: Black level offset
            white_level: White level saturation
            num_scenes: Number of scenes to generate
            scene_config: Scene generation configuration
            name: Generator name
        """
        super().__init__(name)
        
        self.image_size = image_size
        self.bayer_pattern = bayer_pattern
        self.noise_std = noise_std
        self.black_level = black_level
        self.white_level = white_level
        self.num_scenes = num_scenes
        self.scene_config = scene_config or SyntheticScene()
        
        # Initialize color converter
        self.color_converter = BT2020Converter()
        
        # Create CFA pattern
        self.cfa_pattern = create_bayer_pattern(bayer_pattern)
        
        # Pre-generate scene parameters for reproducibility
        np.random.seed(42)  # For reproducible synthetic data
        self.scene_params = [self._generate_scene_params() for _ in range(num_scenes)]
        np.random.seed()  # Reset to random
    
    def _generate_scene_params(self) -> Dict[str, Any]:
        """Generate parameters for a synthetic scene."""
        return {
            "scene_type": np.random.choice([
                "gradient", "checkerboard", "natural", "noise", "geometric"
            ]),
            "base_colors": np.random.rand(3, 3),  # 3 base colors, RGB
            "spatial_freq": np.random.uniform(0.1, 2.0),
            "contrast": np.random.uniform(0.3, 1.0),
            "brightness": np.random.uniform(0.2, 0.8),
            "color_cast": np.random.uniform(0.8, 1.2, 3),
            "vignetting": np.random.uniform(0.8, 1.0),
            "rotation": np.random.uniform(0, 360),
        }
    
    def generate_sample(self, scene_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single synthetic Bayer/RGB pair.
        
        Args:
            scene_idx: Scene index (random if None)
            
        Returns:
            Tuple of (bayer_image, target_rgb)
        """
        if scene_idx is None:
            scene_idx = np.random.randint(0, len(self.scene_params))
        
        params = self.scene_params[scene_idx % len(self.scene_params)]
        
        # Generate clean RGB image
        clean_rgb = self._generate_scene_image(params)
        
        # Apply sensor effects
        degraded_rgb = self._apply_sensor_effects(clean_rgb, params)
        
        # Create Bayer mosaic
        bayer_image = create_synthetic_bayer(
            degraded_rgb,
            self.bayer_pattern,
            add_noise=True,
            noise_std=self.noise_std
        )
        
        # Convert clean RGB to BT.2020 for target
        target_rgb = self.color_converter.srgb_to_linear_bt2020(clean_rgb)
        
        return bayer_image, target_rgb
    
    def _generate_scene_image(self, params: Dict[str, Any]) -> np.ndarray:
        """Generate a synthetic scene image based on parameters."""
        height, width = self.image_size
        scene_type = params["scene_type"]
        
        if scene_type == "gradient":
            return self._generate_gradient_scene(params)
        elif scene_type == "checkerboard":
            return self._generate_checkerboard_scene(params)
        elif scene_type == "natural":
            return self._generate_natural_scene(params)
        elif scene_type == "noise":
            return self._generate_noise_scene(params)
        elif scene_type == "geometric":
            return self._generate_geometric_scene(params)
        else:
            # Default to gradient
            return self._generate_gradient_scene(params)
    
    def _generate_gradient_scene(self, params: Dict[str, Any]) -> np.ndarray:
        """Generate gradient scene."""
        height, width = self.image_size
        
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        x_norm = x / width
        y_norm = y / height
        
        # Generate gradients
        base_colors = params["base_colors"]
        spatial_freq = params["spatial_freq"]
        
        # Horizontal and vertical gradients
        h_grad = np.sin(2 * np.pi * spatial_freq * x_norm)
        v_grad = np.cos(2 * np.pi * spatial_freq * y_norm)
        
        # Combine gradients
        pattern = (h_grad[:, :, np.newaxis] + v_grad[:, :, np.newaxis]) / 2
        
        # Apply base colors
        image = np.zeros((height, width, 3))
        for c in range(3):
            image[:, :, c] = (
                base_colors[0, c] * (1 + pattern[:, :, 0]) / 2 +
                base_colors[1, c] * (1 - pattern[:, :, 0]) / 2
            )
        
        return np.clip(image, 0, 1)
    
    def _generate_checkerboard_scene(self, params: Dict[str, Any]) -> np.ndarray:
        """Generate checkerboard scene."""
        height, width = self.image_size
        spatial_freq = params["spatial_freq"]
        base_colors = params["base_colors"]
        
        # Create checkerboard pattern
        tile_size = max(1, int(min(height, width) / (8 * spatial_freq)))
        
        y, x = np.ogrid[:height, :width]
        checker = ((x // tile_size) + (y // tile_size)) % 2
        
        # Apply colors
        image = np.zeros((height, width, 3))
        for c in range(3):
            image[:, :, c] = np.where(
                checker == 0,
                base_colors[0, c],
                base_colors[1, c]
            )
        
        return np.clip(image, 0, 1)
    
    def _generate_natural_scene(self, params: Dict[str, Any]) -> np.ndarray:
        """Generate natural-looking scene with multiple objects."""
        height, width = self.image_size
        base_colors = params["base_colors"]
        
        # Start with background gradient
        image = self._generate_gradient_scene(params)
        
        # Add circular objects
        for i in range(3):
            # Random object parameters
            center_x = np.random.uniform(0.2, 0.8) * width
            center_y = np.random.uniform(0.2, 0.8) * height
            radius = np.random.uniform(0.05, 0.15) * min(height, width)
            
            # Create circular mask
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 < radius**2
            
            # Apply object color
            object_color = base_colors[i % len(base_colors)]
            for c in range(3):
                image[mask, c] = object_color[c]
        
        return np.clip(image, 0, 1)
    
    def _generate_noise_scene(self, params: Dict[str, Any]) -> np.ndarray:
        """Generate noise-based scene."""
        height, width = self.image_size
        contrast = params["contrast"]
        brightness = params["brightness"]
        
        # Generate colored noise
        noise = np.random.rand(height, width, 3)
        
        # Apply contrast and brightness
        image = (noise - 0.5) * contrast + brightness
        
        return np.clip(image, 0, 1)
    
    def _generate_geometric_scene(self, params: Dict[str, Any]) -> np.ndarray:
        """Generate geometric pattern scene."""
        height, width = self.image_size
        base_colors = params["base_colors"]
        spatial_freq = params["spatial_freq"]
        
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        x_norm = (x - width/2) / width
        y_norm = (y - height/2) / height
        
        # Generate geometric patterns
        r = np.sqrt(x_norm**2 + y_norm**2)
        theta = np.arctan2(y_norm, x_norm)
        
        # Radial and angular patterns
        radial_pattern = np.sin(2 * np.pi * spatial_freq * r * 5)
        angular_pattern = np.cos(6 * theta)
        
        # Combine patterns
        pattern = (radial_pattern + angular_pattern) / 2
        
        # Apply colors
        image = np.zeros((height, width, 3))
        for c in range(3):
            image[:, :, c] = (
                base_colors[0, c] * (1 + pattern) / 2 +
                base_colors[1, c] * (1 - pattern) / 2
            )
        
        return np.clip(image, 0, 1)
    
    def _apply_sensor_effects(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply realistic sensor effects to clean image."""
        result = image.copy()
        
        # Apply color cast (white balance error)
        color_cast = params["color_cast"]
        result = result * color_cast.reshape(1, 1, 3)
        
        # Apply vignetting
        height, width = self.image_size
        y, x = np.ogrid[:height, :width]
        x_norm = (x - width/2) / (width/2)
        y_norm = (y - height/2) / (height/2)
        r_norm = np.sqrt(x_norm**2 + y_norm**2)
        
        vignetting_strength = params["vignetting"]
        vignette = 1 - (1 - vignetting_strength) * r_norm**2
        vignette = np.clip(vignette, 0.3, 1.0)
        
        result = result * vignette[:, :, np.newaxis]
        
        # Apply brightness adjustment
        brightness = params["brightness"]
        result = result * brightness
        
        # Clip to valid range
        result = np.clip(result, self.black_level, self.white_level)
        
        return result
    
    def generate_batch(
        self,
        batch_size: int,
        start_idx: int = 0
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate a batch of synthetic data.
        
        Args:
            batch_size: Number of samples to generate
            start_idx: Starting scene index
            
        Returns:
            List of (bayer_image, target_rgb) pairs
        """
        batch = []
        for i in range(batch_size):
            scene_idx = (start_idx + i) % len(self.scene_params)
            bayer_image, target_rgb = self.generate_sample(scene_idx)
            batch.append((bayer_image, target_rgb))
        
        return batch
    
    def __len__(self) -> int:
        """Return number of synthetic scenes."""
        return self.num_scenes
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Get a synthetic sample with metadata."""
        bayer_image, target_rgb = self.generate_sample(idx)
        
        metadata = {
            "scene_idx": idx,
            "scene_params": self.scene_params[idx % len(self.scene_params)],
            "bayer_pattern": self.bayer_pattern,
            "image_size": self.image_size,
            "noise_std": self.noise_std,
            "type": "synthetic",
        }
        
        return bayer_image, target_rgb, metadata
    
    def data_get(
        self,
        n_img: int = 1,
        start_idx: int = 0
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
        """Get multiple images (compatible with L3 interface)."""
        raw_images = []
        target_images = []
        metadata_list = []
        
        for i in range(n_img):
            idx = (start_idx + i) % len(self)
            bayer_image, target_rgb, metadata = self[idx]
            raw_images.append(bayer_image)
            target_images.append(target_rgb)
            metadata_list.append(metadata)
        
        return raw_images, target_images, metadata_list
    
    @property
    def cfa(self) -> np.ndarray:
        """Get CFA pattern for the dataset."""
        return self.cfa_pattern
