"""Color space conversion utilities for L3 ISP."""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np
try:
    import colour
except ImportError:
    print("Warning: colour-science not available. Color conversions may be limited.")
    colour = None


class ColorSpaceConverter:
    """Base color space converter."""
    
    def __init__(self):
        """Initialize color space converter."""
        pass
    
    def convert(self, image: np.ndarray, source: str = "linear", target: str = "sRGB") -> np.ndarray:
        """Convert between color spaces.
        
        Args:
            image: Input image
            source: Source color space
            target: Target color space
            
        Returns:
            Converted image
        """
        if source == target:
            return image
        
        # Basic conversions
        if source == "linear" and target == "sRGB":
            return self.linear_to_srgb(image)
        elif source == "sRGB" and target == "linear":
            return self.srgb_to_linear(image)
        else:
            return image  # Placeholder
    
    def linear_to_srgb(self, image: np.ndarray) -> np.ndarray:
        """Convert linear RGB to sRGB."""
        # Apply sRGB gamma curve
        return np.where(
            image <= 0.0031308,
            12.92 * image,
            1.055 * np.power(image, 1.0 / 2.4) - 0.055
        )
    
    def srgb_to_linear(self, image: np.ndarray) -> np.ndarray:
        """Convert sRGB to linear RGB."""
        # Apply inverse sRGB gamma curve
        return np.where(
            image <= 0.04045,
            image / 12.92,
            np.power((image + 0.055) / 1.055, 2.4)
        )


class BT2020Converter(ColorSpaceConverter):
    """BT.2020 color space converter."""
    
    def __init__(self):
        """Initialize BT.2020 converter."""
        super().__init__()
        
        # Color transformation matrices
        # sRGB to XYZ (D65)
        self.srgb_to_xyz = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        # XYZ to BT.2020 (D65)
        self.xyz_to_bt2020 = np.array([
            [1.7166511, -0.3556708, -0.2533663],
            [-0.6666844, 1.6164812, 0.0157685],
            [0.0176399, -0.0427706, 0.9421031]
        ])
        
        # Combined transformation: sRGB -> BT.2020
        self.srgb_to_bt2020 = self.xyz_to_bt2020 @ self.srgb_to_xyz
        
        # Inverse transformation: BT.2020 -> sRGB
        self.bt2020_to_srgb = np.linalg.inv(self.srgb_to_bt2020)
    
    def srgb_to_linear_bt2020(self, image: np.ndarray) -> np.ndarray:
        """Convert sRGB to linear BT.2020.
        
        Args:
            image: sRGB image (0-1 range)
            
        Returns:
            Linear BT.2020 image
        """
        # First convert sRGB to linear
        linear_srgb = self.srgb_to_linear(image)
        
        # Then convert to BT.2020 color space
        # Reshape for matrix multiplication
        original_shape = linear_srgb.shape
        if len(original_shape) == 3 and original_shape[2] == 3:
            linear_srgb_flat = linear_srgb.reshape(-1, 3)
            bt2020_flat = (self.srgb_to_bt2020 @ linear_srgb_flat.T).T
            return bt2020_flat.reshape(original_shape)
        else:
            return linear_srgb  # Return as-is if not RGB
    
    def linear_bt2020_to_srgb(self, image: np.ndarray) -> np.ndarray:
        """Convert linear BT.2020 to sRGB.
        
        Args:
            image: Linear BT.2020 image
            
        Returns:
            sRGB image (0-1 range)
        """
        # First convert from BT.2020 to linear sRGB
        original_shape = image.shape
        if len(original_shape) == 3 and original_shape[2] == 3:
            bt2020_flat = image.reshape(-1, 3)
            linear_srgb_flat = (self.bt2020_to_srgb @ bt2020_flat.T).T
            linear_srgb = linear_srgb_flat.reshape(original_shape)
        else:
            linear_srgb = image
        
        # Then apply sRGB gamma curve
        return self.linear_to_srgb(linear_srgb)
    
    def apply_bt2020_gamma(self, image: np.ndarray, system_gamma: float = 2.4) -> np.ndarray:
        """Apply BT.2020 gamma curve.
        
        Args:
            image: Linear BT.2020 image
            system_gamma: System gamma value (default 2.4)
            
        Returns:
            Gamma-corrected BT.2020 image
        """
        # BT.2020 uses the same gamma curve as sRGB for compatibility
        return self.linear_to_srgb(image)
    
    def convert(self, image: np.ndarray, source: str = "sRGB", target: str = "BT.2020") -> np.ndarray:
        """Convert between color spaces.
        
        Args:
            image: Input image
            source: Source color space (sRGB, linear, BT.2020)
            target: Target color space (sRGB, linear, BT.2020)
            
        Returns:
            Converted image
        """
        if source == target:
            return image
        
        # Handle conversions
        if source == "sRGB" and target == "BT.2020":
            return self.srgb_to_linear_bt2020(image)
        elif source == "BT.2020" and target == "sRGB":
            return self.linear_bt2020_to_srgb(image)
        elif source == "linear" and target == "BT.2020":
            # Assume linear is sRGB linear
            return self._apply_color_matrix(image, self.srgb_to_bt2020)
        elif source == "BT.2020" and target == "linear":
            return self._apply_color_matrix(image, self.bt2020_to_srgb)
        else:
            # Fall back to parent class
            return super().convert(image, source, target)
    
    def _apply_color_matrix(self, image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Apply color transformation matrix to image.
        
        Args:
            image: Input image
            matrix: 3x3 transformation matrix
            
        Returns:
            Transformed image
        """
        original_shape = image.shape
        if len(original_shape) == 3 and original_shape[2] == 3:
            image_flat = image.reshape(-1, 3)
            transformed_flat = (matrix @ image_flat.T).T
            return transformed_flat.reshape(original_shape)
        else:
            return image
    
    def get_bt2020_primaries(self) -> dict:
        """Get BT.2020 color primaries.
        
        Returns:
            Dictionary with RGB primaries in xy coordinates
        """
        return {
            "red": (0.708, 0.292),
            "green": (0.170, 0.797),
            "blue": (0.131, 0.046),
            "white": (0.3127, 0.3290)  # D65 white point
        }
    
    def get_bt2020_transfer_function(self) -> str:
        """Get BT.2020 transfer function name."""
        return "ITU-R BT.2020"


def create_color_correction_matrix(
    source_primaries: Dict[str, Tuple[float, float]],
    target_primaries: Dict[str, Tuple[float, float]]
) -> np.ndarray:
    """Create color correction matrix between two sets of primaries.
    
    Args:
        source_primaries: Source color primaries
        target_primaries: Target color primaries
        
    Returns:
        3x3 color correction matrix
    """
    # This is a simplified implementation
    # In practice, you'd use more sophisticated color science
    if colour is not None:
        # Use colour-science library if available
        try:
            source_rgb = colour.RGB_COLOURSPACES["sRGB"]
            target_rgb = colour.RGB_COLOURSPACES["ITU-R BT.2020"]
            return colour.matrix_RGB_to_RGB(source_rgb, target_rgb)
        except Exception:
            pass
    
    # Fallback to identity matrix
    return np.eye(3)


def apply_white_balance(
    image: np.ndarray,
    gains: Union[Tuple[float, float, float], np.ndarray],
    bayer_pattern: Optional[str] = None
) -> np.ndarray:
    """Apply white balance gains to image.
    
    Args:
        image: Input image (RGB or Bayer)
        gains: White balance gains (R, G, B)
        bayer_pattern: Bayer pattern if input is Bayer
        
    Returns:
        White-balanced image
    """
    if isinstance(gains, tuple):
        gains = np.array(gains)
    
    if bayer_pattern is not None:
        # Apply to Bayer image
        return _apply_bayer_white_balance(image, gains, bayer_pattern)
    else:
        # Apply to RGB image
        if len(image.shape) == 3 and image.shape[2] == 3:
            return image * gains.reshape(1, 1, 3)
        else:
            return image


def _apply_bayer_white_balance(
    bayer_image: np.ndarray,
    gains: np.ndarray,
    bayer_pattern: str
) -> np.ndarray:
    """Apply white balance to Bayer image.
    
    Args:
        bayer_image: Bayer mosaic image
        gains: White balance gains [R, G, B]
        bayer_pattern: Bayer pattern string
        
    Returns:
        White-balanced Bayer image
    """
    result = bayer_image.copy()
    pattern = bayer_pattern.upper()
    
    # Define gain mapping for each Bayer pattern
    gain_maps = {
        "RGGB": [(0, 0, gains[0]), (0, 1, gains[1]), (1, 0, gains[1]), (1, 1, gains[2])],
        "BGGR": [(0, 0, gains[2]), (0, 1, gains[1]), (1, 0, gains[1]), (1, 1, gains[0])],
        "GRBG": [(0, 0, gains[1]), (0, 1, gains[0]), (1, 0, gains[2]), (1, 1, gains[1])],
        "GBRG": [(0, 0, gains[1]), (0, 1, gains[2]), (1, 0, gains[0]), (1, 1, gains[1])],
    }
    
    if pattern in gain_maps:
        for row_offset, col_offset, gain in gain_maps[pattern]:
            result[row_offset::2, col_offset::2] *= gain
    
    return result
