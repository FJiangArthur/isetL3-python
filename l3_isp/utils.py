"""Utility functions for L3 ISP."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
try:
    from numba import jit
except ImportError:
    # Fallback when numba is not available
    def jit(nopython=True):
        def decorator(func):
            return func
        return decorator


def l3_root_path() -> str:
    """Get the root path of the L3 repository.
    
    Returns:
        Root path as string
    """
    return str(Path(__file__).parent.parent)


def create_bayer_pattern(pattern: str) -> np.ndarray:
    """Create Bayer CFA pattern array.
    
    Args:
        pattern: Bayer pattern string (RGGB, BGGR, GRBG, GBRG)
        
    Returns:
        2x2 pattern array with channel indices
    """
    pattern = pattern.upper()
    channel_map = {"R": 0, "G": 1, "B": 2}
    
    pattern_arrays = {
        "RGGB": np.array([[0, 1], [1, 2]]),  # R G / G B
        "BGGR": np.array([[2, 1], [1, 0]]),  # B G / G R
        "GRBG": np.array([[1, 0], [2, 1]]),  # G R / B G
        "GBRG": np.array([[1, 2], [0, 1]]),  # G B / R G
    }
    
    if pattern not in pattern_arrays:
        raise ValueError(f"Unknown Bayer pattern: {pattern}")
    
    return pattern_arrays[pattern]


def get_bayer_type_map(bayer_image: np.ndarray, pattern: str) -> np.ndarray:
    """Get pixel type map for Bayer image.
    
    Args:
        bayer_image: Bayer mosaic image
        pattern: Bayer pattern string
        
    Returns:
        Array with pixel types (0=R, 1=G, 2=B)
    """
    height, width = bayer_image.shape
    pattern_array = create_bayer_pattern(pattern)
    
    # Create type map by tiling the 2x2 pattern
    type_map = np.tile(pattern_array, (height // 2 + 1, width // 2 + 1))
    return type_map[:height, :width]


@jit(nopython=True)
def extract_patches_fast(
    image: np.ndarray,
    patch_size: Tuple[int, int],
    stride: int = 1,
    max_patches: int = 1000
) -> np.ndarray:
    """Fast patch extraction using Numba.
    
    Args:
        image: Input image
        patch_size: Size of patches (height, width)
        stride: Stride for patch extraction
        max_patches: Maximum number of patches to extract
        
    Returns:
        Array of patches
    """
    height, width = image.shape[:2]
    patch_h, patch_w = patch_size
    
    patches = []
    count = 0
    
    for y in range(0, height - patch_h + 1, stride):
        for x in range(0, width - patch_w + 1, stride):
            if count >= max_patches:
                break
            
            patch = image[y:y + patch_h, x:x + patch_w]
            patches.append(patch)
            count += 1
        
        if count >= max_patches:
            break
    
    return np.array(patches)


def extract_patches(
    image: np.ndarray,
    patch_size: Tuple[int, int],
    stride: int = 1,
    max_patches: int = 1000,
    random_selection: bool = False
) -> List[np.ndarray]:
    """Extract patches from image.
    
    Args:
        image: Input image
        patch_size: Size of patches (height, width)
        stride: Stride for patch extraction
        max_patches: Maximum number of patches to extract
        random_selection: Whether to randomly select patches
        
    Returns:
        List of patches
    """
    height, width = image.shape[:2]
    patch_h, patch_w = patch_size
    
    # Calculate all possible patch positions
    positions = []
    for y in range(0, height - patch_h + 1, stride):
        for x in range(0, width - patch_w + 1, stride):
            positions.append((y, x))
    
    # Select positions
    if len(positions) > max_patches:
        if random_selection:
            indices = np.random.choice(len(positions), max_patches, replace=False)
            selected_positions = [positions[i] for i in indices]
        else:
            # Take evenly spaced positions
            step = len(positions) // max_patches
            selected_positions = [positions[i] for i in range(0, len(positions), step)][:max_patches]
    else:
        selected_positions = positions
    
    # Extract patches
    patches = []
    for y, x in selected_positions:
        patch = image[y:y + patch_h, x:x + patch_w]
        patches.append(patch)
    
    return patches


def compute_patch_statistics(
    patch: np.ndarray,
    channel_map: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute statistics for a patch.
    
    Args:
        patch: Image patch
        channel_map: Channel map for Bayer images
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    if channel_map is not None:
        # Bayer patch statistics
        for channel in [0, 1, 2]:  # R, G, B
            mask = channel_map == channel
            if np.any(mask):
                channel_values = patch[mask]
                stats[f"channel_{channel}_mean"] = float(np.mean(channel_values))
                stats[f"channel_{channel}_std"] = float(np.std(channel_values))
                stats[f"channel_{channel}_max"] = float(np.max(channel_values))
    else:
        # RGB patch statistics
        if len(patch.shape) == 3:
            for c in range(patch.shape[2]):
                channel_data = patch[:, :, c]
                stats[f"channel_{c}_mean"] = float(np.mean(channel_data))
                stats[f"channel_{c}_std"] = float(np.std(channel_data))
                stats[f"channel_{c}_max"] = float(np.max(channel_data))
        else:
            stats["mean"] = float(np.mean(patch))
            stats["std"] = float(np.std(patch))
            stats["max"] = float(np.max(patch))
    
    # Global statistics
    stats["overall_mean"] = float(np.mean(patch))
    stats["overall_std"] = float(np.std(patch))
    stats["contrast"] = float(np.max(patch) - np.min(patch))
    
    return stats


def normalize_image(
    image: np.ndarray,
    method: str = "minmax",
    clip_percentile: float = 0.1
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Normalize image to [0, 1] range.
    
    Args:
        image: Input image
        method: Normalization method (minmax, percentile, zscore)
        clip_percentile: Percentile for clipping (used with percentile method)
        
    Returns:
        Tuple of (normalized_image, normalization_params)
    """
    params = {"method": method}
    
    if method == "minmax":
        min_val = np.min(image)
        max_val = np.max(image)
        params["min_val"] = float(min_val)
        params["max_val"] = float(max_val)
        
        if max_val > min_val:
            normalized = (image - min_val) / (max_val - min_val)
        else:
            normalized = image
    
    elif method == "percentile":
        low_val = np.percentile(image, clip_percentile)
        high_val = np.percentile(image, 100 - clip_percentile)
        params["low_val"] = float(low_val)
        params["high_val"] = float(high_val)
        
        normalized = np.clip((image - low_val) / (high_val - low_val), 0, 1)
    
    elif method == "zscore":
        mean_val = np.mean(image)
        std_val = np.std(image)
        params["mean_val"] = float(mean_val)
        params["std_val"] = float(std_val)
        
        if std_val > 0:
            normalized = (image - mean_val) / std_val
            # Scale to [0, 1]
            normalized = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))
        else:
            normalized = image
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def denormalize_image(
    normalized_image: np.ndarray,
    params: Dict[str, Any]
) -> np.ndarray:
    """Denormalize image using stored parameters.
    
    Args:
        normalized_image: Normalized image
        params: Normalization parameters
        
    Returns:
        Denormalized image
    """
    method = params["method"]
    
    if method == "minmax":
        min_val = params["min_val"]
        max_val = params["max_val"]
        return normalized_image * (max_val - min_val) + min_val
    
    elif method == "percentile":
        low_val = params["low_val"]
        high_val = params["high_val"]
        return normalized_image * (high_val - low_val) + low_val
    
    elif method == "zscore":
        # This is approximate since z-score normalization loses information
        return normalized_image
    
    else:
        return normalized_image


def create_synthetic_bayer(
    rgb_image: np.ndarray,
    bayer_pattern: str = "RGGB",
    add_noise: bool = True,
    noise_std: float = 0.01
) -> np.ndarray:
    """Create synthetic Bayer image from RGB.
    
    Args:
        rgb_image: RGB image
        bayer_pattern: Bayer pattern string
        add_noise: Whether to add noise
        noise_std: Standard deviation of noise
        
    Returns:
        Synthetic Bayer image
    """
    height, width = rgb_image.shape[:2]
    bayer_image = np.zeros((height, width), dtype=rgb_image.dtype)
    
    # Get pattern array
    pattern_array = create_bayer_pattern(bayer_pattern)
    
    # Fill Bayer image
    for i in range(height):
        for j in range(width):
            channel = pattern_array[i % 2, j % 2]
            bayer_image[i, j] = rgb_image[i, j, channel]
    
    # Add noise if requested
    if add_noise:
        noise = np.random.normal(0, noise_std, bayer_image.shape)
        bayer_image = bayer_image + noise.astype(bayer_image.dtype)
        bayer_image = np.clip(bayer_image, 0, 1)
    
    return bayer_image


def save_image_with_metadata(
    image: np.ndarray,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None,
    format: str = "png"
) -> None:
    """Save image with metadata.
    
    Args:
        image: Image to save
        filepath: Output filepath
        metadata: Metadata dictionary
        format: Image format
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert to uint8 for saving
    if image.dtype != np.uint8:
        if np.max(image) <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image_uint8 = image
    
    # Save based on format
    if format.lower() in ["png", "jpg", "jpeg"]:
        try:
            from PIL import Image
            if len(image_uint8.shape) == 3:
                pil_image = Image.fromarray(image_uint8)
            else:
                pil_image = Image.fromarray(image_uint8, mode="L")
            pil_image.save(filepath)
        except ImportError:
            # Fallback to basic saving
            np.save(filepath.replace(f".{format}", ".npy"), image)
    else:
        # Save as numpy array
        np.save(filepath, image)
    
    # Save metadata if provided
    if metadata is not None:
        import json
        metadata_path = filepath.replace(f".{format}", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)


def load_image_with_metadata(
    filepath: str
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Load image with metadata.
    
    Args:
        filepath: Image filepath
        
    Returns:
        Tuple of (image, metadata)
    """
    # Load image
    if filepath.endswith(".npy"):
        image = np.load(filepath)
    else:
        try:
            from PIL import Image
            pil_image = Image.open(filepath)
            image = np.array(pil_image)
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
        except ImportError:
            image = np.load(filepath.replace(".png", ".npy").replace(".jpg", ".npy"))
    
    # Load metadata
    metadata = None
    for ext in [".png", ".jpg", ".jpeg", ".npy"]:
        metadata_path = filepath.replace(ext, "_metadata.json")
        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                break
            except Exception:
                pass
    
    return image, metadata


def calculate_image_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate image quality metrics.
    
    Args:
        predicted: Predicted image
        target: Target image
        mask: Optional mask for valid pixels
        
    Returns:
        Dictionary of metrics
    """
    if mask is not None:
        predicted = predicted[mask]
        target = target[mask]
    
    # Flatten for calculations
    pred_flat = predicted.flatten()
    target_flat = target.flatten()
    
    # MSE
    mse = np.mean((pred_flat - target_flat) ** 2)
    
    # PSNR
    if mse > 0:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    else:
        psnr = float("inf")
    
    # MAE
    mae = np.mean(np.abs(pred_flat - target_flat))
    
    # SSIM (simplified)
    try:
        from skimage.metrics import structural_similarity
        if len(predicted.shape) == 3:
            ssim = structural_similarity(predicted, target, multichannel=True, channel_axis=2)
        else:
            ssim = structural_similarity(predicted, target)
    except ImportError:
        ssim = 0.0
    
    return {
        "mse": float(mse),
        "psnr": float(psnr),
        "mae": float(mae),
        "ssim": float(ssim),
    }
