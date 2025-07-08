"""MIT-Adobe FiveK dataset preprocessing and handling."""

from __future__ import annotations

import os
import json
import requests
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import rawpy
from PIL import Image
import cv2
from tqdm import tqdm

from .base import L3DataBase
from ..color import BT2020Converter
from ..utils import create_bayer_pattern, extract_patches


class FiveKDataset(L3DataBase):
    """MIT-Adobe FiveK dataset handler with Bayer extraction and BT.2020 conversion."""
    
    FIVEK_URL_BASE = "https://data.csail.mit.edu/graphics/fivek/"
    RAW_SUBDIR = "raw_photos"
    RETOUCHED_SUBDIR = "retouched"
    
    def __init__(
        self,
        dataset_path: str,
        subset: str = "train",
        download: bool = True,
        max_images: Optional[int] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
        expert_choice: str = "C",  # A, B, C, D, E
        name: str = "FiveK Dataset"
    ):
        """Initialize FiveK dataset.
        
        Args:
            dataset_path: Path to store/load dataset
            subset: Dataset subset (train, val, test)
            download: Whether to download if not present
            max_images: Maximum number of images to load
            target_resolution: Target resolution (width, height) for resizing
            expert_choice: Expert retouching choice (A-E)
            name: Dataset name
        """
        super().__init__(name)
        
        self.dataset_path = Path(dataset_path)
        self.subset = subset
        self.max_images = max_images
        self.target_resolution = target_resolution
        self.expert_choice = expert_choice.upper()
        
        # Create dataset directory
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Dataset splits (standard split from literature)
        self.splits = {
            "train": list(range(1, 4001)),  # Images 1-4000
            "val": list(range(4001, 4501)),  # Images 4001-4500
            "test": list(range(4501, 5001)),  # Images 4501-5000
        }
        
        # Initialize color converter
        self.color_converter = BT2020Converter()
        
        # Download and prepare data
        if download:
            self._download_and_prepare()
        
        # Load image lists
        self._load_image_lists()
    
    def _download_and_prepare(self) -> None:
        """Download and prepare FiveK dataset."""
        print(f"Preparing FiveK dataset at {self.dataset_path}")
        
        # Create subdirectories
        raw_dir = self.dataset_path / self.RAW_SUBDIR
        retouched_dir = self.dataset_path / self.RETOUCHED_SUBDIR
        raw_dir.mkdir(exist_ok=True)
        retouched_dir.mkdir(exist_ok=True)
        
        # Download metadata if not present
        metadata_file = self.dataset_path / "metadata.json"
        if not metadata_file.exists():
            self._download_metadata()
        
        # Check if we need to download images
        image_ids = self.splits[self.subset]
        if self.max_images:
            image_ids = image_ids[:self.max_images]
        
        missing_raw = []
        missing_retouched = []
        
        for img_id in image_ids:
            raw_file = raw_dir / f"{img_id:05d}.dng"
            retouched_file = retouched_dir / f"{img_id:05d}_{self.expert_choice}.jpg"
            
            if not raw_file.exists():
                missing_raw.append(img_id)
            if not retouched_file.exists():
                missing_retouched.append(img_id)
        
        # Download missing files
        if missing_raw:
            print(f"Downloading {len(missing_raw)} raw files...")
            self._download_raw_files(missing_raw)
        
        if missing_retouched:
            print(f"Downloading {len(missing_retouched)} retouched files...")
            self._download_retouched_files(missing_retouched)
    
    def _download_metadata(self) -> None:
        """Download dataset metadata."""
        # This is a placeholder - in reality you'd download from the actual FiveK site
        # or use a pre-prepared metadata file
        metadata = {
            "description": "MIT-Adobe FiveK Dataset",
            "expert_choices": ["A", "B", "C", "D", "E"],
            "total_images": 5000,
            "splits": self.splits,
        }
        
        with open(self.dataset_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _download_raw_files(self, image_ids: List[int]) -> None:
        """Download raw DNG files."""
        # Placeholder implementation
        # In practice, you'd download from the actual FiveK dataset
        print("Note: Actual download not implemented. Please manually download FiveK raw files.")
        print(f"Expected files: {[f'{img_id:05d}.dng' for img_id in image_ids[:5]]}...")
    
    def _download_retouched_files(self, image_ids: List[int]) -> None:
        """Download retouched JPG files."""
        # Placeholder implementation
        print("Note: Actual download not implemented. Please manually download FiveK retouched files.")
        print(f"Expected files: {[f'{img_id:05d}_{self.expert_choice}.jpg' for img_id in image_ids[:5]]}...")
    
    def _load_image_lists(self) -> None:
        """Load lists of available images."""
        raw_dir = self.dataset_path / self.RAW_SUBDIR
        retouched_dir = self.dataset_path / self.RETOUCHED_SUBDIR
        
        # Find available image pairs
        available_pairs = []
        image_ids = self.splits[self.subset]
        
        if self.max_images:
            image_ids = image_ids[:self.max_images]
        
        for img_id in image_ids:
            raw_file = raw_dir / f"{img_id:05d}.dng"
            retouched_file = retouched_dir / f"{img_id:05d}_{self.expert_choice}.jpg"
            
            if raw_file.exists() and retouched_file.exists():
                available_pairs.append((str(raw_file), str(retouched_file)))
        
        self.image_pairs = available_pairs
        print(f"Found {len(self.image_pairs)} image pairs for {self.subset} set")
    
    def __len__(self) -> int:
        """Return number of image pairs."""
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Get a single image pair.
        
        Args:
            idx: Image index
            
        Returns:
            Tuple of (raw_bayer, target_rgb, metadata)
        """
        if idx >= len(self.image_pairs):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        raw_path, retouched_path = self.image_pairs[idx]
        
        # Load and process raw image
        raw_bayer, raw_metadata = self._load_and_process_raw(raw_path)
        
        # Load and process target image
        target_rgb = self._load_and_process_target(retouched_path, raw_metadata)
        
        # Create metadata
        metadata = {
            "raw_path": raw_path,
            "retouched_path": retouched_path,
            "expert_choice": self.expert_choice,
            "raw_metadata": raw_metadata,
            "image_id": Path(raw_path).stem,
        }
        
        return raw_bayer, target_rgb, metadata
    
    def _load_and_process_raw(self, raw_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load and process raw DNG file to extract Bayer mosaic.
        
        Args:
            raw_path: Path to raw DNG file
            
        Returns:
            Tuple of (bayer_array, metadata)
        """
        with rawpy.imread(raw_path) as raw:
            # Extract raw Bayer data
            bayer_data = raw.raw_image.copy()
            
            # Get metadata
            metadata = {
                "width": raw.sizes.width,
                "height": raw.sizes.height,
                "raw_width": raw.sizes.raw_width,
                "raw_height": raw.sizes.raw_height,
                "bayer_pattern": self._get_bayer_pattern(raw),
                "black_level": getattr(raw, 'black_level_per_channel', [0, 0, 0, 0]),
                "white_level": getattr(raw, 'white_level', 65535),
                "color_matrix": getattr(raw, 'color_matrix', None),
                "camera_whitebalance": getattr(raw, 'camera_whitebalance', [1, 1, 1, 1]),
            }
            
            # Crop to visible area
            visible_area = raw.sizes
            bayer_data = bayer_data[
                visible_area.top_margin:visible_area.top_margin + visible_area.height,
                visible_area.left_margin:visible_area.left_margin + visible_area.width
            ]
            
            # Normalize to [0, 1] range
            bayer_data = bayer_data.astype(np.float32)
            bayer_data = (bayer_data - metadata["black_level"][0]) / (metadata["white_level"] - metadata["black_level"][0])
            bayer_data = np.clip(bayer_data, 0, 1)
            
            # Resize if target resolution specified
            if self.target_resolution is not None:
                bayer_data = cv2.resize(bayer_data, self.target_resolution, interpolation=cv2.INTER_LINEAR)
                metadata["resized_to"] = self.target_resolution
        
        return bayer_data, metadata
    
    def _load_and_process_target(self, retouched_path: str, raw_metadata: Dict[str, Any]) -> np.ndarray:
        """Load and process target retouched image.
        
        Args:
            retouched_path: Path to retouched JPG file
            raw_metadata: Metadata from corresponding raw file
            
        Returns:
            Target RGB image in linear BT.2020 space
        """
        # Load retouched image
        target_image = Image.open(retouched_path)
        target_rgb = np.array(target_image).astype(np.float32) / 255.0
        
        # Resize to match raw if needed
        target_size = (raw_metadata["width"], raw_metadata["height"])
        if self.target_resolution is not None:
            target_size = self.target_resolution
        
        if target_rgb.shape[:2] != target_size[::-1]:  # Note: cv2 uses (width, height), numpy uses (height, width)
            target_rgb = cv2.resize(target_rgb, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert from sRGB to linear BT.2020
        target_rgb = self.color_converter.srgb_to_linear_bt2020(target_rgb)
        
        return target_rgb
    
    def _get_bayer_pattern(self, raw: rawpy.RawPy) -> str:
        """Extract Bayer pattern from raw image.
        
        Args:
            raw: rawpy RawPy object
            
        Returns:
            Bayer pattern string (e.g., 'RGGB')
        """
        # Get color description
        color_desc = raw.color_desc
        
        # Map to standard patterns
        pattern_map = {
            b'RGBG': 'RGGB',
            b'BGRG': 'BGGR',
            b'GRBG': 'GRBG',
            b'GBRG': 'GBRG',
        }
        
        return pattern_map.get(color_desc, 'RGGB')
    
    def get_batch(
        self,
        batch_size: int,
        start_idx: int = 0,
        shuffle: bool = False
    ) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
        """Get a batch of image pairs.
        
        Args:
            batch_size: Number of images in batch
            start_idx: Starting index
            shuffle: Whether to shuffle indices
            
        Returns:
            List of (raw_bayer, target_rgb, metadata) tuples
        """
        indices = list(range(start_idx, min(start_idx + batch_size, len(self))))
        
        if shuffle:
            np.random.shuffle(indices)
        
        return [self[idx] for idx in indices]
    
    def create_training_data(
        self,
        patch_size: Tuple[int, int] = (64, 64),
        stride: int = 32,
        max_patches_per_image: int = 100
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create training patches from the dataset.
        
        Args:
            patch_size: Size of patches to extract
            stride: Stride for patch extraction
            max_patches_per_image: Maximum patches per image
            
        Returns:
            List of (raw_patch, target_patch) pairs
        """
        training_patches = []
        
        print(f"Extracting training patches from {len(self)} images...")
        
        for idx in tqdm(range(len(self))):
            raw_bayer, target_rgb, metadata = self[idx]
            
            # Extract patches
            raw_patches = extract_patches(
                raw_bayer, patch_size, stride, max_patches_per_image
            )
            target_patches = extract_patches(
                target_rgb, patch_size, stride, max_patches_per_image
            )
            
            # Add to training data
            for raw_patch, target_patch in zip(raw_patches, target_patches):
                training_patches.append((raw_patch, target_patch))
        
        print(f"Extracted {len(training_patches)} training patches")
        return training_patches
    
    def data_get(
        self,
        n_img: int = 1,
        start_idx: int = 0
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
        """Get multiple images (compatible with L3 interface).
        
        Args:
            n_img: Number of images to get
            start_idx: Starting index
            
        Returns:
            Tuple of (raw_images, target_images, metadata_list)
        """
        raw_images = []
        target_images = []
        metadata_list = []
        
        for i in range(n_img):
            idx = start_idx + i
            if idx >= len(self):
                break
            
            raw_bayer, target_rgb, metadata = self[idx]
            raw_images.append(raw_bayer)
            target_images.append(target_rgb)
            metadata_list.append(metadata)
        
        return raw_images, target_images, metadata_list
    
    @property
    def cfa(self) -> np.ndarray:
        """Get CFA pattern for the dataset."""
        # Return a default RGGB pattern
        # In practice, this should be determined from the actual raw files
        return create_bayer_pattern("RGGB")
