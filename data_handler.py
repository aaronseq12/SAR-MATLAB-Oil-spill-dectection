"""
Data handling and preprocessing for SAR oil spill detection
Supports various SAR image formats and provides data augmentation capabilities
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import logging
from dataclasses import dataclass
import json

import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage import io, transform, filters
from skimage.segmentation import slic
import albumentations as A

logger = logging.getLogger(__name__)

@dataclass
class DatasetStatistics:
    """Container for dataset statistics"""
    total_images: int
    images_with_land: int
    images_without_land: int
    average_size: Tuple[int, int]
    oil_spill_percentage: float
    class_distribution: Dict[str, int]

class SARImagePreprocessor:
    """Advanced preprocessing for SAR images"""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        self.augmentation_pipeline = self._create_augmentation_pipeline()
    
    def _create_augmentation_pipeline(self):
        """Create data augmentation pipeline for SAR images"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(
                alpha=1, sigma=50, alpha_affine=50, 
                interpolation=1, border_mode=1, p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.3
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
        ], additional_targets={'mask': 'mask'})
    
    def preprocess_sar_image(self, sar_image: np.ndarray, 
                           normalize: bool = True, 
                           enhance_contrast: bool = True) -> np.ndarray:
        """
        Comprehensive preprocessing for SAR images
        
        Args:
            sar_image: Input SAR image
            normalize: Whether to normalize intensity values
            enhance_contrast: Whether to apply contrast enhancement
            
        Returns:
            Preprocessed SAR image
        """
        processed_image = sar_image.copy()
        
        # Convert to grayscale if needed
        if len(processed_image.shape) == 3:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
        
        # Resize to target size
        processed_image = cv2.resize(
            processed_image, self.target_size, 
            interpolation=cv2.INTER_LANCZOS4
        )
        
        # Despeckling (reduce SAR noise)
        processed_image = self._despeckle_sar_image(processed_image)
        
        # Contrast enhancement
        if enhance_contrast:
            processed_image = self._enhance_sar_contrast(processed_image)
        
        # Normalization
        if normalize:
            processed_image = self._normalize_sar_intensity(processed_image)
        
        return processed_image
    
    def _despeckle_sar_image(self, image: np.ndarray) -> np.ndarray:
        """Apply despeckling filter to reduce SAR noise"""
        # Lee filter for SAR despeckling
        return cv2.bilateralFilter(image.astype(np.float32), 9, 75, 75)
    
    def _enhance_sar_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast for better oil spill visibility"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image.astype(np.uint8))
    
    def _normalize_sar_intensity(self, image: np.ndarray) -> np.ndarray:
        """Normalize SAR intensity values"""
        # Robust normalization using percentiles
        lower_percentile = np.percentile(image, 2)
        upper_percentile = np.percentile(image, 98)
        
        normalized_image = np.clip(
            (image - lower_percentile) / (upper_percentile - lower_percentile),
            0, 1
        )
        
        return normalized_image.astype(np.float32)
    
    def augment_image_pair(self, image: np.ndarray, 
                          mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to image-mask pair"""
        augmented = self.augmentation_pipeline(image=image, mask=mask)
        return augmented['image'], augmented['mask']

class SARDatasetHandler:
    """Comprehensive handler for SAR oil spill datasets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessor = SARImagePreprocessor(
            target_size=tuple(config['image_processing']['target_size'])
        )
        
        # Dataset structure
        self.dataset_root = None
        self.images_with_land = []
        self.images_without_land = []
        self.image_paths = {}
        self.mask_paths = {}
        
        # Supported file formats
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        
        logger.info("SAR Dataset Handler initialized")
    
    def load_dataset(self, dataset_root_path: str) -> bool:
        """
        Load SAR dataset from directory structure
        
        Expected structure:
        dataset_root/
        ├── train/
        │   ├── images/
        │   ├── images_with_land/
        │   ├── labels/
        │   └── labels_with_land/
        
        Args:
            dataset_root_path: Root directory of dataset
            
        Returns:
            Success status
        """
        self.dataset_root = Path(dataset_root_path)
        
        if not self.dataset_root.exists():
            logger.error(f"Dataset root does not exist: {dataset_root_path}")
            return False
        
        try:
            # Load images without land
            images_dir = self.dataset_root / "train" / "images"
            labels_dir = self.dataset_root / "train" / "labels"
            
            if images_dir.exists() and labels_dir.exists():
                self.images_without_land = self._load_image_list(images_dir, labels_dir, False)
            
            # Load images with land
            images_with_land_dir = self.dataset_root / "train" / "images_with_land"
            labels_with_land_dir = self.dataset_root / "train" / "labels_with_land"
            
            if images_with_land_dir.exists() and labels_with_land_dir.exists():
                self.images_with_land = self._load_image_list(
                    images_with_land_dir, labels_with_land_dir, True
                )
            
            total_images = len(self.images_without_land) + len(self.images_with_land)
            if total_images == 0:
                logger.error("No valid image pairs found in dataset")
                return False
            
            logger.info(f"Loaded {total_images} image pairs successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return False
    
    def _load_image_list(self, images_dir: Path, labels_dir: Path, 
                        has_land: bool) -> List[str]:
        """Load list of valid image-mask pairs"""
        valid_pairs = []
        
        for image_file in images_dir.iterdir():
            if image_file.suffix.lower() not in self.supported_formats:
                continue
            
            # Find corresponding mask file
            mask_file = labels_dir / image_file.name
            if not mask_file.exists():
                logger.warning(f"No mask found for {image_file.name}")
                continue
            
            # Store paths
            image_key = f"{image_file.stem}_{has_land}"
            self.image_paths[image_key] = str(image_file)
            self.mask_paths[image_key] = str(mask_file)
            
            valid_pairs.append(image_file.name)
        
        return valid_pairs
    
    def get_available_images(self, has_land: bool) -> List[str]:
        """Get list of available images for specified category"""
        return self.images_with_land if has_land else self.images_without_land
    
    def load_image_pair(self, image_name: str, 
                       has_land: bool) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load and preprocess image-mask pair
        
        Args:
            image_name: Name of the image file
            has_land: Whether image contains land
            
        Returns:
            Tuple of (preprocessed_image, ground_truth_mask)
        """
        try:
            image_key = f"{Path(image_name).stem}_{has_land}"
            
            if image_key not in self.image_paths:
                logger.error(f"Image not found: {image_name}")
                return None, None
            
            # Load raw image and mask
            raw_image = io.imread(self.image_paths[image_key])
            raw_mask = io.imread(self.mask_paths[image_key])
            
            # Preprocess image
            processed_image = self.preprocessor.preprocess_sar_image(
                raw_image,
                normalize=self.config['image_processing']['normalize']
            )
            
            # Process mask
            processed_mask = self._process_ground_truth_mask(raw_mask, has_land)
            
            return processed_image, processed_mask
            
        except Exception as e:
            logger.error(f"Error loading image pair {image_name}: {str(e)}")
            return None, None
    
    def _process_ground_truth_mask(self, raw_mask: np.ndarray, 
                                  has_land: bool) -> np.ndarray:
        """
        Process ground truth mask to standard format
        
        Mask classes:
        - 0: Sea/Water (black)
        - 1: Oil spill (white/cyan in visualization)  
        - 2: Land (green in visualization, only if has_land=True)
        """
        if len(raw_mask.shape) == 3:
            raw_mask = cv2.cvtColor(raw_mask, cv2.COLOR_RGB2GRAY)
        
        # Resize to target size
        processed_mask = cv2.resize(
            raw_mask, self.preprocessor.target_size,
            interpolation=cv2.INTER_NEAREST
        )
        
        # Convert to class indices
        if has_land:
            # Three classes: sea (0), oil spill (1), land (2)
            mask_normalized = processed_mask / 255.0
            class_mask = np.zeros_like(processed_mask, dtype=np.uint8)
            
            # Oil spill detection (typically dark areas)
            class_mask[mask_normalized < 0.3] = 1  # Oil spill
            class_mask[mask_normalized > 0.7] = 2  # Land
            # Sea remains 0
        else:
            # Two classes: sea (0), oil spill (1)
            mask_normalized = processed_mask / 255.0
            class_mask = (mask_normalized < 0.5).astype(np.uint8)
        
        return class_mask
    
    def get_dataset_statistics(self) -> DatasetStatistics:
        """Calculate and return comprehensive dataset statistics"""
        total_images = len(self.images_without_land) + len(self.images_with_land)
        
        # Calculate oil spill coverage percentage
        oil_spill_pixels = 0
        total_pixels = 0
        
        # Sample a subset for statistics calculation
        sample_images = (self.images_without_land[:10] + 
                        self.images_with_land[:10])
        
        for image_name in sample_images:
            has_land = image_name in self.images_with_land
            _, mask = self.load_image_pair(image_name, has_land)
            
            if mask is not None:
                oil_spill_pixels += np.sum(mask == 1)
                total_pixels += mask.size
        
        oil_spill_percentage = (oil_spill_pixels / total_pixels * 100) if total_pixels > 0 else 0
        
        return DatasetStatistics(
            total_images=total_images,
            images_with_land=len(self.images_with_land),
            images_without_land=len(self.images_without_land),
            average_size=self.preprocessor.target_size,
            oil_spill_percentage=oil_spill_percentage,
            class_distribution={
                'sea': total_pixels - oil_spill_pixels,
                'oil_spill': oil_spill_pixels
            }
        )
    
    def create_data_generators(self, batch_size: int = 8, 
                             validation_split: float = 0.2):
        """
        Create TensorFlow data generators for training
        
        Args:
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            
        Returns:
            Tuple of (train_generator, validation_generator)
        """
        # Collect all image paths
        all_image_paths = []
        all_mask_paths = []
        
        for image_name in self.images_without_land:
            image_key = f"{Path(image_name).stem}_False"
            all_image_paths.append(self.image_paths[image_key])
            all_mask_paths.append(self.mask_paths[image_key])
        
        for image_name in self.images_with_land:
            image_key = f"{Path(image_name).stem}_True"
            all_image_paths.append(self.image_paths[image_key])
            all_mask_paths.append(self.mask_paths[image_key])
        
        # Split data
        train_images, val_images, train_masks, val_masks = train_test_split(
            all_image_paths, all_mask_paths, 
            test_size=validation_split, random_state=42
        )
        
        # Create TensorFlow datasets
        train_dataset = self._create_tf_dataset(train_images, train_masks, batch_size, True)
        val_dataset = self._create_tf_dataset(val_images, val_masks, batch_size, False)
        
        return train_dataset, val_dataset
    
    def _create_tf_dataset(self, image_paths: List[str], mask_paths: List[str], 
                          batch_size: int, augment: bool = False):
        """Create TensorFlow dataset from file paths"""
        def load_and_preprocess(image_path, mask_path):
            # Load image
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=1)
            image = tf.image.resize(image, self.preprocessor.target_size)
            image = tf.cast(image, tf.float32) / 255.0
            
            # Load mask
            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_image(mask, channels=1)
            mask = tf.image.resize(mask, self.preprocessor.target_size, method='nearest')
            mask = tf.cast(mask, tf.uint8)
            
            return image, mask
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        if augment:
            dataset = dataset.map(self._tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _tf_augment(self, image, mask):
        """TensorFlow-compatible data augmentation"""
        # Random horizontal flip
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        
        # Random vertical flip
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
        
        # Random rotation (90 degrees)
        if tf.random.uniform(()) > 0.5:
            k = tf.random.uniform(shape=[], minval=1, maxval=4, dtype=tf.int32)
            image = tf.image.rot90(image, k)
            mask = tf.image.rot90(mask, k)
        
        return image, mask

    def export_processed_dataset(self, output_dir: str):
        """Export preprocessed dataset for external use"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export dataset statistics
        stats = self.get_dataset_statistics()
        with open(output_path / "dataset_statistics.json", 'w') as f:
            json.dump(stats.__dict__, f, indent=2)
        
        logger.info(f"Dataset exported to {output_path}")
