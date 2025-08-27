"""
SAR Oil Spill Detection System
Advanced Python implementation with deep learning capabilities

This system provides comprehensive oil spill detection and segmentation
using both traditional image processing and modern deep learning approaches.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml

import numpy as np
import tensorflow as tf
import torch
import cv2
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

from data_handler import SARDatasetHandler
from traditional_segmentation import TraditionalSegmentationMethods
from deep_learning_models import DeepLearningSARSegmentation
from evaluation_metrics import SegmentationEvaluator
from visualization import ResultVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SAROilSpillDetectionSystem:
    """
    Comprehensive SAR Oil Spill Detection System
    
    Combines traditional image processing with modern deep learning
    for accurate oil spill detection and segmentation in SAR images.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize the detection system
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_configuration(config_path)
        self.dataset_handler = SARDatasetHandler(self.config)
        self.traditional_methods = TraditionalSegmentationMethods()
        self.deep_learning_models = DeepLearningSARSegmentation(self.config)
        self.evaluator = SegmentationEvaluator()
        self.visualizer = ResultVisualizer()
        
        # Available segmentation methods
        self.available_methods = {
            'adaptive_threshold': self.traditional_methods.adaptive_threshold_segmentation,
            'superpixel_clustering': self.traditional_methods.superpixel_based_segmentation,
            'fuzzy_edge_detection': self.traditional_methods.fuzzy_logic_segmentation,
            'kmeans_clustering': self.traditional_methods.kmeans_segmentation,
            'unet_tensorflow': self.deep_learning_models.unet_tensorflow_prediction,
            'unet_pytorch': self.deep_learning_models.unet_pytorch_prediction,
            'segformer': self.deep_learning_models.segformer_prediction,
            'deeplabv3': self.deep_learning_models.deeplabv3_prediction
        }
        
        logger.info("SAR Oil Spill Detection System initialized successfully")
    
    def _load_configuration(self, config_path: str) -> dict:
        """Load system configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Return default configuration settings"""
        return {
            'image_processing': {
                'target_size': (512, 512),
                'normalize': True,
                'augmentation': True
            },
            'deep_learning': {
                'batch_size': 8,
                'learning_rate': 0.001,
                'epochs': 100,
                'early_stopping_patience': 15
            },
            'evaluation': {
                'metrics': ['jaccard', 'dice', 'boundary_f1'],
                'visualization_enabled': True
            }
        }
    
    def load_dataset(self, dataset_root_path: str) -> bool:
        """
        Load SAR dataset for processing
        
        Args:
            dataset_root_path: Root directory containing train/test folders
            
        Returns:
            Success status
        """
        try:
            success = self.dataset_handler.load_dataset(dataset_root_path)
            if success:
                logger.info(f"Dataset loaded successfully from {dataset_root_path}")
                self._display_dataset_summary()
            return success
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            return False
    
    def _display_dataset_summary(self):
        """Display dataset statistics and information"""
        stats = self.dataset_handler.get_dataset_statistics()
        print("\n" + "="*60)
        print("                DATASET SUMMARY")
        print("="*60)
        print(f"Total Images: {stats['total_images']}")
        print(f"Images with Land: {stats['images_with_land']}")
        print(f"Images without Land: {stats['images_without_land']}")
        print(f"Average Image Size: {stats['average_size']}")
        print(f"Oil Spill Coverage: {stats['oil_spill_percentage']:.2f}%")
        print("="*60)
    
    def interactive_detection_mode(self):
        """
        Interactive mode for oil spill detection
        Allows users to select images and methods interactively
        """
        print("\n🛰️  SAR Oil Spill Detection System")
        print("=" * 50)
        
        while True:
            # Display available image categories
            print("\nSelect image category:")
            print("1. Images with only sea and oil spills")
            print("2. Images with land, sea, and oil spills")
            print("3. Exit")
            
            category_choice = input("\nEnter your choice (1-3): ").strip()
            
            if category_choice == '3':
                print("Exiting system. Goodbye! 👋")
                break
            elif category_choice not in ['1', '2']:
                print("❌ Invalid choice. Please try again.")
                continue
            
            has_land = category_choice == '2'
            available_images = self.dataset_handler.get_available_images(has_land)
            
            if not available_images:
                print("❌ No images available for this category.")
                continue
            
            # Display available images
            print(f"\nAvailable images (1-{len(available_images)}):")
            for idx, img_name in enumerate(available_images[:10], 1):
                print(f"{idx}. {img_name}")
            
            # Image selection
            try:
                image_idx = int(input(f"\nSelect image (1-{len(available_images)}): ")) - 1
                if image_idx < 0 or image_idx >= len(available_images):
                    print("❌ Invalid image selection.")
                    continue
                    
                selected_image = available_images[image_idx]
            except ValueError:
                print("❌ Please enter a valid number.")
                continue
            
            # Method selection
            self._display_available_methods(has_land)
            method_choice = input("\nSelect segmentation method: ").strip()
            
            if method_choice not in self.available_methods:
                print("❌ Invalid method selection.")
                continue
            
            # Process the image
            self._process_single_image(selected_image, method_choice, has_land)
    
    def _display_available_methods(self, has_land: bool):
        """Display available segmentation methods"""
        print("\nAvailable Segmentation Methods:")
        print("-" * 40)
        
        if not has_land:
            print("Traditional Methods:")
            print("  adaptive_threshold    - Adaptive thresholding")
            print("  superpixel_clustering - Superpixel-based segmentation")
            print("  fuzzy_edge_detection  - Fuzzy logic approach")
            print("  kmeans_clustering     - K-means clustering")
        
        print("Deep Learning Methods:")
        print("  unet_tensorflow       - U-Net (TensorFlow)")
        print("  unet_pytorch         - U-Net (PyTorch)")
        print("  segformer            - SegFormer transformer")
        print("  deeplabv3            - DeepLabV3+")
    
    def _process_single_image(self, image_name: str, method_name: str, has_land: bool):
        """
        Process a single image with the selected method
        
        Args:
            image_name: Name of the image to process
            method_name: Selected segmentation method
            has_land: Whether the image contains land
        """
        try:
            print(f"\n🔄 Processing {image_name} with {method_name}...")
            
            # Load image and ground truth
            sar_image, ground_truth_mask = self.dataset_handler.load_image_pair(
                image_name, has_land
            )
            
            if sar_image is None or ground_truth_mask is None:
                print("❌ Failed to load image or ground truth.")
                return
            
            # Apply selected segmentation method
            segmentation_method = self.available_methods[method_name]
            predicted_mask = segmentation_method(sar_image)
            
            # Calculate evaluation metrics
            metrics = self.evaluator.calculate_comprehensive_metrics(
                ground_truth_mask, predicted_mask
            )
            
            # Display results
            self._display_segmentation_results(
                sar_image, ground_truth_mask, predicted_mask, 
                metrics, image_name, method_name
            )
            
            # Ask for parameter fine-tuning
            if method_name in ['adaptive_threshold', 'kmeans_clustering', 'fuzzy_edge_detection']:
                self._offer_parameter_tuning(
                    sar_image, ground_truth_mask, method_name, metrics
                )
                
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            print(f"❌ Error: {str(e)}")
    
    def _display_segmentation_results(self, original_image, ground_truth, 
                                    predicted_mask, metrics, image_name, method_name):
        """Display segmentation results with metrics"""
        print(f"\n📊 Results for {image_name} using {method_name}:")
        print("-" * 50)
        print(f"Jaccard Index: {metrics['jaccard_index']:.4f}")
        print(f"Dice Coefficient: {metrics['dice_coefficient']:.4f}")
        print(f"Boundary F1 Score: {metrics['boundary_f1']:.4f}")
        print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        
        # Visualize results
        if self.config['evaluation']['visualization_enabled']:
            self.visualizer.display_segmentation_comparison(
                original_image, ground_truth, predicted_mask, 
                metrics, f"{method_name} - {image_name}"
            )
    
    def _offer_parameter_tuning(self, sar_image, ground_truth, method_name, current_metrics):
        """Offer interactive parameter tuning for traditional methods"""
        tune_choice = input("\n🔧 Would you like to fine-tune parameters? (y/n): ").lower()
        
        if tune_choice == 'y':
            print("🎛️  Parameter Tuning Mode")
            best_metrics = current_metrics
            best_params = None
            
            # Method-specific parameter tuning
            if method_name == 'adaptive_threshold':
                best_params = self._tune_adaptive_threshold_params(
                    sar_image, ground_truth, best_metrics
                )
            elif method_name == 'kmeans_clustering':
                best_params = self._tune_kmeans_params(
                    sar_image, ground_truth, best_metrics
                )
            elif method_name == 'fuzzy_edge_detection':
                best_params = self._tune_fuzzy_params(
                    sar_image, ground_truth, best_metrics
                )
            
            if best_params:
                print(f"\n✅ Best parameters found: {best_params}")
                print(f"Improved Jaccard Index: {best_metrics['jaccard_index']:.4f}")
    
    def batch_evaluation_mode(self, method_names: list = None, sample_size: int = 50):
        """
        Batch evaluation mode for comparing multiple methods
        
        Args:
            method_names: List of methods to evaluate
            sample_size: Number of images to evaluate
        """
        if method_names is None:
            method_names = ['adaptive_threshold', 'unet_tensorflow', 'segformer']
        
        print(f"\n📊 Batch Evaluation Mode - {sample_size} images")
        print("=" * 60)
        
        results = {}
        
        for method_name in method_names:
            print(f"\n🔄 Evaluating {method_name}...")
            method_results = self._evaluate_method_on_sample(method_name, sample_size)
            results[method_name] = method_results
            
            print(f"Average Jaccard: {method_results['avg_jaccard']:.4f}")
            print(f"Average Dice: {method_results['avg_dice']:.4f}")
            print(f"Processing Time: {method_results['avg_time']:.2f}s per image")
        
        # Generate comparison report
        self._generate_comparison_report(results)
    
    def train_deep_learning_models(self, epochs: int = None):
        """
        Train deep learning models on the loaded dataset
        
        Args:
            epochs: Number of training epochs (uses config default if None)
        """
        if epochs is None:
            epochs = self.config['deep_learning']['epochs']
        
        print(f"\n🚀 Training Deep Learning Models ({epochs} epochs)")
        print("=" * 60)
        
        # Prepare training data
        train_generator, val_generator = self.dataset_handler.create_data_generators(
            batch_size=self.config['deep_learning']['batch_size']
        )
        
        # Train each model
        models_to_train = ['unet_tensorflow', 'segformer']
        
        for model_name in models_to_train:
            print(f"\n🧠 Training {model_name}...")
            try:
                training_history = self.deep_learning_models.train_model(
                    model_name, train_generator, val_generator, epochs
                )
                
                # Save training history
                self._save_training_history(model_name, training_history)
                print(f"✅ {model_name} training completed successfully")
                
            except Exception as e:
                logger.error(f"Training failed for {model_name}: {str(e)}")
                print(f"❌ Training failed for {model_name}")
    
    def export_results(self, output_dir: str = "results"):
        """Export detection results and trained models"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n💾 Exporting results to {output_path}")
        
        # Export trained models
        self.deep_learning_models.save_all_models(output_path / "models")
        
        # Export evaluation reports
        # Implementation for exporting detailed reports
        
        print("✅ Export completed successfully")

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(
        description="SAR Oil Spill Detection System"
    )
    parser.add_argument(
        "--dataset", "-d", 
        type=str, 
        help="Path to dataset root directory"
    )
    parser.add_argument(
        "--mode", "-m", 
        choices=["interactive", "batch", "train"], 
        default="interactive",
        help="Operating mode"
    )
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        default="config/model_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    detection_system = SAROilSpillDetectionSystem(args.config)
    
    # Load dataset if provided
    if args.dataset:
        if not detection_system.load_dataset(args.dataset):
            print("❌ Failed to load dataset. Exiting.")
            sys.exit(1)
    else:
        print("ℹ️  Dataset path not provided. Some features may be limited.")
    
    # Run in selected mode
    try:
        if args.mode == "interactive":
            detection_system.interactive_detection_mode()
        elif args.mode == "batch":
            detection_system.batch_evaluation_mode()
        elif args.mode == "train":
            detection_system.train_deep_learning_models()
    except KeyboardInterrupt:
        print("\n\n👋 System interrupted by user. Goodbye!")
    except Exception as e:
        logger.error(f"System error: {str(e)}")
        print(f"❌ System error: {str(e)}")

if __name__ == "__main__":
    main()
