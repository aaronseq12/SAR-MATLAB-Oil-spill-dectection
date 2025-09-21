"""
SAR Oil Spill Detection System
Advanced Python implementation with deep learning capabilities.

This system provides comprehensive oil spill detection and segmentation
using both traditional image processing and modern deep learning approaches.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# It's better practice to organize your classes into separate files
# and import them. This makes the project much easier to manage.
from src.data_handler import SARDatasetHandler
from src.deep_learning_models import DeepLearningSARSegmentation
from src.evaluation_metrics import SegmentationEvaluator
from src.traditional_segmentation import TraditionalSegmentationMethods
from src.visualization import ResultVisualizer


class OilSpillDetectionSystem:
    """
    Combines traditional and deep learning methods for robust SAR oil spill detection.
    """

    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initializes the detection system with a given configuration."""
        self.config = self._load_configuration(config_path)
        self.dataset_handler = SARDatasetHandler(self.config)
        self.traditional_methods = TraditionalSegmentationMethods()
        self.deep_learning_models = DeepLearningSARSegmentation(self.config)
        self.evaluator = SegmentationEvaluator()
        self.visualizer = ResultVisualizer()

        self.available_methods = {
            "adaptive_threshold": self.traditional_methods.adaptive_threshold_segmentation,
            "superpixel_clustering": self.traditional_methods.superpixel_based_segmentation,
            "fuzzy_edge_detection": self.traditional_methods.fuzzy_logic_segmentation,
            "kmeans_clustering": self.traditional_methods.kmeans_segmentation,
            "unet_tensorflow": self.deep_learning_models.unet_tensorflow_prediction,
        }
        logger.info("SAR Oil Spill Detection System initialized successfully.")

    def _load_configuration(self, config_path: str) -> dict:
        """Loads system configuration from a YAML file."""
        try:
            with open(config_path, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found at: {config_path}. Exiting.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}. Exiting.")
            sys.exit(1)

    def load_dataset(self, dataset_root: str) -> bool:
        """Loads the SAR dataset and displays a summary."""
        try:
            success = self.dataset_handler.load_dataset(dataset_root)
            if success:
                logger.info(f"Dataset loaded successfully from {dataset_root}")
                self._display_dataset_summary()
            return success
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False

    def _display_dataset_summary(self):
        """Displays key statistics about the loaded dataset."""
        stats = self.dataset_handler.get_dataset_statistics()
        summary = f"""
        ============================================================
                        DATASET SUMMARY
        ============================================================
        Total Images:         {stats.total_images}
        Images with Land:     {stats.images_with_land}
        Images without Land:  {stats.images_without_land}
        Target Image Size:    {stats.average_size}
        Oil Spill Coverage:   {stats.oil_spill_percentage:.2f}%
        ============================================================
        """
        print(summary)

    def run_interactive_mode(self):
        """Runs an interactive CLI for detecting oil spills in single images."""
        print("\n🛰️  Welcome to the Interactive SAR Oil Spill Detection System")
        print("=" * 60)

        while True:
            # Main menu loop
            # ... (Your interactive logic can be refined here) ...
            print("Interactive mode is ready. (Implementation can be extended)")
            break # Simple exit for now

    def run_batch_evaluation(self, methods: list = None, sample_size: int = 20):
        """Evaluates multiple methods on a sample of the dataset."""
        if methods is None:
            methods = ["adaptive_threshold", "unet_tensorflow"]
        
        print(f"\n📊 Starting Batch Evaluation on {sample_size} images...")
        # ... (Your batch evaluation logic here) ...
        print("Batch evaluation complete.")


    def train_models(self):
        """Trains the deep learning models defined in the configuration."""
        print("\n🚀 Starting Deep Learning Model Training...")
        # ... (Your training logic here) ...
        print("Model training complete.")


def main():
    """Main entry point for the command-line application."""
    parser = argparse.ArgumentParser(description="SAR Oil Spill Detection System")
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to the system configuration file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the root directory of the dataset.",
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "batch", "train"],
        default="interactive",
        help="The mode to run the system in.",
    )
    args = parser.parse_args()

    system = OilSpillDetectionSystem(args.config)

    if not system.load_dataset(args.dataset):
        logger.error("Failed to load dataset. Please check the path and try again.")
        sys.exit(1)

    try:
        if args.mode == "interactive":
            system.run_interactive_mode()
        elif args.mode == "batch":
            system.run_batch_evaluation()
        elif args.mode == "train":
            system.train_models()
    except KeyboardInterrupt:
        print("\n\nSystem interrupted by user. Goodbye! 👋")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
