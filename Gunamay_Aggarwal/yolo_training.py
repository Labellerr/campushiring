
"""
YOLO-Seg Training Script for Vehicle/Pedestrian Detection
Labellerr AI Internship Assignment
"""

import os
import yaml
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOSegTrainer:
    def __init__(self, model_name='yolov8n-seg.pt', data_dir='dataset'):
        """
        Initialize YOLO Segmentation trainer

        Args:
            model_name: Pretrained model to use (yolov8n-seg.pt, yolov8s-seg.pt, etc.)
            data_dir: Directory containing the dataset
        """
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.model = None

    def setup_dataset_yaml(self, classes):
        """Create dataset configuration YAML file"""
        dataset_config = {
            'path': str(self.data_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(classes),
            'names': classes
        }

        yaml_path = self.data_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        logger.info(f"Dataset YAML created at {yaml_path}")
        return yaml_path

    def prepare_directory_structure(self):
        """Create necessary directory structure for YOLO training"""
        directories = [
            self.data_dir / 'train' / 'images',
            self.data_dir / 'train' / 'labels',
            self.data_dir / 'val' / 'images',
            self.data_dir / 'val' / 'labels',
            self.data_dir / 'test' / 'images',
            self.data_dir / 'test' / 'labels'
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info("Directory structure created")

    def train_model(self, epochs=100, imgsz=640, batch_size=16, lr0=0.01, 
                   patience=50, save_period=10):
        """
        Train YOLO segmentation model

        Args:
            epochs: Number of training epochs
            imgsz: Input image size
            batch_size: Batch size for training
            lr0: Initial learning rate
            patience: Early stopping patience
            save_period: Save model every N epochs
        """
        # Check if GPU is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")

        # Initialize model
        self.model = YOLO(self.model_name)

        # Setup dataset YAML
        classes = ['vehicle', 'pedestrian']  # Modify as needed
        yaml_path = self.setup_dataset_yaml(classes)

        # Training arguments
        train_args = {
            'data': str(yaml_path),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'lr0': lr0,
            'patience': patience,
            'save_period': save_period,
            'device': device,
            'project': 'runs/segment',
            'name': 'yolo_seg_exp',
            'exist_ok': True,
            'verbose': True
        }

        logger.info("Starting training...")
        results = self.model.train(**train_args)

        return results

    def evaluate_model(self, model_path=None):
        """Evaluate trained model on validation set"""
        if model_path:
            self.model = YOLO(model_path)
        elif self.model is None:
            logger.error("No model loaded. Train model first or provide model path.")
            return None

        # Evaluate on validation set
        results = self.model.val()

        # Print key metrics
        logger.info(f"mAP50: {results.box.map50:.3f}")
        logger.info(f"mAP50-95: {results.box.map:.3f}")
        logger.info(f"Precision: {results.box.mp:.3f}")
        logger.info(f"Recall: {results.box.mr:.3f}")

        return results

    def predict_and_visualize(self, image_path, model_path=None, save_path=None):
        """Make predictions on a single image and visualize results"""
        if model_path:
            self.model = YOLO(model_path)
        elif self.model is None:
            logger.error("No model loaded.")
            return None

        # Make prediction
        results = self.model(image_path)

        # Visualize results
        for r in results:
            # Plot results
            im_array = r.plot()
            im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(12, 8))
            plt.imshow(im)
            plt.axis('off')
            plt.title(f'YOLO Segmentation Results: {Path(image_path).name}')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            plt.show()

        return results

def main():
    """Main training pipeline"""
    # Initialize trainer
    trainer = YOLOSegTrainer(model_name='yolov8n-seg.pt', data_dir='dataset')

    # Prepare directory structure
    trainer.prepare_directory_structure()

    # Train model
    logger.info("Starting YOLO segmentation training...")
    results = trainer.train_model(
        epochs=100,
        imgsz=640,
        batch_size=16,
        lr0=0.01,
        patience=50
    )

    # Evaluate model
    logger.info("Evaluating trained model...")
    eval_results = trainer.evaluate_model()

    # Test prediction on sample image
    # trainer.predict_and_visualize('test_image.jpg', save_path='prediction_result.jpg')

    logger.info("Training and evaluation completed!")
    return results

if __name__ == "__main__":
    main()
