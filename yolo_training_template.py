
# YOLO Training and Evaluation Template
import os
from ultralytics import YOLO
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

class YOLOSegmentationTrainer:
    def __init__(self, model_name='yolov8n-seg.pt'):
        self.model_name = model_name
        self.model = None
        self.training_results = None

    def create_dataset_yaml(self, train_path, val_path, test_path, class_names):
        """Create dataset configuration file"""
        dataset_config = {
            'train': train_path,
            'val': val_path,
            'test': test_path,
            'nc': len(class_names),
            'names': class_names
        }

        with open('dataset.yaml', 'w') as f:
            yaml.dump(dataset_config, f)

        print("Dataset configuration created: dataset.yaml")
        return 'dataset.yaml'

    def train_model(self, data_config, epochs=100, imgsz=640, batch=16):
        """Train YOLO segmentation model"""
        # Load pre-trained model
        self.model = YOLO(self.model_name)

        # Start training
        print(f"Starting training with {epochs} epochs...")
        self.training_results = self.model.train(
            data=data_config,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name='vehicle_pedestrian_seg',
            exist_ok=True,
            amp=False  # Disable AMP for compatibility
        )

        print("Training completed!")
        return self.training_results

    def evaluate_model(self, data_config):
        """Evaluate trained model"""
        if self.model is None:
            print("No model loaded. Please train or load a model first.")
            return None

        # Run validation
        print("Evaluating model...")
        results = self.model.val(data=data_config)

        # Print key metrics
        print(f"\nEvaluation Results:")
        print(f"Box mAP50: {results.box.map50:.3f}")
        print(f"Box mAP50-95: {results.box.map:.3f}")
        print(f"Mask mAP50: {results.seg.map50:.3f}")
        print(f"Mask mAP50-95: {results.seg.map:.3f}")

        return results

    def run_inference(self, source_path, conf=0.25, save=True):
        """Run inference on test images"""
        if self.model is None:
            print("No model loaded. Please train or load a model first.")
            return None

        print(f"Running inference on: {source_path}")
        results = self.model.predict(
            source=source_path,
            conf=conf,
            save=save,
            project='inference_results'
        )

        return results

    def generate_confusion_matrix(self, val_results, class_names):
        """Generate and plot confusion matrix"""
        # This is a simplified version - actual implementation would depend on 
        # how YOLO stores validation results
        plt.figure(figsize=(8, 6))

        # Placeholder data - replace with actual confusion matrix from YOLO results
        cm = np.random.rand(len(class_names), len(class_names))
        cm = cm / cm.sum(axis=1, keepdims=True)  # Normalize

        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Segmentation Results')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_training_metrics(self, results_dir='runs/segment/vehicle_pedestrian_seg'):
        """Plot training metrics"""
        results_file = os.path.join(results_dir, 'results.csv')

        if os.path.exists(results_file):
            import pandas as pd
            df = pd.read_csv(results_file)

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Plot losses
            axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')
            axes[0, 0].plot(df['epoch'], df['train/seg_loss'], label='Seg Loss')
            axes[0, 0].set_title('Training Losses')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()

            # Plot mAP
            axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='Box mAP50')
            axes[0, 1].plot(df['epoch'], df['metrics/mAP50(M)'], label='Mask mAP50')
            axes[0, 1].set_title('Mean Average Precision')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('mAP')
            axes[0, 1].legend()

            # Plot precision and recall
            axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Box Precision')
            axes[1, 0].plot(df['epoch'], df['metrics/precision(M)'], label='Mask Precision')
            axes[1, 0].set_title('Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()

            axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Box Recall')
            axes[1, 1].plot(df['epoch'], df['metrics/recall(M)'], label='Mask Recall')
            axes[1, 1].set_title('Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()

            plt.tight_layout()
            plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
            plt.show()

# Usage example
if __name__ == "__main__":
    # Initialize trainer
    trainer = YOLOSegmentationTrainer('yolov8n-seg.pt')

    # Create dataset configuration
    class_names = ['vehicle', 'pedestrian']
    data_config = trainer.create_dataset_yaml(
        train_path='./dataset/train/images',
        val_path='./dataset/val/images', 
        test_path='./dataset/test/images',
        class_names=class_names
    )

    # Train model
    training_results = trainer.train_model(data_config, epochs=100)

    # Evaluate model
    evaluation_results = trainer.evaluate_model(data_config)

    # Run inference on test set
    inference_results = trainer.run_inference('./dataset/test/images')

    # Generate visualizations
    trainer.plot_training_metrics()
    trainer.generate_confusion_matrix(evaluation_results, class_names)
