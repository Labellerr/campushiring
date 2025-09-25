
"""
Dataset Preparation Script for YOLO Segmentation Training
Supports various dataset formats and creates YOLO-compatible structure
"""

import os
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import requests
import zipfile
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreparer:
    def __init__(self, output_dir: str = "dataset"):
        self.output_dir = Path(output_dir)
        self.class_names = []

    def download_sample_dataset(self):
        """Download sample vehicle/pedestrian dataset for demo purposes"""
        # This would typically download from a public dataset
        # For demo purposes, we'll create placeholder structure
        logger.info("Setting up sample dataset structure...")

        # Create basic structure
        (self.output_dir / "raw_images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "annotations").mkdir(parents=True, exist_ok=True)

        logger.info("Sample dataset structure created")
        logger.info("Please place your images in the 'raw_images' folder")
        logger.info("and annotations in the 'annotations' folder")

    def create_coco_annotation_template(self, image_paths: List[str], 
                                      categories: List[Dict]) -> Dict:
        """Create a COCO format annotation template"""
        annotation = {
            "info": {
                "description": "Custom dataset for YOLO segmentation",
                "version": "1.0",
                "year": 2024
            },
            "licenses": [{"id": 1, "name": "Custom License"}],
            "images": [],
            "annotations": [],
            "categories": categories
        }

        for i, img_path in enumerate(image_paths):
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                annotation["images"].append({
                    "id": i,
                    "file_name": Path(img_path).name,
                    "width": w,
                    "height": h
                })

        return annotation

    def convert_coco_to_yolo(self, coco_annotation_path: str, 
                           images_dir: str) -> None:
        """Convert COCO format annotations to YOLO format"""
        with open(coco_annotation_path, 'r') as f:
            coco_data = json.load(f)

        # Create output directories
        output_labels_dir = self.output_dir / "labels"
        output_labels_dir.mkdir(exist_ok=True)

        # Create category mapping
        category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
        self.class_names = [cat['name'] for cat in coco_data['categories']]

        # Process annotations
        image_annotations = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)

        # Convert each image's annotations
        for image_info in coco_data['images']:
            image_id = image_info['id']
            image_name = image_info['file_name']
            width = image_info['width']
            height = image_info['height']

            # Create YOLO format label file
            label_name = Path(image_name).stem + '.txt'
            label_path = output_labels_dir / label_name

            with open(label_path, 'w') as f:
                if image_id in image_annotations:
                    for ann in image_annotations[image_id]:
                        category_id = ann['category_id']
                        class_id = list(category_map.keys()).index(category_id)

                        if 'segmentation' in ann and ann['segmentation']:
                            # Handle polygon segmentation
                            segmentation = ann['segmentation'][0]  # Take first polygon

                            # Normalize coordinates
                            normalized_seg = []
                            for i in range(0, len(segmentation), 2):
                                x = segmentation[i] / width
                                y = segmentation[i + 1] / height
                                normalized_seg.extend([x, y])

                            # Write YOLO format: class_id x1 y1 x2 y2 ... xn yn
                            line = f"{class_id} " + " ".join(map(str, normalized_seg))
                            f.write(line + "\n")

        logger.info(f"Converted COCO annotations to YOLO format in {output_labels_dir}")

    def split_dataset(self, images_dir: str, labels_dir: str, 
                     train_ratio: float = 0.7, val_ratio: float = 0.2,
                     test_ratio: float = 0.1) -> None:
        """Split dataset into train/val/test sets"""
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)

        # Get all image files
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        image_files = [f.stem for f in image_files]

        # Split data
        train_files, temp_files = train_test_split(
            image_files, test_size=(1 - train_ratio), random_state=42
        )

        val_size = val_ratio / (val_ratio + test_ratio)
        val_files, test_files = train_test_split(
            temp_files, test_size=(1 - val_size), random_state=42
        )

        # Create split directories
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        for split_name, file_list in splits.items():
            # Create directories
            split_images_dir = self.output_dir / split_name / 'images'
            split_labels_dir = self.output_dir / split_name / 'labels'
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)

            # Copy files
            for file_stem in file_list:
                # Copy image
                for ext in ['.jpg', '.png', '.jpeg']:
                    src_img = images_path / (file_stem + ext)
                    if src_img.exists():
                        dst_img = split_images_dir / (file_stem + ext)
                        shutil.copy2(src_img, dst_img)
                        break

                # Copy label
                src_label = labels_path / (file_stem + '.txt')
                if src_label.exists():
                    dst_label = split_labels_dir / (file_stem + '.txt')
                    shutil.copy2(src_label, dst_label)

        logger.info(f"Dataset split completed:")
        logger.info(f"  Train: {len(train_files)} files")
        logger.info(f"  Val: {len(val_files)} files") 
        logger.info(f"  Test: {len(test_files)} files")

    def create_dataset_yaml(self, class_names: List[str] = None) -> str:
        """Create dataset.yaml file for YOLO training"""
        if class_names is None:
            class_names = self.class_names or ['vehicle', 'pedestrian']

        yaml_content = f"""path: {self.output_dir.absolute()}
train: train/images
val: val/images
test: test/images

nc: {len(class_names)}
names: {class_names}
"""

        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        logger.info(f"Dataset YAML created at {yaml_path}")
        return str(yaml_path)

    def validate_dataset(self) -> bool:
        """Validate dataset structure and files"""
        required_dirs = [
            self.output_dir / 'train' / 'images',
            self.output_dir / 'train' / 'labels',
            self.output_dir / 'val' / 'images',
            self.output_dir / 'val' / 'labels'
        ]

        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.error(f"Missing directory: {dir_path}")
                return False

            files = list(dir_path.glob("*"))
            if len(files) == 0:
                logger.warning(f"Empty directory: {dir_path}")

        # Check if dataset.yaml exists
        yaml_path = self.output_dir / 'dataset.yaml'
        if not yaml_path.exists():
            logger.warning("dataset.yaml not found")
            return False

        logger.info("Dataset validation completed successfully")
        return True

def main():
    """Main dataset preparation pipeline"""
    preparer = DatasetPreparer("dataset")

    # Download sample dataset (or create structure)
    preparer.download_sample_dataset()

    # Example workflow:
    # 1. If you have COCO format annotations
    # preparer.convert_coco_to_yolo("annotations.json", "raw_images")

    # 2. Split dataset
    # preparer.split_dataset("raw_images", "labels")

    # 3. Create dataset YAML
    preparer.create_dataset_yaml(['vehicle', 'pedestrian'])

    # 4. Validate dataset
    preparer.validate_dataset()

    logger.info("Dataset preparation completed!")
    logger.info("Next steps:")
    logger.info("1. Add your images to dataset/raw_images/")
    logger.info("2. Add corresponding annotations")  
    logger.info("3. Run the conversion and split functions")
    logger.info("4. Start YOLO training")

if __name__ == "__main__":
    main()
