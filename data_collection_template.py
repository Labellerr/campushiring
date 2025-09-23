
# Data Collection and Preparation Template
import os
import json
import wget
import zipfile
from pathlib import Path
import shutil

def download_coco_subset():
    """Download COCO dataset subset for vehicle and pedestrian detection"""
    # Create directory structure
    os.makedirs('dataset/raw_images', exist_ok=True)
    os.makedirs('dataset/train/images', exist_ok=True)
    os.makedirs('dataset/val/images', exist_ok=True)
    os.makedirs('dataset/test/images', exist_ok=True)

    # Download COCO validation set (smaller for quick setup)
    print("Downloading COCO validation images...")
    wget.download('http://images.cocodataset.org/zips/val2017.zip', 'val2017.zip')

    print("Downloading COCO annotations...")
    wget.download('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', 'annotations.zip')

    # Extract files
    with zipfile.ZipFile('val2017.zip', 'r') as zip_ref:
        zip_ref.extractall('dataset/raw_images/')

    with zipfile.ZipFile('annotations.zip', 'r') as zip_ref:
        zip_ref.extractall('dataset/')

    # Clean up zip files
    os.remove('val2017.zip')
    os.remove('annotations.zip')

    print("COCO subset downloaded successfully!")

def filter_vehicle_pedestrian_images(annotations_path, target_classes=['person', 'car', 'truck', 'bus', 'motorcycle']):
    """Filter images containing vehicles and pedestrians"""
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    # Get category IDs for target classes
    target_category_ids = []
    for category in coco_data['categories']:
        if category['name'] in target_classes:
            target_category_ids.append(category['id'])

    # Find images with target objects
    target_image_ids = set()
    for annotation in coco_data['annotations']:
        if annotation['category_id'] in target_category_ids:
            target_image_ids.add(annotation['image_id'])

    # Get image filenames
    target_images = []
    for image in coco_data['images']:
        if image['id'] in target_image_ids:
            target_images.append(image['file_name'])

    print(f"Found {len(target_images)} images with target classes")
    return target_images[:200]  # Limit to 200 images

def create_data_splits(image_list, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """Split data into train, validation, and test sets"""
    import random
    random.shuffle(image_list)

    total_images = len(image_list)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)

    train_images = image_list[:train_count]
    val_images = image_list[train_count:train_count + val_count]
    test_images = image_list[train_count + val_count:]

    return train_images, val_images, test_images

def create_sources_md(sources_info):
    """Create sources.md file documenting data sources and licenses"""
    content = """# Data Sources and Licensing Information

## Primary Dataset Sources

### COCO Dataset
- **Source:** Microsoft COCO Dataset
- **URL:** https://cocodataset.org/
- **License:** Creative Commons Attribution 4.0 License
- **Usage:** Training and evaluation images for vehicle and pedestrian detection
- **Classes Used:** person, car, truck, bus, motorcycle

### Custom Additions
- **Challenging scenarios:** Low-light conditions, heavy occlusion, adverse weather
- **Synthetic data:** Generated edge cases for robust training

## Compliance
All data usage complies with respective licenses and terms of service.
Data is used for educational and research purposes only.
"""

    with open('dataset/sources.md', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    # Execute data collection pipeline
    download_coco_subset()

    # Filter for target classes
    target_images = filter_vehicle_pedestrian_images('dataset/annotations/instances_val2017.json')

    # Create data splits
    train_imgs, val_imgs, test_imgs = create_data_splits(target_images)

    print(f"Data split: Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

    # Create sources documentation
    create_sources_md({})
