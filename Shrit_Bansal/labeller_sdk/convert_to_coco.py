import os
import json
from PIL import Image

IMG_DIR = './Labeller_Assignment_Dataset/predictions'
LBL_DIR = './Labeller_Assignment_Dataset/pred_results_labels'
OUTPUT_JSON = './Labeller_Assignment_Dataset/coco_annotations.json'

CLASS_MAP = {
    '0': 1,  # vehicle -> category id 1
    '1': 2,  # pedestrian -> category id 2
}

CATEGORIES = [
    {'id': 1, 'name': 'vehicle'},
    {'id': 2, 'name': 'pedestrian'},
]

def yolo_to_coco_bbox(x_center, y_center, width, height, img_width, img_height):
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    w = width * img_width
    h = height * img_height
    return [x_min, y_min, w, h]

def convert():
    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    for filename in os.listdir(IMG_DIR):
        if not filename.endswith('.jpg'):
            continue
        img_path = os.path.join(IMG_DIR, filename)
        label_path = os.path.join(LBL_DIR, os.path.splitext(filename)[0] + '.txt')

        # Read image size
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        images.append({
            'id': img_id,
            'file_name': filename,
            'width': img_width,
            'height': img_height
        })

        # Read label file and convert boxes
        if not os.path.exists(label_path):
            print(f'Warning: Label file not found for {filename}')
            img_id += 1
            continue

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x_c, y_c, w, h = parts
                if cls not in CLASS_MAP:
                    continue

                bbox = yolo_to_coco_bbox(float(x_c), float(y_c), float(w), float(h), img_width, img_height)

                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': CLASS_MAP[cls],
                    'bbox': [round(coord, 2) for coord in bbox],
                    'area': round(bbox[2] * bbox[3], 2),
                    'iscrowd': 0
                })
                ann_id += 1

        img_id += 1

    coco_output = {
        'images': images,
        'annotations': annotations,
        'categories': CATEGORIES
    }

    with open(OUTPUT_JSON, 'w') as out_file:
        json.dump(coco_output, out_file, indent=4)

    print(f'Successfully wrote COCO annotations to {OUTPUT_JSON}')

if __name__ == '__main__':
    convert()
