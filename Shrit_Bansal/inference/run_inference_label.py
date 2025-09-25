import os
from ultralytics import YOLO
from PIL import Image

def save_yolo_labels(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Classes to keep
    accepted_classes = {0, 1}
    
    for result in results:
        img_path = result.path
        img_name = os.path.basename(img_path)
        txt_file_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(save_dir, txt_file_name)
        
        with open(txt_path, 'w') as f:
            for bbox_norm, cls_id in zip(result.boxes.xywhn.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)):
                if cls_id not in accepted_classes:
                    continue
                line = f"{cls_id} " + " ".join([f"{x:.6f}" for x in bbox_norm]) + "\n"
                f.write(line)
        print(f"Saved labels: {txt_path}")

if __name__ == "__main__":
    model_path = "./Labeller_Assignment_Dataset/weights/best.pt"
    test_images_dir = "./Labeller_Assignment_Dataset/dataset_folder/images/test"
    save_labels_dir = "./Labeller_Assignment_Dataset/pred_results_labels"
    
    model = YOLO(model_path)
    results = model.predict(source=test_images_dir)
    
    save_yolo_labels(results, save_labels_dir)
