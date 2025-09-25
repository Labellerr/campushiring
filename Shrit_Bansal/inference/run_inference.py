from ultralytics import YOLO
import os

def main():
    # Path to model weights
    weights_path = os.path.join('..', 'Labeller_Assignment_Dataset', 'weights', 'best.pt')

    # Initialize model
    model = YOLO(weights_path)

    # Source images path for inference
    source_path = os.path.join('..', 'Labeller_Assignment_Dataset', 'dataset_folder', 'images', 'test')

    # Directory to save inference results (create if not exist)
    save_dir = os.path.join('..', 'Labeller_Assignment_Dataset', 'predictions')
    os.makedirs(save_dir, exist_ok=True)

    # Run inference and save results
    results = model.predict(source=source_path, save=True, save_dir=save_dir)
    print(f"Inference done. Results saved at {save_dir}")

if __name__ == "__main__":
    main()
