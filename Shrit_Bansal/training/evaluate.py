from ultralytics import YOLO
import os
import shutil
import pandas as pd
import json

def main():
    # Path to model weights
    weights_path = os.path.join('..', 'Labeller_Assignment_Dataset', 'weights', 'best.pt')

    # Load trained model
    model = YOLO(weights_path)

    # Path to data config yaml (adjust accordingly)
    data_yaml = os.path.join(os.path.dirname(__file__), 'configs', 'data.yaml')

    print("Running model evaluation ...")
    metrics = model.val(data=data_yaml)

    print("\n=== Model Performance Metrics ===")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

    # Evaluation output folder
    eval_folder = os.path.join('..', 'Labeller_Assignment_Dataset', 'evaluation')
    os.makedirs(eval_folder, exist_ok=True)

    # Results run folder (last validation)
    val_run_path = os.path.join('..', 'runs', 'segment')
    val_folders = [f for f in os.listdir(val_run_path) if f.startswith('val')]

    if val_folders:
        latest_val = sorted(val_folders)[-1]
        val_plots_path = os.path.join(val_run_path, latest_val)
        print(f"Copying evaluation plots from: {val_plots_path}")

        plot_files = []
        for file in os.listdir(val_plots_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src = os.path.join(val_plots_path, file)
                dst = os.path.join(eval_folder, file)
                shutil.copy(src, dst)
                plot_files.append(file)
                print(f"Saved: {file}")

        print(f"Total {len(plot_files)} evaluation plots saved to {eval_folder}")

    # Calculate F1-Score
    f1_score = 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr) if (metrics.box.mp + metrics.box.mr) > 0 else 0

    # Create DataFrame and save as CSV
    metrics_dict = {
        'Metric': ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score'],
        'Value': [
            float(metrics.box.map50),
            float(metrics.box.map),
            float(metrics.box.mp),
            float(metrics.box.mr),
            float(f1_score)
        ]
    }

    df_metrics = pd.DataFrame(metrics_dict)
    csv_path = os.path.join(eval_folder, 'model_metrics.csv')
    df_metrics.to_csv(csv_path, index=False)

    # Save JSON metrics
    json_path = os.path.join(eval_folder, 'model_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(dict(zip(metrics_dict['Metric'], metrics_dict['Value'])), f, indent=2)

    print("\n=== Final Metrics Summary ===")
    print(df_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
