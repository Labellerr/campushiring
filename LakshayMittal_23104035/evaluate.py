c:\Users\mitta\OneDrive\Desktop\labler\dataset\data.yamlfrom ultralytics import YOLO
import matplotlib.pyplot as plt

# Load trained YOLO-Seg model
model = YOLO("runs/segment/train/weights/last.pt")  # Update to the correct weights file

# Validate the model on the validation set
metrics = model.val(data="c:/Users/mitta/OneDrive/Desktop/labler/dataset/data.yaml")  # Ensure the dataset path is correct

# Extract metrics
print(f"mAP: {metrics['metrics/mAP50']:.4f}")
print(f"Precision: {metrics['metrics/precision']:.4f}")
print(f"Recall: {metrics['metrics/recall']:.4f}")

# Plot Precision-Recall Curve
pr_curve = metrics['pr_curve']
plt.plot(pr_curve['recall'], pr_curve['precision'], label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
