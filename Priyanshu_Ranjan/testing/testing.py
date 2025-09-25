from ultralytics import YOLO

MODEL_PATH = r"runs/train/my_yolo_model/weights/best.pt"
TEST_IMAGES = r"E:\labeller_dataset\export-#rMIQQmGnv5Mxg876WvMi\test_images"
TEST_DATA = r"E:\labeller_dataset\export-#rMIQQmGnv5Mxg876WvMi\data.yaml"
VAL_FOLDER = r"runs/segment/val"
METRICS_TXT = "evaluation_metrics.txt"

def main():
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=TEST_IMAGES,
        imgsz=640,
        conf=0.5,
        save=True,
        save_dir=VAL_FOLDER
    )

    metrics = model.val(
        data=TEST_DATA,
        imgsz=640,
        batch=8,
        workers=0
    )

    with open(METRICS_TXT, 'w') as f:
        f.write("Precision: {}\n".format(metrics.box.p.tolist()))
        f.write("Recall: {}\n".format(metrics.box.r.tolist()))
        f.write("mAP50: {}\n".format(metrics.box.map50.tolist()))
        f.write("mAP50_95: {}\n".format(metrics.box.map.tolist()))

if __name__ == "__main__":
    main()
