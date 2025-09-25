import os
from labellerr.client import LabellerrClient
from labellerr.exceptions import LabellerrError
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
PROJECT_ID = "nance_previous_mockingbird_34242"

MODEL_PATH = r"runs/train/my_yolo_model/weights/best.pt"
METRICS_TXT = "evaluation_metrics.txt"
PREDICTIONS_ZIP = "predictions.zip"
VAL_ZIP = "val_predictions.zip"

def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist!")

def main():
    client = LabellerrClient(api_key=API_KEY, api_secret=API_SECRET)

    check_file(MODEL_PATH)
    client.upload_files([MODEL_PATH], PROJECT_ID)
    print("Model uploaded successfully")

    check_file(METRICS_TXT)
    client.upload_files([METRICS_TXT], PROJECT_ID)
    print("Metrics uploaded successfully")

    check_file(PREDICTIONS_ZIP)
    client.upload_files([PREDICTIONS_ZIP], PROJECT_ID)
    print("Predictions zip uploaded successfully")

    check_file(VAL_ZIP)
    client.upload_files([VAL_ZIP], PROJECT_ID)
    print("Validation predictions zip uploaded successfully")

if __name__ == "__main__":
    main()
