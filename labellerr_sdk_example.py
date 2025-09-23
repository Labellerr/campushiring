# labellerr_sdk_example.py
import os
import json
import requests

# CONFIG - fill in
LABELLERR_API_KEY = os.getenv("LABELLERR_API_KEY", "YOUR_LABELLERR_API_KEY")
LABELLERR_API_BASE = "https://api.labellerr.com"  # replace if different
PROJECT_NAME = "yolo_seg_demo_train"  # or any name

HEADERS = {
    "Authorization": f"Bearer {LABELLERR_API_KEY}",
    "Content-Type": "application/json"
}

def create_project(name, description="YOLO Segmentation demo"):
    url = f"{LABELLERR_API_BASE}/v1/projects"
    payload = {"name": name, "description": description, "type": "segmentation"}
    r = requests.post(url, headers=HEADERS, json=payload)
    r.raise_for_status()
    return r.json()

def upload_image(project_id, img_path):
    # Example using multipart file upload endpoint - check SDK docs for exact endpoints
    url = f"{LABELLERR_API_BASE}/v1/projects/{project_id}/files"
    with open(img_path, "rb") as f:
        files = {"file": f}
        r = requests.post(url, headers={"Authorization": f"Bearer {LABELLERR_API_KEY}"}, files=files)
    r.raise_for_status()
    return r.json()

def upload_predictions_as_annotations(project_id, predictions_coco_json_path):
    """
    Upload model predictions exported as COCO-style JSON (Labellerr accepts COCO predictions).
    """
    url = f"{LABELLERR_API_BASE}/v1/projects/{project_id}/predictions/import"
    with open(predictions_coco_json_path, "rb") as f:
        files = {"file": f}
        r = requests.post(url, headers={"Authorization": f"Bearer {LABELLERR_API_KEY}"}, files=files)
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    # Example usage
    # 1) Create project
    resp = create_project(PROJECT_NAME)
    print("Created project:", resp)
    project_id = resp.get("id")

    # 2) Upload one example image (loop over folder in practice)
    # img_resp = upload_image(project_id, "data/images/train/img001.jpg")
    # print("Uploaded image response:", img_resp)

    # 3) After inference, upload predictions JSON
    # upload_resp = upload_predictions_as_annotations(project_id, "predictions/test_predictions_coco.json")
    # print("Upload predictions response:", upload_resp)
