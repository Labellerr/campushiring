"""
Upload model predictions to Labellerr test project so they appear as pre-annotations.
Edit: API base, auth, and field names to match Labellerr SDK spec.
"""
import json, os, sys
# from labellerr_sdk import Client  # replace with actual import per SDK

LABELLERR_API_KEY = os.getenv("LABELLERR_API_KEY", "PUT-KEY")
TEST_PROJECT_ID = os.getenv("LABELLERR_TEST_PROJECT_ID", "PUT-ID")

def main(pred_json_path="predictions/results.json"):
    # client = Client(api_key=LABELLERR_API_KEY)
    with open(pred_json_path) as f:
        preds = json.load(f)
    for p in preds:
        image_name = p["image"]
        dets = p["detections"]
        # file_id = client.find_file(TEST_PROJECT_ID, image_name)
        # client.upload_prediction(TEST_PROJECT_ID, file_id, dets)  # conform to expected schema
        print(f"[DRY-RUN] Would upload {len(dets)} detections for {image_name}")
    print("Done (dry run). Fill SDK calls and run again.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "predictions/results.json")
