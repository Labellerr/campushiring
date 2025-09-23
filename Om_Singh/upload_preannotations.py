import os
from labellerr.client import LabellerrClient
from labellerr.exceptions import LabellerrError
from dotenv import load_dotenv 
import json

# Load credentials from your .env file
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
CLIENT_ID = os.getenv("CLIENT_ID") 
TEST_PROJECT_ID = os.getenv("TEST_PROJECT_ID")


PATH_TO_TEST_IMAGES = os.path.join(BASE_DIR, "data", "test")
print(PATH_TO_TEST_IMAGES)
PATH_TO_PREDICTION_LABELS = os.path.join(BASE_DIR, "data", "labels")
print(PATH_TO_PREDICTION_LABELS)


# The name of the temporary JSON file we will create
OUTPUT_JSON_FILE = "predictions_for_upload.json"

CLASS_MAP = {
    0: 'Vehicle',
    1: 'Pedestrian'
}

# --- 2. SCRIPT LOGIC ---

def upload_predictions_to_labellerr():
    """
    Reads YOLO prediction .txt files, converts them to the Labellerr JSON format,
    and uploads them as pre-annotations using the SDK.
    """
    print("Starting the prediction processing...")

    all_annotations_for_upload = []

    # Iterating through every prediction file in the labels folder
    for label_filename in os.listdir(PATH_TO_PREDICTION_LABELS):
        if not label_filename.endswith('.txt'):
            continue

        image_name = os.path.splitext(label_filename)[0] + '.jpg'
        image_path = os.path.join(PATH_TO_TEST_IMAGES, image_name)

        if not os.path.exists(image_path):
            print(f"Warning: Could not find matching image for label '{label_filename}'. Skipping.")
            continue
            
        file_annotation_data = {
            "filePath": image_path,
            "annotations": []
        }

        with open(os.path.join(PATH_TO_PREDICTION_LABELS, label_filename), 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            class_index = int(parts[0])
            class_name = CLASS_MAP.get(class_index, "Unknown")
            
            coords = [float(p) for p in parts[1:]]
            
            points_list = []
            for i in range(0, len(coords), 2):
                points_list.append({"x": coords[i], "y": coords[i+1]})

            annotation_object = {
                "label": class_name,
                "type": "POLYGON",
                "points": points_list
            }
            file_annotation_data["annotations"].append(annotation_object)

        if file_annotation_data["annotations"]:
            all_annotations_for_upload.append(file_annotation_data)
            print(f"Processed predictions for: {image_name}")

    if not all_annotations_for_upload:
        print("No valid predictions found to process.")
        return
    
    print(f"\nReady to upload {len(all_annotations_for_upload)} files with annotations.")

    # --- 3. SAVE TO JSON AND UPLOAD TO LABELLERR (Corrected Logic) ---

    print(f"\nMerging {len(all_annotations_for_upload)} files into a single JSON file...")
    
    # Write the merged list of annotations to a single JSON file
    try:
        with open(OUTPUT_JSON_FILE, 'w') as f:
            json.dump(all_annotations_for_upload, f, indent=4)
        print(f"Successfully created '{OUTPUT_JSON_FILE}'.")
    except Exception as e:
        print(f"Error writing to JSON file: {e}")
        return

    print("Attempting to upload pre-annotations...")
    try:
        # Initialize the client with your API key and secret
        client = LabellerrClient(API_KEY, API_SECRET)
        print("Successfully connected to Labellerr.")

        # This is the correct SDK function from the documentation you provided
        result = client.upload_preannotation_by_project_id(
            project_id=TEST_PROJECT_ID,
            client_id=CLIENT_ID,
            annotation_format='json', # Use this format for the structure we created
            annotation_file=os.path.join(BASE_DIR, "predictions_for_upload.json")
        )

        # Check the final status
        if result.get('response', {}).get('status') == 'completed':
            print("\n--- UPLOAD SUCCESSFUL! ---")
            print("Response from Labellerr:", result)
            print("\nPlease go to your 'Test Project' on the Labellerr website to verify the pre-annotations.")
        else:
            print("\n--- UPLOAD FAILED OR IS PROCESSING ---")
            print(f"Final status was not 'completed'. Full response: {result}")

    except LabellerrError as e:
        print(f"\n--- UPLOAD FAILED! ---")
        print(f"An error occurred: {str(e)}")
    except Exception as e:
        print(f"\n--- AN UNEXPECTED ERROR OCCURRED! ---")
        print(f"Error: {e}")

if __name__ == "__main__":
    upload_predictions_to_labellerr()

