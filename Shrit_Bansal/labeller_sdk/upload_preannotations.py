import time
from labellerr.client import LabellerrClient
from labellerr.exceptions import LabellerrError
from dotenv import load_dotenv
import os

load_dotenv()  

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
CLIENT_ID = os.getenv('CLIENT_ID')
PROJECT_ID = os.getenv('PROJECT_ID')
ANNOTATION_FORMAT = "coco_json"
ANNOTATION_FILE = "./Labeller_Assignment_Dataset/coco_annotations.json"  # Path to your COCO JSON file



def upload_annotations():
    client = LabellerrClient(API_KEY, API_SECRET)
    try:
        result = client.upload_preannotation_by_project_id(PROJECT_ID, CLIENT_ID, ANNOTATION_FORMAT, ANNOTATION_FILE)
        if result['response']['status'] == 'completed':
            print("Pre-annotations uploaded and processed successfully!")
        else:
            print("Upload finished with status:", result['response']['status'])
    except LabellerrError as e:
        print(f"Upload failed: {str(e)}")

if __name__ == '__main__':
    upload_annotations()
