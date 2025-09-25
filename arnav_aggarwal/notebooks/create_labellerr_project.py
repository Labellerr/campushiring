import os
from labellerr.client import LabellerrClient
from labellerr.exceptions import LabellerrError


API_KEY = "5a15d5.c1e6904ffdafbd16c1198f79d9"
API_SECRET = "21088e9278773ba9e6acc0ffefa3fd948427b8a6c6ec7b749093756aa2db7c28"
CLIENT_ID = "" 
CREATOR_EMAIL = "your_email@example.com"
PROJECT_NAME = "Vehicle_Pedestrian_Segmentation_Train_SDK"
DATASET_NAME = "Difficult_Vehicle_Ped_Dataset"

DATA_PATH = "C:/Users/DELL/Desktop/final_dataset" 


annotation_guide = {
    "tools": [
        {
            "tool": "polygon",
            "name": "pedestrian",
            "color": "#FF0000"
        },
        {
            "tool": "polygon",
            "name": "vehicle",
            "color": "#0000FF"
        }
    ]
}

project_payload = {
    'client_id': CLIENT_ID,
    'dataset_name': DATASET_NAME,
    'dataset_description': 'A challenging dataset of vehicles and pedestrians with occlusions.',
    'data_type': 'image',
    'created_by': CREATOR_EMAIL,
    'project_name': PROJECT_NAME,
    'annotation_guide': annotation_guide,
    'folder_to_upload': DATA_PATH
}

try:
    print("Initializing Labellerr client...")
    client = LabellerrClient(API_KEY, API_SECRET)
    
    print(f"Creating project '{PROJECT_NAME}' and uploading data from '{DATA_PATH}'...")
    result = client.initiate_create_project(project_payload)
    
    project_id = result.get('project_id', {}).get('response', {}).get('project_id')
    if project_id:
        print(f"Project created successfully. Project ID: {project_id}")
    else:
        print("Project creation may have succeeded, but failed to retrieve project ID.")
        print("Full response:", result)

except LabellerrError as e:
    print(f"An error occurred during project creation: {str(e)}")
except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")