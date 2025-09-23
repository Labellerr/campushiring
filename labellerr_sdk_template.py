
# Labellerr SDK Integration Template
from labellerr.client import LabellerrClient
from labellerr.exceptions import LabellerrError
import json
import os

class LabellerrManager:
    def __init__(self, api_key, api_secret, client_id, email):
        self.client = LabellerrClient(api_key, api_secret)
        self.client_id = client_id
        self.email = email

    def create_training_project(self, dataset_path, project_name="Vehicle_Pedestrian_Segmentation"):
        """Create training project for annotation"""
        project_payload = {
            'client_id': self.client_id,
            'dataset_name': f'{project_name}_Dataset',
            'dataset_description': 'Vehicle and pedestrian instance segmentation dataset',
            'data_type': 'image',
            'created_by': self.email,
            'project_name': project_name,
            'annotation_guide': [
                {
                    "question_number": 1,
                    "question": "Segment vehicles and pedestrians",
                    "question_id": "vehicle-pedestrian-seg",
                    "option_type": "polygon",
                    "required": True,
                    "options": [
                        {"option_name": "Vehicle"},
                        {"option_name": "Pedestrian"}
                    ]
                }
            ],
            'rotation_config': {
                'annotation_rotation_count': 1,
                'review_rotation_count': 1,
                'client_review_rotation_count': 1
            },
            'autolabel': False,
            'folder_to_upload': dataset_path
        }

        try:
            result = self.client.initiate_create_project(project_payload)
            project_id = result['project_id']['response']['project_id']
            print(f"Training project created successfully. Project ID: {project_id}")
            return project_id
        except LabellerrError as e:
            print(f"Project creation failed: {str(e)}")
            return None

    def export_annotations(self, project_id, export_format='yolo_segmentation'):
        """Export annotations from Labellerr project"""
        try:
            export_result = self.client.export_project(
                project_id=project_id,
                export_format=export_format
            )
            print("Annotations exported successfully!")
            return export_result
        except LabellerrError as e:
            print(f"Export failed: {str(e)}")
            return None

    def create_test_project(self, test_images_path, project_name="Vehicle_Pedestrian_Test"):
        """Create test project for model predictions"""
        test_project_payload = {
            'client_id': self.client_id,
            'dataset_name': f'{project_name}_Dataset',
            'dataset_description': 'Test dataset for model predictions',
            'data_type': 'image',
            'created_by': self.email,
            'project_name': project_name,
            'annotation_template_id': None,  # Will be set based on training project
            'rotation_config': {
                'annotation_rotation_count': 1,
                'review_rotation_count': 1,
                'client_review_rotation_count': 1
            },
            'autolabel': False,
            'folder_to_upload': test_images_path
        }

        try:
            result = self.client.initiate_create_project(test_project_payload)
            project_id = result['project_id']['response']['project_id']
            print(f"Test project created successfully. Project ID: {project_id}")
            return project_id
        except LabellerrError as e:
            print(f"Test project creation failed: {str(e)}")
            return None

    def upload_predictions(self, project_id, predictions):
        """Upload model predictions to test project"""
        try:
            # Format predictions according to Labellerr SDK requirements
            formatted_predictions = self.format_predictions_for_labellerr(predictions)

            result = self.client.upload_predictions(
                project_id=project_id,
                predictions=formatted_predictions
            )
            print("Predictions uploaded successfully!")
            return result
        except LabellerrError as e:
            print(f"Prediction upload failed: {str(e)}")
            return None

    def format_predictions_for_labellerr(self, predictions):
        """Convert YOLO predictions to Labellerr format"""
        # Implementation depends on specific prediction format
        formatted_predictions = []

        for pred in predictions:
            # Convert YOLO segmentation masks to Labellerr polygon format
            formatted_pred = {
                'image_id': pred['image_id'],
                'annotations': [
                    {
                        'class_name': pred['class_name'],
                        'polygon_points': pred['polygon_points'],
                        'confidence': pred['confidence']
                    }
                ]
            }
            formatted_predictions.append(formatted_pred)

        return formatted_predictions

# Usage example
if __name__ == "__main__":
    # Initialize Labellerr manager
    manager = LabellerrManager(
        api_key="your-api-key",
        api_secret="your-api-secret", 
        client_id="your-client-id",
        email="your-email@example.com"
    )

    # Create training project
    train_project_id = manager.create_training_project("./dataset/train/images")

    # After annotation is complete, export annotations
    if train_project_id:
        annotations = manager.export_annotations(train_project_id)

    # Create test project and upload predictions (after model training)
    test_project_id = manager.create_test_project("./dataset/test/images")
