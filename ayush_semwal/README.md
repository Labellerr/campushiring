End-to-End Segmentation and Tracking Pipeline for Vehicles & Pedestrians
This repository contains the complete submission for the Labellerr AI Internship Assignment. It implements a comprehensive, end-to-end computer vision pipeline that begins with raw images and culminates in actionable tracking data. The system performs instance segmentation and multi-object tracking, built with a state-of-the-art YOLOv8-Seg model and the robust ByteTrack algorithm. The entire workflow is deeply integrated with the Labellerr platform, which was used for the critical tasks of data annotation and model-assisted quality assurance.

🚀 Core Features
Instance Segmentation: The pipeline utilizes a fine-tuned YOLOv8-Seg model to produce precise, pixel-level masks for each detected object. This goes beyond simple bounding boxes, providing a rich understanding of an object's exact shape and extents, which is crucial for navigating complex scenes with occlusions.

Multi-Object Tracking: By integrating the powerful ByteTrack algorithm, the system can reliably assign and maintain stable tracking IDs for objects as they move across video frames. This addresses the significant challenge of maintaining object identity through occlusions and complex interactions.

Labellerr Workflow: This project demonstrates the full, cyclical nature of a real-world machine learning lifecycle. The process starts with manual data annotation in Labellerr, and after model training, the system's predictions are programmatically uploaded back to the platform via the SDK, simulating a "model-in-the-loop" process for efficient human review and quality control.

Complete & Reproducible: Every component required to replicate this project is included and clearly documented. This includes all necessary scripts for data preparation, model training, inference, and evaluation, ensuring that the results can be independently verified.

📂 Repository Structure
The project is organized into a clear and logical directory structure to separate concerns and facilitate ease of use.

.
├── README.md # This guide
├── report.pdf # Detailed project report with findings
├── data.yaml # Configuration file for the dataset
├── requirements.txt # Required Python packages
├── data/ # All dataset-related files
│ ├── sources.md
│ ├── train/ # Training images and labels
│ ├── val/ # Validation images and labels
│ └── test/ # Test images
├── src/ # All Python source code
│ ├── train.py
│ ├── inference.py
│ ├── bytetrack_integration.py
│ └── labellerr_upload.py
├── runs/ # Output from training (models, logs)
└── results/ # Output from inference and tracking
├── predictions/ # Per-image JSON predictions
├── results.json # Final tracking data
└── demo_video.mp4 # Visual demo of the tracker

⚙️ Setup Instructions
To configure the project on your local machine, please follow these setup instructions. This project requires Python 3.8 or higher.

1. Clone the Repository
   First, clone this repository to create a local copy of the project on your machine.

git clone [https://github.com/Ayushsemz/campushiring.git](https://github.com/Ayushsemz/campushiring.git)
cd campushiring/AYUSH_SEMWAL/

2. Create and Activate a Virtual Environment
   Using a virtual environment is a best practice that isolates project dependencies and avoids conflicts with other Python projects.

# Create the virtual environment named 'venv'

python -m venv venv

# Activate it (on Windows)

venv\Scripts\activate

# Activate it (on macOS/Linux)

source venv/bin/activate

3. Install Dependencies
   Install all the required Python packages from the requirements.txt file using pip. This command will automatically download and install the correct versions of all necessary libraries.

pip install -r requirements.txt

🏃‍♂️ How to Run the Pipeline
Follow this step-by-step guide to execute the entire workflow, from preparing the data to generating the final tracking results and uploading predictions.

Step 1: Data Preparation
Annotations: The data/train and data/val directories are pre-populated with images and their corresponding YOLO-format labels, which were manually annotated and exported from the Labellerr platform.

Test Set: The data/test/images folder contains the unlabeled images reserved for the final inference step.

Configuration: Before running any scripts, you must open the data.yaml file and update the path variable. This critical step tells the YOLOv8 training script the absolute location of the data directory on your machine, as well as the names of the object classes.

Step 2: Model Training
To fine-tune the YOLOv8-Seg model on the custom dataset, execute the training script. This process will train the model for a specified number of epochs, periodically saving checkpoints and logging performance metrics. The best-performing model checkpoint (best.pt) will be saved to the runs/ directory upon completion.

python src/train.py --data data.yaml --epochs 100 --imgsz 640 --batch 16

Step 3: Run Inference on the Test Set
Use the best trained model to generate predictions on the unseen test images. This script processes each image and saves its predictions—including segmentation masks, bounding boxes, and confidence scores—into a corresponding .json file within the results/predictions/ folder.

python src/inference.py \
 --weights runs/train/labellerr_seg/weights/best.pt \
 --source data/test/images/ \
 --output results/predictions/

Step 4: Run Video Tracking
Process a sample video file to perform multi-object tracking. This script uses the trained model to detect objects in each frame and feeds them to ByteTrack. It generates two key outputs: a demo_video.mp4 for visual verification and a results.json file, which contains the detailed, structured data of every tracked object across all frames.

python src/bytetrack_integration.py \
 --weights runs/train/labellerr_seg/weights/best.pt \
 --source path/to/your/test_video.mp4 \
 --output_json results/results.json \
 --output_video results/demo_video.mp4

Step 5: Upload Predictions to Labellerr
To complete the model-in-the-loop workflow, this script uses the Labellerr SDK to upload the JSON predictions from Step 3 to your test project on the Labellerr platform. This populates the project with model-generated annotations, making them immediately available for human review.

Note: You must replace the placeholder values with your personal Labellerr API key and the specific project ID for your test set.

python src/labellerr_upload.py \
 --api_key "YOUR_LABELLERR_API_KEY" \
 --project_id "YOUR_LABELLERR_PROJECT_ID" \
 --predictions_dir results/predictions/
