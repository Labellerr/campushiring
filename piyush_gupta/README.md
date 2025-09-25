# Image Segmentation Project

## Project Overview

This project focuses on **image segmentation** using a YOLO-based model. The workflow covers **data collection, manual labeling, model training, validation, and testing**.

---

## Dataset

* **Trained Data And Validation Dataset**: [Google Drive Link](https://drive.google.com/drive/folders/18sliMofl84HGbs84Xl-faBHWp_f6nTyp?usp=sharing)
* **Testing Unlabelled Data**: [Google Drive Link](https://drive.google.com/drive/folders/1rXHWN2rihxO9EzSxzn4PvLy-S8sgfDVm?usp=sharing)

> **Note:** Images were manually annotated using Labellerr.

---

## Links

* **Live Deployed Project Link:** [Render Link](https://motiondetectapp.onrender.com) 
* **Video Demo Link:** [Google Drive Link](https://drive.google.com/file/d/1rI1B5Jty-O1Aq44o6N-lVJSYkeeM3C8S/view?usp=sharing)

---

## Workflow

1. **Data Collection:**

   * Downloaded **150 raw images** containing pedestrians, vehicles, roads, and two-wheelers.

2. **Manual Labeling:**

   * Annotated all images using **Labellerr** to create the labeled dataset.

3. **Validation Dataset:**

   * Selected **20 images** for validation purposes.

4. **Model Training:**

   * Trained a YOLO segmentation model on the **94 training images**.

5. **Model Validation:**

   * Validated the model on the **20 validation images**.

6. **Evaluation Metrics:**

   * **Curve And Confusion Matrix**
     <img width="1544" height="1068" alt="image" src="https://github.com/user-attachments/assets/a753008d-fa20-440a-8085-ba8795030ce4" />

7. **Testing & Label Generation:**

   * Generated predictions for **20 test images** using `model.predict`.
   * Plan to verify predictions manually after fetching the **Client ID from Labellerr**.

---

## Notebook Link

* **Model Training Notebook:** `This notebook contains the complete workflow for model training, including data preprocessing, training, validation, and evaluation steps.  `

---

## Future Work

* Integrate automatic uploading of model-generated labels to Labellerr using **Client ID**.
* Expand dataset with more diverse images to improve model generalization.

