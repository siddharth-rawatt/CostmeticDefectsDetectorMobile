# AI-Driven Mobile Device Assessment System

## Overview

This project was developed as part of the Data Science Postgraduate Project (COSC2667) at RMIT University, in collaboration with an industry partner. The system automates the visual detection and grading of physical mobile phone defects. It supports defect identification for use cases such as insurance claims, returns, and resale evaluation.

---

## Team

- Academic Supervisor: Nhat Quang Cao  
- Team Members:
  - Aakash Thekkinkattil Anand (s4001776)  
  - Arvind Krishnan Ramesh (s3977208)  
  - Anjali Sudhakar Rahate (s4020263)  
  - Samrudhi Pravin Joshi (s3989113)  
  - Siddharth Rawat (s4012307)  

---

## Key Features

### Synthetic Defect Image Generation

- Generated over 300 synthetic images to augment rare defect classes such as diagonal scratches, corner cracks, and oil smudges.
- Designed structured prompts to control defect type, location, lighting, and material characteristics.
- Achieved approximately 97 percent prompt-to-image fidelity, improving the classifier's ability to generalize across device types.

### Image Upscaling with AI

- The system was designed to operate on low-megapixel input images to reduce hardware costs.
- Image enhancement was used to expose micro-level surface defects such as faint scratches and fingerprints.
- After testing several techniques, one approach was selected that balanced edge clarity and runtime performance.
- Upscaling improved the downstream F1-score by 14 to 19 percent on real validation data.

### 3D Modelling Feasibility Testing

- Multiple 3D reconstruction techniques were evaluated to assess their ability to capture surface-level defects spatially.
- Most tools required 30 to 50 input images, which exceeded the project’s 10-image capture design.
- Generative models introduced hallucinations or failed to detect subtle defects.
- Based on these findings, 3D modelling was excluded from the deployed pipeline but remains a future area for development.

### Web-Based UI (Flask)

- A modular Flask application was built to visualize predictions, inspect defect annotations, and generate downloadable reports.
- Designed for usability by logistics, insurance, and QC teams.
- Supports both CSV and PDF output formats for reports.

---

## Repository Structure

| Folder / File                       | Description |
|------------------------------------|-------------|
| `3d model creator/`                | Experiments for 3D reconstruction testing |
| `Image Upscaler/`                  | Scripts and notebooks for image enhancement |
| `notebook/`                        | Jupyter notebooks for model testing and analysis |
| `output/`                          | Processed and annotated image results |
| `output_synthetic_imgaes_flip/`    | Horizontally flipped image augmentations |
| `output_synthetic_imgaes_smudges/` | Synthetic smudge-focused image set |
| `mobile_defect_detection/`         | Flask web application source code |
| `src/`                             | Core Python modules for detection logic |
| `PhysicalModelling/`               | Calibration scripts for lighting and positioning |
| `gemini_image_output.png`          | Example output from synthetic generation |
| `requirements.txt`                 | Python dependency list |
| `setup.py`                         | Package setup file |
| `venv/`                            | Python virtual environment (optional) |

---

## Setup and Execution Instructions

### Step 1: Clone the Repository

In the bash/terminal

git clone https://github.com/siddharth-rawatt/CostmeticDefectsDetectorMobile

cd CostmeticDefectsDetectorMobile

### Step 2: Create and Activate a Virtual Environment
Using Python venv:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### Step 3: Install Dependencies
Using pip:
pip install -r requirements.txt

Alternatively, using Conda:
conda create -n phone-defect python=3.10
conda activate phone-defect
pip install -r requirements.txt

### Model Files Setup
Model weights are not included in the repository due to file size limitations. Please download the required model files from the shared Google Drive folder provided by the project team.

Required Files
phone_segmentation.pt
scratch_detection.pt
surface_segmentation.pt
Directory Structure
Ensure the downloaded files are placed in the following structure:

mobile_defect_detection/
├── model/
│   ├── phone_segmentation.pt
│   ├── scratch_detection.pt
│   └── surface_segmentation.pt

### Running the Application

Navigate to the Flask application directory:
cd mobile_defect_detection
python app.py

Once the server is running, open your browser and visit: http://127.0.0.1:5000/