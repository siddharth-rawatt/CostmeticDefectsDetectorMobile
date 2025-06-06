# AI-Driven Mobile Device Assessment System

## Overview

This project was developed as part of the **Data Science Postgraduate Project (COSC2667)** at **RMIT University**, in collaboration with **Phi Technologies Pty Ltd**. The system aims to automate the assessment of mobile phone defects for use cases such as returns, insurance claims, and resale evaluation.

### Industry Partner
- **Phi Technologies Pty Ltd**
- **Supervisor:** Steve Moss

### Academic Supervisor
- **Nhat Quang Cao**

### Team Members
- Aakash Thekkinkattil Anand (s4001776)  
- Arvind Krishnan Ramesh (s3977208)  
- Anjali Sudhakar Rahate (s4020263)  
- Samrudhi Pravin Joshi (s3989113)  
- Siddharth Rawat (s4012307)  

---

## Repository Structure

3d model creator # 3D reconstruction experiment (COLMAP, DreamGaussian, etc.)
Image Upscaler # ESRGAN and related upscaling modules
notebook # Development notebooks for training, testing
output # Final outputs and visualizations
output_synthetic_imgaes_flip # Flipped synthetic image augmentation
output_synthetic_imgaes_smudges # Smudge-specific synthetic outputs
phone_damage_ui_final # Final UI components and Flask app
src # Core source code and detection pipeline
model # Trained model weights (YOLO, classifiers, etc.)
venv # Python virtual environment
cosmeticDefectsDetector.egg-info# Package metadata
PhysicalModelling # Lighting and camera distance calibration models
gemini_image_output.png # Example of synthetic image output using Gemini 2.0
requirements.txt # Python dependencies
setup.py # Package setup
README.md # Project overview and setup instructions



---

## Key Features

### Synthetic Defect Image Generation
- Utilized **Gemini 2.0 Flash Exponential** for generating over 300 synthetic phone defect images.
- Achieved 97% prompt fidelity.
- Structured prompt library and quality-controlled dataset.

### Image Upscaling with ESRGAN
- Preserved micro-defects like scratches and smudges using AI-based upscaling.
- Outperformed Real-ESRGAN and SwinIR in PSNR and SSIM benchmarks.

### Feasibility Testing of 3D Modelling
- Evaluated COLMAP, Meshroom, DreamGaussian, and Polycam.
- Concluded that current 3D tools are unsuitable due to hallucination and scalability concerns.

### Camera and Lighting System
- Selected **Sony IMX519** for MVP and OwlSight 64MP for future integration.
- Calibrated working distance based on focal length and sensor dimensions.
- Multi-light Arduino-controlled rig for smudges, scratches, and under-glass defects.

### Web-Based UI (Flask Dashboard)
- Frontend to visualize annotated images, grades, and decision-ready reports.
- Modular and user-friendly for logistics or insurance staff.

---
