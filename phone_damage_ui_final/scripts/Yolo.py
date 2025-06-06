import cv2
import os
import numpy as np
import csv
import pandas as pd
from ultralytics import YOLO

# === Paths (relative to Flask base directory) ===
base_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(base_dir, ".."))  # Go up to project root

dataset_path = os.path.join(base_dir, "static", "images", "Single_Phone")
detected_output_path = os.path.join(base_dir, "static", "images", "Detected_Images_Yolo")
mask_output_path = os.path.join(base_dir, "static", "images", "Masked_Images")
report_path = os.path.join(base_dir, "csv", "condition_report.csv")
final_report_path = os.path.join(base_dir, "csv", "final_phone_grades.csv")

model_dir = os.path.join(base_dir, "model")

# === Surface Importance Weights ===
surface_weights = {
    "Surface-AA": 1.0,
    "Camera": 1.0,
    "Surface-A": 0.6,
    "Surface-B": 0.3
}

# === Class Colors ===
class_colors = {
    "Broken": (255, 0, 0),
    "MissingBack": (0, 255, 255),
    "MultiScratches": (255, 0, 255),
    "Scratch": (0, 255, 0),
    "Smudge": (0, 165, 255),
    "SmudgeAndScratch": (128, 0, 128)
}

# === YOLO Models (lazy load once) ===
yolo_phone_model = YOLO(os.path.join(model_dir, "bestv11.pt"))
yolo_scratch_model = YOLO(os.path.join(model_dir, "best2.pt"))
yolo_surface_model = YOLO(os.path.join(model_dir, "Surface.pt"))

def segment_phone(image):
    h, w = image.shape[:2]
    results = yolo_phone_model.predict(source=image, imgsz=640, conf=0.8, verbose=False)
    phone_mask = np.zeros((h, w), dtype=np.uint8)
    for r in results:
        if r.masks is not None:
            for mask in r.masks.data:
                binary_mask = (mask.cpu().numpy() * 255).astype(np.uint8)
                phone_mask = cv2.bitwise_or(phone_mask, cv2.resize(binary_mask, (w, h)))
    return phone_mask

def segment_surfaces(image):
    h, w = image.shape[:2]
    results = yolo_surface_model.predict(source=image, imgsz=640, conf=0.1, verbose=False)
    surface_masks = {}
    for r in results:
        if r.masks is not None and r.names:
            for mask, cls_id in zip(r.masks.data, r.boxes.cls):
                name = r.names[int(cls_id)]
                binary_mask = (mask.cpu().numpy() * 255).astype(np.uint8)
                surface_masks[name] = cv2.resize(binary_mask, (w, h))
    return surface_masks

def detect_and_colorize(image, filename):
    h, w = image.shape[:2]
    overlay = image.copy()
    scratch_mask = np.zeros((h, w), dtype=np.uint8)

    results = yolo_scratch_model.predict(source=image, imgsz=640, conf=0.2, verbose=False)
    for r in results:
        if r.masks is not None and r.names:
            for mask, cls_id in zip(r.masks.data, r.boxes.cls):
                class_name = r.names[int(cls_id)]
                binary_mask = (mask.cpu().numpy() * 255).astype(np.uint8)
                binary_mask_resized = cv2.resize(binary_mask, (w, h))
                scratch_mask = cv2.bitwise_or(scratch_mask, binary_mask_resized)

                contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cv2.drawContours(overlay, [cnt], -1, class_colors.get(class_name, (255, 255, 255)), 2)

    phone_mask = segment_phone(image)
    surface_masks = segment_surfaces(image)
    final_mask = cv2.bitwise_and(phone_mask, scratch_mask)

    weighted_score = 0.0
    surface_contribs = []
    for name in surface_weights:
        mask = surface_masks.get(name)
        if mask is not None:
            overlap = cv2.bitwise_and(mask, scratch_mask)
            total = np.count_nonzero(mask)
            overlap_pixels = np.count_nonzero(overlap)
            contrib = (overlap_pixels / total) * surface_weights[name] if total > 0 else 0
            weighted_score += contrib
            surface_contribs.append(overlap_pixels)
        else:
            surface_contribs.append(0)

    grade = "A" if weighted_score <= 0.05 else "B" if weighted_score <= 0.15 else "C"
    scratch_region = cv2.bitwise_and(overlay, overlay, mask=final_mask)
    result = cv2.addWeighted(image, 0.8, scratch_region, 0.6, 0)
    cv2.putText(result, f"Grade: {grade} (Score: {weighted_score:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(os.path.join(detected_output_path, f"highlighted_{filename}"), result)
    cv2.imwrite(os.path.join(mask_output_path, f"mask_{filename}"), scratch_mask)

    phone_id = filename.split("_")[1] if "_" in filename else filename.split(".")[0]

    with open(report_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            phone_id,
            filename,
            f"{weighted_score:.4f}",
            grade,
            np.count_nonzero(phone_mask),
            np.count_nonzero(scratch_mask),
            np.count_nonzero(final_mask)
        ] + surface_contribs)

    print(f"✅ {filename}: Grade {grade} | Score: {weighted_score:.2f}")

def main():
    # === Clear Output Folders ===
    for folder in [detected_output_path, mask_output_path]:
        os.makedirs(folder, exist_ok=True)
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    print("✅ Old detected images and masks cleared.")

    # === Write Report Header ===
    with open(report_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Phone_ID", "Filename", "Score", "Grade", "Phone Area", "Scratch Area", "Final Scratch Area"] + list(surface_weights.keys()))

    # === Run Detection Loop ===
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Could not read {filename}")
                continue
            detect_and_colorize(img, filename)

    print("\n📋 Condition report saved!")

    # === Aggregate Results ===
    df = pd.read_csv(report_path)
    df["Score"] = df["Score"].astype(float)

    summary = df.groupby("Phone_ID").agg({
        "Score": "mean",
        "Phone Area": "sum",
        "Scratch Area": "sum",
        "Final Scratch Area": "sum",
        **{col: "sum" for col in surface_weights.keys()}
    }).reset_index()

    def get_final_grade(score):
        if score <= 0.05:
            return "A"
        elif score <= 0.15:
            return "B"
        else:
            return "C"

    summary["Final_Grade"] = summary["Score"].apply(get_final_grade)
    summary.to_csv(final_report_path, index=False)
    print(f"\n📊 Final phone summary saved to: {final_report_path}")

# === Entry Point ===
if __name__ == "__main__":
    main()
