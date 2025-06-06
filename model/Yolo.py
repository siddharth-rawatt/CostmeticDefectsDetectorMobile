import cv2
import os
import numpy as np
from ultralytics import YOLO

# === Paths ===
dataset_path = "/Users/aakash/Documents/Subjects/SEMESTER_4/Project/Trial_Images"
detected_output_path = "/Users/aakash/Documents/Subjects/SEMESTER_4/Project/Detected_Images_Yolo"
mask_output_path = "/Users/aakash/Documents/Subjects/SEMESTER_4/Project/Masked_Images"

# === Setup Output Folders ===
for folder in [detected_output_path, mask_output_path]:
    os.makedirs(folder, exist_ok=True)
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
print("✅ Old detected images and masks cleared.")

# === Load YOLO Models ===s
print("✅ Loading local YOLO models...")
yolo_phone_model = YOLO("/Users/aakash/Documents/Subjects/SEMESTER_4/Project/bestv11.pt")   # Phone segmentation model
yolo_scratch_model = YOLO("/Users/aakash/Documents/Subjects/SEMESTER_4/Project/best2.pt") # Scratch segmentation model

# === Colors for Each Scratch Class (BGR) ===
class_colors = {
    "Broken": (255, 0, 0),
    "MissingBack": (0, 255, 255),
    "MultiScratches": (255, 0, 255),
    "Scratch": (0, 255, 0),
    "Smudge": (0, 165, 255),
    "SmudgeAndScratch": (128, 0, 128)
}

# === Segment Phone using YOLO ===
def segment_phone(image):
    h, w = image.shape[:2]
    results = yolo_phone_model.predict(source=image, imgsz=640, conf=0.8, verbose=False)
    
    phone_mask = np.zeros((h, w), dtype=np.uint8)

    for r in results:
        if r.masks is not None:
            for mask in r.masks.data:
                binary_mask = (mask.cpu().numpy() * 255).astype(np.uint8)
                binary_mask_resized = cv2.resize(binary_mask, (w, h))
                phone_mask = cv2.bitwise_or(phone_mask, binary_mask_resized)

    return phone_mask

# === Detect Defects and Visualize ===
def detect_and_colorize(image, filename):
    h, w = image.shape[:2]
    scratch_mask = np.zeros((h, w), dtype=np.uint8)
    overlay = image.copy()

    results = yolo_scratch_model.predict(source=image, imgsz=640, conf=0.1, verbose=False)

    for r in results:
        if r.masks is not None and r.names:
            for mask, cls_id in zip(r.masks.data, r.boxes.cls):
                class_name = r.names[int(cls_id)]
                if class_name in class_colors:
                    binary_mask = (mask.cpu().numpy() * 255).astype(np.uint8)
                    binary_mask_resized = cv2.resize(binary_mask, (w, h))
                    scratch_mask = cv2.bitwise_or(scratch_mask, binary_mask_resized)

                    # Draw outline and label
                    contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        cv2.drawContours(overlay, [cnt], -1, class_colors[class_name], 2)
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.putText(overlay, class_name, (cx-20, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors[class_name], 2)

    # Save raw scratch mask
    mask_save_path = os.path.join(mask_output_path, f"mask_{filename}")
    cv2.imwrite(mask_save_path, scratch_mask)

    # Segment phone and apply mask
    phone_mask = segment_phone(image)
    final_mask = cv2.bitwise_and(phone_mask, scratch_mask)

    # Apply mask overlay
    scratch_region = cv2.bitwise_and(overlay, overlay, mask=final_mask)
    result = cv2.addWeighted(image, 0.8, scratch_region, 0.6, 0)

    output_save_path = os.path.join(detected_output_path, f"highlighted_{filename}")
    cv2.imwrite(output_save_path, result)

    print(f"✅ Processed {filename}")

# === Main Loop ===
for filename in os.listdir(dataset_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(dataset_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read {filename}")
            continue

        detect_and_colorize(img, filename)

print("\n🎯 All images processed successfully!")