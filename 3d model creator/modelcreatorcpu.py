import os
import subprocess
from pathlib import Path
from PIL import Image
import sqlite3

# === CONFIG ===
PROJECT_DIR = Path(
    "/Users/siddharth/Documents/Semester 4/ACES project/3d model creator/"
)
IMAGES_DIR = PROJECT_DIR / "images"
DB_PATH = PROJECT_DIR / "database.db"
SPARSE_DIR = PROJECT_DIR / "sparse"
DENSE_DIR = PROJECT_DIR / "dense"
COLMAP_CMD = "/opt/homebrew/bin/colmap"  # Change if different

# === Clean Previous Runs ===
for path in [DB_PATH, SPARSE_DIR, DENSE_DIR]:
    if path.exists():
        if path.is_file():
            path.unlink()
        else:
            for sub in path.glob("*"):
                if sub.is_file():
                    sub.unlink()
            path.rmdir()

for d in [PROJECT_DIR, IMAGES_DIR, SPARSE_DIR, DENSE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Estimate Intrinsics from First Image ===
sample_img = next(
    (f for f in IMAGES_DIR.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]),
    None,
)
if not sample_img:
    raise FileNotFoundError("❌ No image found in the images folder!")

with Image.open(sample_img) as img:
    width, height = img.size
    fx = fy = 1.2 * max(width, height)
    cx = width / 2
    cy = height / 2
camera_params = f"{fx},{fy},{cx},{cy}"


# === Utility ===
def run_colmap_step(cmd_args):
    print(f"\n▶️ Running: {' '.join([COLMAP_CMD] + cmd_args)}")
    subprocess.run([COLMAP_CMD] + cmd_args, check=True)


# === Pipeline ===
try:
    # 1. Feature Extraction
    run_colmap_step(
        [
            "feature_extractor",
            "--database_path",
            str(DB_PATH),
            "--image_path",
            str(IMAGES_DIR),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "PINHOLE",
            "--ImageReader.camera_params",
            camera_params,
        ]
    )

    # 2. Matching
    run_colmap_step(["exhaustive_matcher", "--database_path", str(DB_PATH)])

    # 3. Sparse Reconstruction
    run_colmap_step(
        [
            "mapper",
            "--database_path",
            str(DB_PATH),
            "--image_path",
            str(IMAGES_DIR),
            "--output_path",
            str(SPARSE_DIR),
        ]
    )

    # 4. Undistortion
    run_colmap_step(
        [
            "image_undistorter",
            "--image_path",
            str(IMAGES_DIR),
            "--input_path",
            str(SPARSE_DIR / "0"),
            "--output_path",
            str(DENSE_DIR),
            "--output_type",
            "COLMAP",
        ]
    )

    # ❌ SKIP patch_match_stereo (needs CUDA)

    # ✅ Run stereo_fusion directly (CPU-compatible)
    run_colmap_step(
        [
            "stereo_fusion",
            "--workspace_path",
            str(DENSE_DIR),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            str(PROJECT_DIR / "fused.ply"),
        ]
    )

    print("\n✅ CPU-only 3D reconstruction complete. Output saved as: fused.ply")

except subprocess.CalledProcessError as e:
    print(f"\n❌ Error running COLMAP step: {e}")
except Exception as e:
    print(f"\n❌ General Error: {e}")
