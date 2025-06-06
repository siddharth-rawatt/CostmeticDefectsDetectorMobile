import os
from PIL import Image
import pillow_heif

# Enable HEIC/HEIF support
pillow_heif.register_heif_opener()


def convert_all_heic_to_jpeg(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".heif"):
            heic_path = os.path.join(folder_path, filename)
            jpeg_path = os.path.join(
                folder_path, os.path.splitext(filename)[0] + ".jpg"
            )

            try:
                image = Image.open(heic_path)
                image = image.convert("RGB")
                image.save(jpeg_path, format="JPEG")
                os.remove(heic_path)
                print(f"Converted and deleted: {filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")


# Example usage
folder = "/Users/siddharth/Downloads/8_5_2025 2"
convert_all_heic_to_jpeg(folder)
