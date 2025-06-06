from flask import Flask, render_template
import pandas as pd
import os
import subprocess

app = Flask(__name__)

@app.route('/')
def loading():
    return render_template('loading.html')

@app.route('/results')
def run_model_and_display():
    # ✅ Run the YOLO detection script
    subprocess.run(["python3", "scripts/Yolo.py"], check=True)

    # ✅ Load the final CSV
    csv_path = os.path.join("csv", "final_phone_grades.csv")
    if not os.path.exists(csv_path):
        return "CSV not found. Detection may have failed."

    df = pd.read_csv(csv_path)
    data = df.to_dict(orient='records')

    # ✅ Get the list of detected images
    images_dir = os.path.join("static", "images", "Detected_Images_Yolo")
    images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    return render_template(
        "results.html",
        data=data,
        images=images
    )

if __name__ == '__main__':
    app.run(debug=True)
