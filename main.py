from flask import Flask, render_template, request
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import base64
import io
import torchvision

app = Flask(__name__)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/best.pt', force_reload=True)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        image_file = request.files["image"]
        image_bytes = image_file.read()
        
        # Load the image and perform object detection with the YOLOv5 model
        img = Image.open(io.BytesIO(image_bytes))
        results = model(img)

        # Convert the prediction results into a base64-encoded string to be displayed on the index.html page
        prediction_buffer = io.BytesIO()
        plt.imsave(prediction_buffer, np.squeeze(results.render()))
        prediction_base64 = base64.b64encode(prediction_buffer.getvalue()).decode('utf-8')
        prediction = f"data:image/png;base64,{prediction_base64}"

        return render_template("index.html", prediction=prediction)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
