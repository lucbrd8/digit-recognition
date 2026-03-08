from flask import Flask, render_template, request, jsonify
import detection_scripts.config as config
import detection_scripts.inference as inference
import detection_scripts.detection_model as detection_model
import torch,torchvision
import torch.nn as nn
from PIL import Image
import io
import numpy as np
import base64

app = Flask(__name__)


# Initialisation du modèle
device = config.device
model = detection_model.BaseModel().to(device)
model.load_state_dict(torch.load("detection_scripts/model_parameters/best_model_base.pth", weights_only=True))

def return_most_probable_digits(prob_list: torch.Tensor,top_k:int =4)->list[dict]:
    top_probs, top_classes = prob_list.topk(top_k)
    return [
        {"class": int(top_classes[0][i]), "probability": float(top_probs[0][i])}
        for i in range(top_k)
    ]


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    image_data=data["image"]
    image_data = base64.b64decode(image_data.split(",")[1])
    image = Image.open(io.BytesIO(image_data)).convert("L")  # Convertir en niveaux de gris
    image = image.resize((28, 28))  # Redimensionner à 28x28

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    predictions = inference.predict_image(image_tensor,model)
    filtered_predictions = return_most_probable_digits(predictions)
    return jsonify(filtered_predictions)

if __name__ == "__main__":
    app.run(debug=True)
