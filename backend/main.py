import os
import threading
from PIL import Image
from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.path.join('static', 'uploads')
CLASS_NAMES = ['Normal', 'Pneumonia']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = None
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_model():
    # download pre-trained model
    print("Loading model")
    global model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2) # neuron for normal and pneumonia

    # load model's weights from transfer learning
    model_weights = torch.load("models/pneumonia_classifier.pth", weights_only=True, map_location='cpu')
    model.load_state_dict(model_weights)
    model.eval()
    print("Model loaded")

def transform_image(image):
    image = transform(image)
    return image

def process_image(file_path) -> torch.Tensor:
    image = Image.open(file_path).convert('RGB')
    image_transformed = transform_image(image)
    return image_transformed

def remove_file(file_path):
    try:
        os.remove(file_path)
    except Exception as error:
        app.logger.error("Error removing or closing downloaded file handle", error)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']

        # save file temporarily
        file_name = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        file.save(file_path)
        
        # process image
        image = process_image(file_path)
        image_batch = image.unsqueeze(0)

        # predict
        with torch.no_grad():
            logits = model(image_batch)
        logits = torch.softmax(logits, dim=-1)
        confidence = round(torch.max(logits, -1)[0].item() * 100, 2)
        prediction = torch.argmax(logits, -1).item()
        prediction_class = CLASS_NAMES[prediction]

        # delete uploaded image after 3 seconds
        t = threading.Timer(3.0, function=remove_file, args=(file_path,))
        t.start()

        return render_template("index.html", original_image=file_path, prediction=prediction_class, confidence=confidence)

    return render_template("index.html")

if __name__ == "__main__":
    load_model()
    app.run(host='0.0.0.0', port=5000)