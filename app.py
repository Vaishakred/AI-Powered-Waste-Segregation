# ===================================================================
# 1. SETUP AND IMPORTS
# ===================================================================
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io
import os

# ===================================================================
# 2. INITIALIZE THE APPLICATION AND LOAD THE MODEL
# ===================================================================
app = Flask(__name__)
CORS(app)

# --- IMPORTANT ---
# This path assumes the 'final_model' folder is in the same directory as this app.py file.
MODEL_PATH = "./final_model"
model = None
image_processor = None
class_names = []

try:
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
        # The class names are stored in the model's config
        class_names = list(model.config.id2label.values())
        print("✅ Model and processor loaded successfully!")
        print(f"Available classes: {class_names}")
    else:
        print(f"❌ Error: Model directory not found at {MODEL_PATH}. Please ensure the unzipped 'final_model' folder is here.")

except Exception as e:
    print(f"❌ An error occurred while loading the model: {e}")

# ===================================================================
# 3. DEFINE THE API ENDPOINT
# ===================================================================
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not image_processor:
        return jsonify({'error': 'Model is not loaded properly. Check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        # Read and prepare the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Process the image and make a prediction
        inputs = image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Get the prediction with the highest score
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class_name = class_names[predicted_class_idx]
        
        # Calculate confidence score
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probabilities[0][predicted_class_idx].item()

        return jsonify({
            'prediction': predicted_class_name,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred during prediction: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
