from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import json
from dotenv import load_dotenv
load_dotenv() ## loading all the environment variables
import google.generativeai as genai

app = Flask(__name__)

# Load trained pneumonia detection model
model = tf.keras.models.load_model("pneumonia_detection.h5")


def preprocess_image(image):
    image = image.resize((512, 512))  # Resize to model input size
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Load health advice from JSON file
def load_advice():
    with open("advice.json", "r", encoding="utf-8") as file:  # Use UTF-8 encoding
        return json.load(file)

advice_data = load_advice()

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

gemini_model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
chat = gemini_model.start_chat(history=[])


@app.route("/")
def home():
    return render_template("page.html")

@app.route("/page")
def page():
    return render_template("page.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files["file"]
        if file:
            # Read image without saving
            image = Image.open(io.BytesIO(file.read()))
            processed_image = preprocess_image(image)

            # Predict
            prediction = model.predict(processed_image)
            if prediction > 0.5:
                result = "Found the presence of Pneumonia in your X-Ray"
                advice = advice_data["pneumonia_detected"]
            else:
                result = "Your X-Ray look's like normal"
                advice = advice_data["normal"]

            return render_template("upload.html", prediction=result, advice=advice)

    return render_template("upload.html", prediction="No file uploaded.")
    

@app.route("/bot")
def bot():
    return render_template("bot.html")
@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_message = request.json.get("message")
    try:
        response = chat.send_message(user_message, stream=True)
        full_response = ""
        for chunk in response:
            full_response += chunk.text
        return jsonify({"response": full_response})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})
    
if __name__ == "__main__":
    app.run(debug=True)