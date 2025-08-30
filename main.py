
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import io
import cv2
import random
import numpy as np

gender_net = cv2.dnn.readNetFromCaffe(
    'C:/Users/Snehal Kayastha/Desktop/projects/Jwellery recomondation system/deploy_gender.prototxt',
    #"C:/Users/Snehal Kayastha/Desktop/projects/Jwellery recomondation system/deploy_gender.prototxt"  
    'C:/Users/Snehal Kayastha/Desktop/projects/Jwellery recomondation system/gender_net.caffemodel'    
)
gender_list = ["Male", "Female"]

# Function to detect gender
def detect_gender(image_data):
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Preprocess for gender model
    blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), (104, 117, 123), swapRB=False, crop=False)
    gender_net.setInput(blob)
    preds = gender_net.forward()
    gender = gender_list[preds[0].argmax()]
    confidence = float(preds[0].max())
    return gender, confidence


app = Flask(__name__)

# Load face shape detection model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Load the jewelry dataset
jewelry_dataset = pd.read_csv('jewelry_dataset.csv')

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect if a face is present
def is_face_present(image_data):
    # Convert to OpenCV image format
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    
    return len(faces) > 0

# Function to predict face shape
def predict_face_shape(image_data):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict the face shape
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = float(prediction[0][index])

    return {"class": class_name, "confidence": confidence_score}

# Function to get random jewelry recommendations
def get_random_jewelry_recommendations(face_shape):
    filtered_data = jewelry_dataset[jewelry_dataset['face_shape'].str.lower() == face_shape.lower()]
    random_recommendations = (
        filtered_data.groupby("type")
        .apply(lambda x: x.sample(1) if len(x) > 0 else None)
        .reset_index(drop=True)
    )
    return random_recommendations.to_dict(orient='records')

@app.route('/')
def home():
    return render_template('home.html')  # Home page

@app.route('/try-on')
def try_on():
    return render_template('index.html') # Try On page

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/tips')
def tips():
    return render_template('tips.html')


@app.route('/randomize', methods=['POST'])
def randomize():
    try:
        # Get face shape from the client
        data = request.json
        face_shape = data["face_shape"]

        # Get random jewelry recommendations
        recommendations = get_random_jewelry_recommendations(face_shape)
        return jsonify({"jewelry_recommendations": recommendations})
    except Exception as e:
        print("Error in /randomize:", str(e))
        return jsonify({"error": "Internal server error. Please try again later."}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get the uploaded image
        image_data = request.files["file"].read()
        
         # Detect gender
        gender, confidence = detect_gender(image_data)
        if gender == "Male":
            return jsonify({"error": "System is designed for women or girls only. Detected gender: Male."}), 400

        # Check for face presence
        if not is_face_present(image_data):
            return jsonify({"error": "No face detected in the image. Please try again with a clear face image."}), 400


        # Predict face shape
        result = predict_face_shape(image_data)

        # Get jewelry recommendations
        recommendations = get_random_jewelry_recommendations(result["class"])

        return jsonify({"face_shape": result, "jewelry_recommendations": recommendations})

    except Exception as e:
        print("Error in /analyze:", str(e))
        return jsonify({"error": "Internal server error. Please try again later."}), 500

if __name__ == '__main__':
    app.run(debug=True)
