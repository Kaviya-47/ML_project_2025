import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import pickle
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# ✅ Base directory (adjust if needed for your deployment)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Feature extraction function (MUST MATCH TRAINING CODE)
def extract_features(image_path, hist_bins=256):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None

    img_resized = cv2.resize(img, (256, 256))

    # GLCM (Texture Features) - 5 Features
    glcm = graycomatrix(img_resized, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Histogram (Color Features)
    hist = cv2.calcHist([img_resized], [0], None, [hist_bins], [0, 256]).flatten()
    hist_features = hist / np.sum(hist)  # Normalize

    # Combine features
    features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, hist_features])
    return features

# ✅ Function to predict eye parameters
def predict_eye_parameters(image_path, model_path):
    if not os.path.exists(model_path):
        print(f"❌ Trained eye model not found at: {model_path}. Please train the model first.")
        return None

    try:
        with open(model_path, "rb") as model_file:
            model, scaler = pickle.load(model_file)
    except Exception as e:
        print(f"❌ Error loading eye model from {model_path}: {e}")
        return None

    features = extract_features(image_path)
    if features is None:
        return "❌ Invalid image for eye prediction!"

    features = scaler.transform([features])
    predictions = model.predict(features)[0]

    labels_to_predict = ["Disease_Risk", "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN"]
    predicted_values = {label: f"{predictions[i]:.2f}" for i, label in enumerate(labels_to_predict)}
    return predicted_values

# ✅ Function to predict age, gender, and heart risk
def predict_heart_risk(image_path, model_path):
    if not os.path.exists(model_path):
        print(f"❌ Trained heart risk models not found at: {model_path}. Please train the models first.")
        return None

    try:
        with open(model_path, "rb") as model_file:
            age_model, gender_model, heart_model, scaler = pickle.load(model_file)
    except Exception as e:
        print(f"❌ Error loading heart risk models from {model_path}: {e}")
        return None

    features = extract_features(image_path, hist_bins=41)
    if features is None:
        return "❌ Invalid image for heart risk prediction!"

    if len(features) != 46:
        print(f"⚠️ Warning: Extracted {len(features)} features, but expected 46 for heart risk model. Adjusting...")
        features = features[:46]

    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)

    age = age_model.predict(features)[0]
    gender = gender_model.predict(features)[0]
    heart_risk = heart_model.predict(features)[0]

    gender_label = "Male" if gender > 0.5 else "Female"

    return {
        "Predicted Age": f"{round(age, 2)}",
        "Predicted Gender": gender_label,
        "Heart Attack Risk (%)": f"{round(heart_risk * 100, 2)}"
    }

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    eye_model_path = os.path.join(BASE_DIR, "trained_model.pkl")
    heart_model_path = os.path.join(BASE_DIR, "2trained_models.pkl")

    eye_results = None
    heart_results = None

    if request.method == 'POST':
        # Handle eye prediction
        if 'eye_image' in request.files:
            eye_image = request.files['eye_image']
            if eye_image and allowed_file(eye_image.filename):
                eye_filename = os.path.join(UPLOAD_FOLDER, "eye_uploaded." + eye_image.filename.rsplit('.', 1)[1].lower())
                eye_image.save(eye_filename)
                eye_results = predict_eye_parameters(eye_filename, eye_model_path)

        # Handle heart risk prediction
        if 'heart_image' in request.files:
            heart_image = request.files['heart_image']
            if heart_image and allowed_file(heart_image.filename):
                heart_filename = os.path.join(UPLOAD_FOLDER, "heart_uploaded." + heart_image.filename.rsplit('.', 1)[1].lower())
                heart_image.save(heart_filename)
                heart_results = predict_heart_risk(heart_filename, heart_model_path)

    return render_template('index.html', eye_results=eye_results, heart_results=heart_results)

if __name__ == '__main__':
    app.run(debug=True)

