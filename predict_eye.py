import os
import cv2
import numpy as np
import pickle
import sys
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

# ✅ Paths
BASE_DIR = r"C:\Users\kaviy\PYCHARM_47\ML_PRO\code & dataset - MLproject"
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.pkl")

# ✅ Check if model exists
if not os.path.exists(MODEL_PATH):
    print("❌ Trained eye model not found. Please train the model first by running 'train eyeperdication.py'.")
    exit()

# ✅ Load trained model
try:
    with open(MODEL_PATH, "rb") as model_file:
        model, scaler = pickle.load(model_file)
except Exception as e:
    print(f"❌ Error loading eye model: {e}")
    exit()


# ✅ Feature extraction function
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None

    img_resized = cv2.resize(img, (256, 256))

    # GLCM (Texture Features)
    glcm = graycomatrix(img_resized, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Histogram (Color Features)
    hist = cv2.calcHist([img_resized], [0], None, [256], [0, 256]).flatten()
    hist_features = hist / np.sum(hist)  # Normalize

    # Combine all features
    features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, hist_features])
    return features


# ✅ Predict function
def predict_parameters(image_path):
    features = extract_features(image_path)
    if features is None:
        return "❌ Invalid image!"

    features = scaler.transform([features])
    predictions = model.predict(features)[0]

    # ✅ Define output labels based on trained model
    labels_to_predict = ["Disease_Risk", "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN"]
    predicted_values = {label: predictions[i] for i, label in enumerate(labels_to_predict)}
    return predicted_values


# ✅ Run prediction
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python perdicate_eye.py <path_to_image>")
        sys.exit(1)

    IMAGE_PATH = sys.argv[1]

    if os.path.exists(IMAGE_PATH):
        print(f"✅ Processing image: {IMAGE_PATH}")
        prediction_result = predict_parameters(IMAGE_PATH)
        print("\n🔹 Eye Disease Prediction Results:")
        if isinstance(prediction_result, str):
            print(prediction_result)
        else:
            for key, value in prediction_result.items():
                print(f"  {key}: {value:.2f}")
    else:
        print(f"❌ Image file not found at: {IMAGE_PATH}")