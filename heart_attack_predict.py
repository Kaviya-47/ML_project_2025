import os
import cv2
import numpy as np
import pickle
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

# ✅ Paths
BASE_DIR = r"C:\Users\kaviy\PYCHARM_47\ML_PRO\code & dataset - MLproject"
IMAGE_PATH = os.path.join(BASE_DIR, "1.png")  # ✅ Single image file
MODEL_PATH = os.path.join(BASE_DIR, "2trained_models.pkl")

# ✅ Load trained models
try:
    with open(MODEL_PATH, "rb") as model_file:
        age_model, gender_model, heart_model, scaler = pickle.load(model_file)
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# ✅ Feature extraction function (MUST MATCH TRAINING CODE)
def extract_features(image_path):
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

    # Histogram (Color Features) - Adjusted to 41 bins
    hist = cv2.calcHist([img_resized], [0], None, [41], [0, 256]).flatten()  # Now 41 bins

    hist_features = hist / np.sum(hist)  # Normalize

    # Combine to exactly 46 features (5 GLCM + 41 histogram bins)
    features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, hist_features])

    # ✅ Ensure the feature count is exactly 46
    if len(features) != 46:
        print(f"⚠️ Warning: Extracted {len(features)} features, but expected 46. Adjusting...")
        features = features[:46]  # Trim if extra
    return features


# ✅ Predict function for a single image
def predict_single_image(image_path):
    features = extract_features(image_path)
    if features is None:
        return "❌ Invalid image!"

    features = np.array(features).reshape(1, -1)  # Ensure correct shape
    features = scaler.transform(features)  # Still works fine, just ignore the warning

    # Predictions
    age = age_model.predict(features)[0]
    gender = gender_model.predict(features)[0]
    heart_risk = heart_model.predict(features)[0]

    # Gender label
    gender_label = "Male" if gender > 0.5 else "Female"

    return {
        "Predicted Age": round(age, 2),
        "Predicted Gender": gender_label,
        "Heart Attack Risk (%)": round(heart_risk * 100, 2)
    }

# ✅ Run prediction for a single image
if __name__ == "__main__":
    if os.path.exists(IMAGE_PATH):
        prediction_result = predict_single_image(IMAGE_PATH)
        print("\n🔹 Prediction Results:")
        if isinstance(prediction_result, str):
            print(prediction_result)
        else:
            for key, value in prediction_result.items():
                print(f"  {key}: {value}")
    else:
        print(f"❌ Image file not found at: {IMAGE_PATH}")
