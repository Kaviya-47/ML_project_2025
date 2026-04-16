import os
import cv2
import numpy as np
import pandas as pd
import pickle
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ✅ Update paths to match your dataset
BASE_DIR = r"C:\Users\kaviy\PYCHARM_47\ML_PRO\code & dataset - MLproject"
IMAGE_DIR = os.path.join(BASE_DIR, "Training")  # Folder containing images
LABELS_CSV = os.path.join(BASE_DIR, "RFMID_Training_Labels.csv")  # CSV file
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.pkl")  # Model save path

# ✅ Check if CSV file exists
if not os.path.exists(LABELS_CSV):
    raise FileNotFoundError(f"❌ CSV file not found at {LABELS_CSV}")

# ✅ Load labels from CSV
try:
    df = pd.read_csv(LABELS_CSV)
except Exception as e:
    raise ValueError(f"❌ Error reading CSV file: {e}. Ensure it's a valid CSV file.")

# ✅ Feature extraction function
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Warning: Could not read image {image_path}")
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

# ✅ Load images and extract features
X, Y = [], []
for _, row in df.iterrows():
    image_filename = f"{row['ID']}.png"      # Match image with ID column
    image_path = os.path.join(IMAGE_DIR, image_filename)

    if os.path.exists(image_path):
        features = extract_features(image_path)
        if features is not None:
            X.append(features)
            Y.append(row[1:].values)  # Exclude 'ID' column
    else:
        print(f"⚠️ Warning: Image not found at {image_path}")

# ✅ Convert to NumPy arrays
X = np.array(X)
Y = np.array(Y)

# ✅ Split dataset for training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

np.save("X_test_eye.npy", X_test)
np.save("Y_test_eye.npy", Y_test)


# ✅ Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# ✅ Save trained model
with open(MODEL_PATH, "wb") as model_file:
    pickle.dump((model, scaler), model_file)

print("✅ Model trained and saved successfully!")

