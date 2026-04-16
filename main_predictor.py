import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, confusion_matrix, classification_report
)

# === Paths ===
BASE_DIR = r"C:\Users\kaviy\PYCHARM_47\ML_PRO\code & dataset - MLproject"
EYE_MODEL_PATH = os.path.join(BASE_DIR, "trained_model.pkl")
HEART_MODEL_PATH = os.path.join(BASE_DIR, "2trained_models.pkl")

# === Load Eye Disease Model ===
if os.path.exists(EYE_MODEL_PATH):
    with open(EYE_MODEL_PATH, "rb") as f:
        eye_model, eye_scaler = pickle.load(f)
    print("✅ Eye model loaded.")
else:
    print("❌ Eye model not found.")

# === Load Heart Risk Models ===
if os.path.exists(HEART_MODEL_PATH):
    with open(HEART_MODEL_PATH, "rb") as f:
        age_model, gender_model, heart_model, heart_scaler = pickle.load(f)
    print("✅ Heart models loaded.")
else:
    print("❌ Heart models not found.")

# === Load Test Data (must be saved in training scripts) ===
if os.path.exists("X_test_eye.npy"):
    X_test_eye = np.load("X_test_eye.npy")
    Y_test_eye = np.load("Y_test_eye.npy")
else:
    X_test_eye, Y_test_eye = None, None
    print("⚠️ Eye test data not found. Please modify train_eyeperdication.py to save test sets.")

if os.path.exists("X_test_heart.npy"):
    X_test_heart = np.load("X_test_heart.npy")
    y_age_test = np.load("y_age_test.npy")
    y_gender_test = np.load("y_gender_test.npy")
    y_heart_test = np.load("y_heart_test.npy")
else:
    X_test_heart, y_age_test, y_gender_test, y_heart_test = None, None, None, None
    print("⚠️ Heart test data not found. Please modify heartattackperdication.py to save test sets.")

# === Evaluate Eye Disease Model ===
if X_test_eye is not None:
    y_eye_pred = eye_model.predict(X_test_eye)
    print("\n📊 Eye Disease Model Performance:")
    print(f"- MAE: {mean_absolute_error(Y_test_eye, y_eye_pred):.4f}")
    print(f"- RMSE: {np.sqrt(mean_squared_error(Y_test_eye, y_eye_pred)):.4f}")
    print(f"- R² Score: {r2_score(Y_test_eye, y_eye_pred):.4f}")

# === Evaluate Heart Risk Models ===
if X_test_heart is not None:
    # Age (Regression)
    age_pred = age_model.predict(X_test_heart)
    print("\n📊 Age Prediction:")
    print(f"- MAE: {mean_absolute_error(y_age_test, age_pred):.4f}")
    print(f"- RMSE: {np.sqrt(mean_squared_error(y_age_test, age_pred)):.4f}")
    print(f"- R² Score: {r2_score(y_age_test, age_pred):.4f}")

    # Gender (Classification)
    gender_pred = gender_model.predict(X_test_heart)
    print("\n📊 Gender Prediction:")
    print(f"- Accuracy: {accuracy_score(y_gender_test, gender_pred) * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_gender_test, gender_pred, target_names=["Female", "Male"]))

    # Confusion Matrix
    cm = confusion_matrix(y_gender_test, gender_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Female", "Male"],
                yticklabels=["Female", "Male"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Gender Prediction")
    plt.show()

    # Heart Attack Risk (Regression)
    heart_pred = heart_model.predict(X_test_heart)
    print("\n📊 Heart Attack Risk Prediction:")
    print(f"- MAE: {mean_absolute_error(y_heart_test, heart_pred):.4f}")
    print(f"- RMSE: {np.sqrt(mean_squared_error(y_heart_test, heart_pred)):.4f}")
    print(f"- R² Score: {r2_score(y_heart_test, heart_pred):.4f}")
