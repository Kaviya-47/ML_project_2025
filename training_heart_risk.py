import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score

# ✅ Define paths
BASE_DIR = r"C:\Users\kaviy\PYCHARM_47\ML_PRO\code & dataset - MLproject"
LABELS_CSV = os.path.join(BASE_DIR, "RFMID_Training_Labels.csv")  # CSV file
MODEL_PATH = os.path.join(BASE_DIR, "2trained_models.pkl")  # Save models

# ✅ Load dataset
df = pd.read_csv(LABELS_CSV)

# ✅ Feature Selection: Using Disease_Risk & Retinal Diseases as Inputs
features = ['Disease_Risk', 'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'ERM', 'LS', 'MS', 'CSR', 'ODC', 'CRVO', 'TV',
            'AH', 'ODP', 'ODE', 'ST', 'AION', 'PT', 'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP', 'CWS', 'CB', 'ODPM', 'PRH',
            'MNF', 'HR', 'CRAO', 'TD', 'CME', 'PTCR', 'CF', 'VH', 'MCA', 'VS', 'BRAO', 'PLQ', 'HPED', 'CL']
X = df[features]

# ✅ Synthetic Label Creation (Estimated Age, Gender, Heart Attack Risk)
df['Age'] = 50 + (df['ARMD'] * 20 + df['DR'] * 10 + df['HR'] * 5 + np.random.randint(-5, 5, size=len(df)))
df['Gender'] = np.where(df['MYA'] > 0.5, 1, 0)  # Approximate Gender (1=Male, 0=Female)
df['Heart_Attack_Risk'] = (df['CRVO'] * 20 + df['CRAO'] * 15 + df['HR'] * 10 + df['Disease_Risk'] * 30) / 100

# ✅ Targets
y_age = df['Age']
y_gender = df['Gender']
y_heart_attack = df['Heart_Attack_Risk']

# ✅ Split data
X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test, y_heart_train, y_heart_test = train_test_split(
    X, y_age, y_gender, y_heart_attack, test_size=0.2, random_state=42)

np.save("X_test_heart.npy", X_test)
np.save("y_age_test.npy", y_age_test)
np.save("y_gender_test.npy", y_gender_test)
np.save("y_heart_test.npy", y_heart_test)


# ✅ Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Train models
age_model = RandomForestRegressor(n_estimators=100, random_state=42)
age_model.fit(X_train, y_age_train)

gender_model = RandomForestClassifier(n_estimators=100, random_state=42)
gender_model.fit(X_train, y_gender_train)

heart_risk_model = XGBRegressor(n_estimators=100, random_state=42)
heart_risk_model.fit(X_train, y_heart_train)

# ✅ Save models
with open(MODEL_PATH, "wb") as model_file:
    pickle.dump((age_model, gender_model, heart_risk_model, scaler), model_file)

# ✅ Evaluate models
age_pred = age_model.predict(X_test)
gender_pred = gender_model.predict(X_test)
heart_pred = heart_risk_model.predict(X_test)

print("📊 Model Performance:")
print(f"- Age Prediction MAE: {mean_absolute_error(y_age_test, age_pred):.2f}")
print(f"- Gender Prediction Accuracy: {accuracy_score(y_gender_test, gender_pred) * 100:.2f}%")
print(f"- Heart Attack Risk Prediction MAE: {mean_absolute_error(y_heart_test, heart_pred):.2f}")

print("✅ 2 Models trained and saved successfully!")
