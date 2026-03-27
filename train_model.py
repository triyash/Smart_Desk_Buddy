import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# -----------------------------
# Step 1: Generate synthetic dataset
# -----------------------------

rows = 500

data = pd.DataFrame({
    "imu_mean": np.random.uniform(2.5, 4.0, rows),
    "imu_std": np.random.uniform(0.3, 1.5, rows),
    "hr_mean": np.random.uniform(70, 100, rows),
    "hr_std": np.random.uniform(1, 6, rows)
})

# -----------------------------
# Step 2: Generate labels using all 4 inputs
# -----------------------------

labels = []

for i in range(rows):

    imu_mean = data.loc[i, "imu_mean"]
    imu_std = data.loc[i, "imu_std"]
    hr_mean = data.loc[i, "hr_mean"]
    hr_std = data.loc[i, "hr_std"]

    # Focus rule combining motion + heart rate
    if (hr_mean < 85) and (imu_std < 0.9) and (hr_std < 4):
        labels.append("Focused")
    else:
        labels.append("Distracted")

data["focus_state"] = labels

print("Dataset created")
print(data.head())

# -----------------------------
# Step 3: Prepare ML inputs
# -----------------------------

X = data[["imu_mean", "imu_std", "hr_mean", "hr_std"]]
y = data["focus_state"]

# -----------------------------
# Step 4: Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 5: Train Random Forest Model
# -----------------------------

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed")

# -----------------------------
# Step 6: Evaluate Model
# -----------------------------

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# -----------------------------
# Step 7: Save Model
# -----------------------------

joblib.dump(model, "focus_model.pkl")

print("Model saved as focus_model.pkl")