import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


rows = 1000

data = pd.DataFrame({
    "imu_mean": np.random.uniform(2.5, 4.5, rows),
    "imu_std": np.random.uniform(0.2, 2.0, rows),
    "hr_mean": np.random.uniform(60, 110, rows),
    "hr_std": np.random.uniform(0.5, 8.0, rows)
})

labels = []

for i in range(rows):

    imu_std = data.loc[i,"imu_std"]
    hr_mean = data.loc[i,"hr_mean"]
    hr_std = data.loc[i,"hr_std"]

    if (imu_std < 0.9 and hr_mean < 85 and hr_std < 4):
        labels.append("Focused")
    else:
        labels.append("Distracted")

data["focus_state"] = labels


X = data[["imu_mean","imu_std","hr_mean","hr_std"]]
y = data["focus_state"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    ))
])


pipeline.fit(X_train,y_train)


pred = pipeline.predict(X_test)

print(classification_report(y_test,pred))


joblib.dump(pipeline,"model/focus_model.pkl")

print("Model saved")