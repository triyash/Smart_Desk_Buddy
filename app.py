from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import datetime

from config import MODEL_PATH
from database.db import init_db, insert_prediction, get_history
from utils.feature_extraction import extract_features
from utils.buffer import SensorBuffer


app = Flask(__name__)
CORS(app)

init_db()

model = joblib.load(MODEL_PATH)

buffer = SensorBuffer()


@app.route("/")
def home():
    return "Smart Desk Buddy API Running"


@app.route("/receive_sensor_data", methods=["POST"])
def receive_sensor_data():

    data = request.json

    if not data:
        return jsonify({"error":"No data provided"}),400

    required = ["ax","ay","az","heart_rate"]

    for r in required:
        if r not in data:
            return jsonify({"error":f"Missing {r}"}),400


    buffer.add(data)


    if not buffer.is_full():
        return jsonify({"status":"collecting"})


    features = extract_features(buffer.get())

    features_np = np.array([features])


    prediction = model.predict(features_np)[0]

    confidence = float(np.max(model.predict_proba(features_np)))


    timestamp = datetime.datetime.now().isoformat()


    insert_prediction((
        features[0],
        features[1],
        features[2],
        features[3],
        prediction,
        confidence,
        timestamp
    ))


    buffer.clear()


    return jsonify({
        "focus_state":prediction,
        "confidence":confidence
    })


@app.route("/focus_history")
def focus_history():

    rows = get_history()

    results = []

    for row in rows:
        results.append({
            "id":row[0],
            "imu_mean":row[1],
            "imu_std":row[2],
            "hr_mean":row[3],
            "hr_std":row[4],
            "prediction":row[5],
            "confidence":row[6],
            "timestamp":row[7]
        })

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)