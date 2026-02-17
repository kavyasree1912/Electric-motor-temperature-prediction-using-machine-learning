from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import traceback
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# -------------------- Load Models --------------------
try:
    manual_model = joblib.load("best_model.pkl")
    manual_scaler = joblib.load("scaler.pkl")
    print("✅ Manual model and scaler loaded.")
except Exception as e:
    print("❌ Error loading manual model or scaler:", e)

try:
    sensor_model = joblib.load("sensor_model.pkl")
    sensor_scaler = joblib.load("sensor_scaler.pkl")
    print("✅ Sensor model and scaler loaded.")
except Exception as e:
    print("❌ Error loading sensor model or scaler:", e)

# -------------------- Routes --------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Manual')
def manual_page():
    return render_template('Manual.html')

@app.route('/sensor')
def sensor_page():
    return render_template('sensor.html')

# --------- Manual Prediction (JS Frontend API) ---------
@app.route("/predict", methods=["POST"])
def predict_manual():
    try:
        data = request.get_json()
        features = data["features"]
        features_array = np.array([features])
        scaled_features = manual_scaler.transform(features_array)
        prediction = manual_model.predict(scaled_features)[0]
        return jsonify({"prediction": round(prediction, 2)})
    except Exception as e:
        print("Manual prediction error:", e)
        print(traceback.format_exc())
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

# --------- Sensor Prediction (HTML Form) ---------
@app.route('/predict_sensor', methods=['POST'])
def predict_sensor():
    try:
        features = [
            float(request.form['ambient']),
            float(request.form['torque']),
            float(request.form['coolant']),
            float(request.form['u_d']),
            float(request.form['u_q']),
            float(request.form['motor_speed']),
            float(request.form['i_d']),
            float(request.form['i_q'])
        ]
        final_features = np.array(features).reshape(1, -1)
        scaled_features = sensor_scaler.transform(final_features)
        prediction = sensor_model.predict(scaled_features)
        output = round(prediction[0], 2)
        return render_template('sensor.html', prediction=f'Motor Temp: {output} °C')
    except Exception as e:
        return render_template('sensor.html', prediction=f'Error: {e}')
    
if __name__ == '__main__':
    app.run(debug=True)

