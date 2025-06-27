from flask import Flask, request, jsonify
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# تحميل النماذج
model = joblib.load("best_random_forest_model (2).pkl")
scaler = joblib.load("scaler.pkl")
labelen = joblib.load("label_encoder.pkl")

@app.route("/", methods=["GET"])
def home():
    return "API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # استقبال البيانات
        data = request.get_json()
        required_fields = ['Destination Port', 'Flow Duration', 'Fwd Packet Length Mean',
                           'Bwd Packet Length Mean', 'Flow IAT Mean', 'Bwd Packets/s',
                           'Average Packet Size']
        
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        # استخراج القيم
        features = np.array([[data['Destination Port'],
                              data['Flow Duration'],
                              data['Fwd Packet Length Mean'],
                              data['Bwd Packet Length Mean'],
                              data['Flow IAT Mean'],
                              data['Bwd Packets/s'],
                              data['Average Packet Size']]])

        # التطبيع والتنبؤ
        user_input_scaled = scaler.transform(features)
        predicted_disease_index = model.predict(user_input_scaled)[0]
        predicted_disease_str = labelen.inverse_transform([predicted_disease_index])[0]
        probabilities = model.predict_proba(user_input_scaled)[0]
        confidence_score = np.max(probabilities)
      

        return jsonify({
            "predicted_label": predicted_disease_str,
            "confidence_score": round(float(confidence_score) * 100, 2)  # كنسبة مئوية
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
