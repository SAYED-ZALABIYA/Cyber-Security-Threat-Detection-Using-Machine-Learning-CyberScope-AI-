import gradio as gr
import numpy as np
import joblib

# تحميل النماذج
model = joblib.load("best_random_forest_model (2).pkl")
scaler = joblib.load("scaler.pkl")
labelen = joblib.load("label_encoder.pkl")

# دالة التنبؤ
def predict_cyber_threat(Destination_Port, Flow_Duration, Fwd_Packet_Length_Mean,
                         Bwd_Packet_Length_Mean, Flow_IAT_Mean, Bwd_Packets_per_s,
                         Average_Packet_Size):
    
    features = np.array([[Destination_Port, Flow_Duration, Fwd_Packet_Length_Mean,
                          Bwd_Packet_Length_Mean, Flow_IAT_Mean, Bwd_Packets_per_s,
                          Average_Packet_Size]])
    
    user_input_scaled = scaler.transform(features)
    predicted_index = model.predict(user_input_scaled)[0]
    predicted_label = labelen.inverse_transform([predicted_index])[0]
    confidence_score = np.max(model.predict_proba(user_input_scaled)[0]) * 100
    
    return f"Prediction: {predicted_label}, Confidence: {round(confidence_score, 2)}%"

# واجهة Gradio
iface = gr.Interface(
    fn=predict_cyber_threat,
    inputs=[
        gr.Number(label="Destination Port"),
        gr.Number(label="Flow Duration"),
        gr.Number(label="Fwd Packet Length Mean"),
        gr.Number(label="Bwd Packet Length Mean"),
        gr.Number(label="Flow IAT Mean"),
        gr.Number(label="Bwd Packets/s"),
        gr.Number(label="Average Packet Size"),
    ],
    outputs="text",
    title="Cyber Security Threat Detection",
    description="Enter flow features to detect possible threats.",
)

iface.launch()
