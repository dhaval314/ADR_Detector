from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import shap
from tensorflow.keras.models import load_model
import io
import base64

app = Flask(__name__)

# Load trained model
model = load_model('adr_model_saved.keras')

# SHAP explainer initialization (use GradientExplainer for TensorFlow eager mode)
dummy_reaction_seq = np.zeros((1, 10))  # Dummy input for reaction sequence
dummy_struct_features = np.zeros((1, 6))  # Dummy input for structured features
explainer = shap.GradientExplainer(model, [dummy_reaction_seq, dummy_struct_features])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data from request JSON
        data = request.json

        # Feature engineering: calculate BMI and prepare structured features
        bmi = float(data['weight']) / ((float(data['age']) / 100) ** 2)
        structured_features = np.array([[
            float(data['age']),
            float(data['weight']),
            float(data['dosage']),
            len(data['reactions'].split(',')),
            bmi,
            int(data['concomitant_drugs'])
        ]])

        # Prepare reaction sequence (dummy zero-padded sequence for simplicity)
        reaction_sequence = np.zeros((1, 10))  # Replace with actual reaction encoding if available

        # Generate prediction using the trained model
        combined_input = np.concatenate([reaction_sequence, structured_features], axis=1)
        prediction_prob = model.predict(structured_features)
 

        # Generate SHAP explanation values for feature importance visualization
        shap_values = explainer.shap_values([reaction_sequence, structured_features])

        # Create SHAP summary plot and save it to a buffer
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values[1], structured_features,
                          feature_names=['Age', 'Weight', 'Dosage', 'Reaction Count', 'BMI', 'Concomitant Drugs'],
                          plot_type='bar',
                          show=False)

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)

        # Encode plot image to Base64 for sending to frontend
        shap_plot_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

        return jsonify({
            'prediction': float(prediction_prob),
            'shap_plot': shap_plot_base64,
            'risk_level': "High" if prediction_prob > 0.5 else "Low"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
