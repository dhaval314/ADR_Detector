from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from tensorflow.keras.models import load_model
import json
import io
import base64

app = Flask(__name__)

# Load trained model
model = load_model('model/adr_model_fold_0.h5', compile=False)
model.make_predict_function()  # Required for thread-safety

# Initialize SHAP explainer
background = [np.zeros((1, 10)), np.zeros((1, 6))]
explainer = shap.GradientExplainer(model, background)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Process reactions
        reactions = [r.strip().lower() for r in data['reactions'].split(',')]
        reaction_encoder = {'headache':1, 'nausea':2, 'vomiting':3, 'dizziness':4, 
                          'rash':5, 'anaphylaxis':6, 'seizure':7, 'unknown':0}
        reaction_seq = [reaction_encoder.get(r, 0) for r in reactions][:10] + [0]*(10-len(reactions))
        
        # Calculate BMI
        bmi = float(data['weight']) / ((float(data['age'])/100)**2)
        bmi = max(15, min(45, bmi))
        
        # Create structured features
        struct_features = np.array([[
            float(data['age']),
            float(data['weight']),
            float(data['dosage']),
            len(reactions),
            bmi,
            int(data['concomitant_drugs'])
        ]])
        
        # Prepare model inputs
        model_inputs = [
            np.array([reaction_seq]),
            struct_features
        ]
        
        # Make prediction
        prediction = model.predict(model_inputs)[0][0]
        
        # Generate SHAP explanation
        shap_values = explainer.shap_values(model_inputs)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[1], struct_features,
                         feature_names=['Age', 'Weight', 'Dosage', 'Reaction Count', 'BMI', 'Concomitant Drugs'],
                         plot_type='bar', show=False)
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight')
        img_buf.seek(0)
        plt.close()
        
        return jsonify({
            'prediction': float(prediction),
            'shap_plot': base64.b64encode(img_buf.read()).decode('utf-8'),
            'risk_level': 'High Risk' if prediction > 0.5 else 'Low Risk'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
