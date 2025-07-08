# Enhanced ADR (Adverse Drug Reaction) Prediction System

An AI-powered web application that predicts the risk of adverse drug reactions based on patient-specific parameters and explains the prediction using SHAP visualizations.

---

## Features

- Predicts ADR risk level (High Risk / Low Risk)
- Visual explanations using SHAP heatmaps
- Hybrid deep learning model combining LSTM (reaction sequences) and dense layers (numerical features)
- Uses OpenFDA data or synthetic fallback
- Flask-based web interface



## Technologies Used

- **Python**, **Flask**, **NumPy**, **pandas**, **TensorFlow/Keras**
- **scikit-learn**, **SMOTE**, **SHAP**, **Matplotlib**, **Seaborn**
- **HTML/CSS/JS** for the frontend

---

## Input Parameters

Users input the following:

- Age
- Weight
- Dosage (mg)
- Concomitant Drugs (number of other medications taken)
- Observed Reactions (comma-separated)

---

## Output

- **Prediction**: High Risk / Low Risk of ADR
- **Risk Meter**: Visual bar showing confidence
- **SHAP Heatmaps**:
  - Reaction sequence influence
  - Feature impact explanation

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/adr-predictor.git
   cd adr-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   python app.py
   ```

4. Visit: `http://localhost:5000`

---

## Train the Model
To retrain with OpenFDA or synthetic data:
```bash
python model.py
```
The final model will be saved as `adr_final_model.h5`.

---

## Future Enhancements

- Upload patient data as CSV
- Real-time ADR database integration
- Deploy to cloud (Render / Railway / Hugging Face Spaces)
- Multi-drug interaction graph



## Disclaimer
This is an educational tool and **not a substitute for professional medical advice**.
Always consult a licensed healthcare provider.
