import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, Reshape, LSTM, Bidirectional, BatchNormalization, Attention
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import seaborn as sns

class EnhancedADRPredictor:
    def __init__(self, drug_name='Paracetamol', max_results=1000):
        self.drug_name = drug_name
        self.max_results = max_results
        self.base_url = "https://api.fda.gov/drug/event.json"
        self.model = None
        self.reaction_encoder = None
        self.scaler = StandardScaler()

    def fetch_openfda_data(self):
        """Fetch ADR reports from OpenFDA API."""
        params = {'search': f'patient.drug.medicinalproduct:"{self.drug_name}"',
                 'limit': min(1000, self.max_results), 'skip': 0}
        all_results = []
        max_retries = 3
        
        for _ in range(max_retries):
            try:
                response = requests.get(self.base_url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                results = data.get('results', [])
                if results:
                    all_results.extend(results)
                if len(all_results) >= self.max_results:
                    break
                params['skip'] += len(results)
                print(f"Fetched {len(all_results)} records so far...")
            except Exception as e:
                print(f"Error fetching data: {e}")
                continue
        return pd.DataFrame(self._process_results(all_results)) if all_results else self.generate_sample_data()

    def _process_results(self, results):
        """Process raw API results into structured data."""
        processed = []
        for event in results:
            try:
                reactions = [reac['reactionmeddrapt'] for reac in event.get('patient',{}).get('reaction',[])]
                concomitant_drugs = len(event.get('patient', {}).get('drug', []))
                entry = {
                    'age': float(event.get('patient',{}).get('patientonsetage',30)),
                    'sex': event.get('patient',{}).get('patientsex','unknown').lower(),
                    'weight': float(event.get('patient',{}).get('patientweight',70)),
                    'dosage': self._extract_dosage(event),
                    'reactions': reactions if reactions else ['unknown'],
                    'concomitant_drugs': concomitant_drugs if concomitant_drugs > 0 else 1,
                    'serious': 1 if any(int(v) for v in event.get('seriousness',{}).values()) else 0
                }
                processed.append(entry)
            except Exception as e:
                print(f"Error processing entry: {e}")
        return processed

    def _extract_dosage(self, event):
        """Extract dosage information from event data."""
        try:
            dosage_text = event.get('patient',{}).get('drug',[{}])[0].get('drugdosagetext','')
            if 'mg' in dosage_text.lower():
                dosage_value = ''.join(filter(str.isdigit, dosage_text.split('mg')[0]))
            else:
                dosage_value = ''.join(filter(str.isdigit, dosage_text))
            return float(dosage_value) if dosage_value and any(c.isdigit() for c in dosage_value) else 1.0
        except Exception:
            return 1.0

    def preprocess_data(self, df):
        """Preprocess and clean the dataset."""
        df['sex'] = df['sex'].map({'male':0, 'female':1}).fillna(0.5)
        df['reactions'] = df['reactions'].apply(lambda x: x if isinstance(x, list) else ['unknown'])
        all_reactions = list(set(reac for sublist in df['reactions'] for reac in sublist)) or ['unknown']
        self.reaction_encoder = {reac:i+1 for i,reac in enumerate(all_reactions)}
        df['reaction_seq'] = df['reactions'].apply(
            lambda x: [self.reaction_encoder.get(r,0) for r in x][:10] + [0]*(10-len(x))
        )
        df['reaction_count'] = df['reactions'].apply(len)
        df['bmi'] = df['weight'] / ((df['age']/100) ** 2)
        df['bmi'] = df['bmi'].clip(15, 45)
        
        numerical_features = ['age', 'weight', 'dosage', 'concomitant_drugs', 'reaction_count', 'bmi']
        df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())
        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        df['target'] = df['serious']
        
        if df['target'].nunique() < 2:
            print("Warning: Single-class dataset detected. Regenerating balanced sample data.")
            return self.generate_sample_data(force_balance=True)
        return df

    def generate_sample_data(self, num_samples=1000, force_balance=True):
        """Generate synthetic data when real data is unavailable."""
        np.random.seed(42)
        serious = np.concatenate([
            np.zeros(int(num_samples*0.7)),
            np.ones(int(num_samples*0.3))
        ]) if force_balance else np.random.choice([0,1], num_samples, p=[0.7,0.3])
        
        common_reactions = ['headache', 'nausea', 'vomiting', 'dizziness', 'rash', 'fatigue']
        serious_reactions = ['anaphylaxis', 'seizure', 'cardiac_arrest', 'liver_failure', 'renal_failure']
        reactions_list = []
        
        for i in range(num_samples):
            if serious[i] == 1:
                num_reactions = np.random.randint(1, 4)
                reactions = list(np.random.choice(serious_reactions, 1))
                reactions.extend(list(np.random.choice(common_reactions, num_reactions-1)))
            else:
                num_reactions = np.random.randint(1, 3)
                reactions = list(np.random.choice(common_reactions, num_reactions))
            reactions_list.append(reactions)
            
        return pd.DataFrame({
            'age': np.random.randint(18, 80, num_samples),
            'sex': np.random.choice(['male','female','unknown'], num_samples),
            'weight': np.random.uniform(50, 100, num_samples),
            'dosage': np.random.exponential(scale=50, size=num_samples),
            'concomitant_drugs': np.random.randint(1, 5, num_samples),
            'reactions': reactions_list,
            'serious': serious
        }).pipe(self.preprocess_data)

    def build_hybrid_model(self):
        """Build hybrid LSTM + structured data model architecture."""
        # Reaction sequence processing branch
        lstm_input = Input(shape=(10,), name='reaction_seq')
        x = Reshape((10, 1))(lstm_input)
        lstm1 = Bidirectional(LSTM(128, return_sequences=True))(x)
        bn1 = BatchNormalization()(lstm1)
        dropout1 = Dropout(0.3)(bn1)
        lstm2 = Bidirectional(LSTM(64, return_sequences=True))(dropout1)
        attention = Attention()([lstm2, lstm2])
        flattened = tf.keras.layers.Flatten()(attention)

        # Structured data processing branch
        struct_input = Input(shape=(6,), name='structured_features')
        y = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(struct_input)
        y = BatchNormalization()(y)
        y = Dropout(0.3)(y)
        y = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(y)

        # Combined model
        combined = concatenate([flattened, y])
        z = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(combined)
        z = BatchNormalization()(z)
        z = Dropout(0.3)(z)
        z = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(z)
        z = Dropout(0.2)(z)
        output = Dense(1, activation='sigmoid')(z)

        self.model = Model(inputs=[lstm_input, struct_input], outputs=output)

        # Focal loss for class imbalance
        def focal_loss(gamma=2., alpha=.25):
            def focal_loss_fixed(y_true, y_pred):
                pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
                pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
                return -tf.reduce_mean(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1 + 1e-7)) - \
                       tf.reduce_mean((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0 + 1e-7))
            return focal_loss_fixed

        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=['accuracy', 
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()]
        )

    def train_model(self, df):
        """Train model with cross-validation and class balancing."""
        X_seq = np.array(df['reaction_seq'].tolist())
        X_struct = df[['age', 'weight', 'dosage', 'concomitant_drugs', 'reaction_count', 'bmi']].values
        y = df['target'].values

        unique_classes, counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_classes, counts))}")
        if len(unique_classes) < 2:
            raise ValueError("Insufficient classes for training. Need at least 2 classes.")

        # Stratified k-fold cross-validation
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_struct, y)):
            print(f"\nTraining fold {fold+1}/{n_splits}")
            X_train_seq, X_test_seq = X_seq[train_idx], X_seq[test_idx]
            X_train_struct, X_test_struct = X_struct[train_idx], X_struct[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Apply SMOTE balancing
            smote = SMOTE(random_state=42)
            X_train_combined = np.concatenate([X_train_seq, X_train_struct], axis=1)
            X_train_combined_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train)
            X_train_seq_resampled = X_train_combined_resampled[:, :10]
            X_train_struct_resampled = X_train_combined_resampled[:, 10:]

            # Build and train model
            self.build_hybrid_model()
            history = self.model.fit(
                [X_train_seq_resampled, X_train_struct_resampled],
                y_train_resampled,
                validation_data=([X_test_seq, X_test_struct], y_test),
                epochs=100,
                batch_size=32,
                callbacks=[
                    EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max'),
                    ModelCheckpoint(f'adr_model_fold_{fold}.h5', monitor='val_auc', mode='max', save_best_only=True)
                ],
                verbose=1
            )

            # Evaluate performance
            y_pred_proba = self.model.predict([X_test_seq, X_test_struct])
            y_pred = (y_pred_proba > 0.5).astype(int)
            fold_auc = roc_auc_score(y_test, y_pred_proba)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            fold_auprc = auc(recall, precision)
            
            print(f"\nFold {fold+1} Classification Report:")
            print(classification_report(y_test, y_pred))
            print(f"AUC-ROC: {fold_auc:.4f}, AUPRC: {fold_auprc:.4f}")
            fold_metrics.append({'auc': fold_auc, 'auprc': fold_auprc, 'history': history.history})

        # Final model evaluation
        avg_auc = np.mean([m['auc'] for m in fold_metrics])
        avg_auprc = np.mean([m['auprc'] for m in fold_metrics])
        print(f"\nAverage AUC-ROC across {n_splits} folds: {avg_auc:.4f}")
        print(f"Average AUPRC across {n_splits} folds: {avg_auprc:.4f}")

        # Plot learning curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for i, m in enumerate(fold_metrics):
            plt.plot(m['history']['val_auc'], label=f'Fold {i+1}')
        plt.title('Validation AUC-ROC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC-ROC')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        for i, m in enumerate(fold_metrics):
            plt.plot(m['history']['val_loss'], label=f'Fold {i+1}')
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('learning_curves.png')
        plt.show()

    def explain_predictions(self, df, n_samples=10):
        """Generate SHAP explanations for model predictions."""
        sample_data = df.sample(n_samples)
        X_seq = np.array(sample_data['reaction_seq'].tolist())
        X_struct = sample_data[['age', 'weight', 'dosage', 'concomitant_drugs', 'reaction_count', 'bmi']].values

        # SHAP explanations
        explainer = shap.DeepExplainer(self.model, [X_seq, X_struct])
        shap_values = explainer.shap_values([X_seq, X_struct])

        # Feature importance visualization
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values[0], X_struct, feature_names=['Age', 'Weight', 'Dosage', 'Concomitant Drugs', 'Reaction Count', 'BMI'])
        plt.savefig('shap_summary.png')
        plt.show()

        plt.figure(figsize=(12, 8))
        feature_importance = np.abs(shap_values[0]).mean(axis=0)
        sns.heatmap(feature_importance.reshape(1, -1), annot=True,
                   xticklabels=['Age', 'Weight', 'Dosage', 'Concomitant Drugs', 'Reaction Count', 'BMI'],
                   yticklabels=['Importance'], cmap='viridis')
        plt.title('Feature Importance Heatmap')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()

    def full_pipeline(self):
        """Execute complete ADR detection pipeline."""
        print("Starting enhanced ADR detection pipeline...")
        try:
            df = self.fetch_openfda_data()
            df = self.preprocess_data(df)
            self.train_model(df)
            self.explain_predictions(df)
            self.model.save('adr_final_model.h5')
            print("Model saved successfully as adr_final_model.h5")
        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            print("Regenerating with synthetic data...")
            df = self.generate_sample_data(force_balance=True)
            self.train_model(df)
            self.explain_predictions(df)
            self.model.save('adr_final_model.h5')

if __name__ == "__main__":
    predictor = EnhancedADRPredictor(drug_name="Aspirin", max_results=5000)
    predictor.full_pipeline()
