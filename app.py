from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import numpy as np
import pickle
import os
import io
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to store model components
model_components = None

def load_model_components():
    """Load all model components once at startup"""
    global model_components
    
    if model_components is not None:
        return model_components
    
    print("Loading model components...")
    components = {}
    
    model_dir = "models/"
    
    try:
        # Load all pickle files
        with open(f"{model_dir}/xgboost_model.pkl", 'rb') as f:
            components['model'] = pickle.load(f)
        
        with open(f"{model_dir}/scaler.pkl", 'rb') as f:
            components['scaler'] = pickle.load(f)
        
        with open(f"{model_dir}/label_encoder.pkl", 'rb') as f:
            components['label_encoder'] = pickle.load(f)
        
        with open(f"{model_dir}/pca_model.pkl", 'rb') as f:
            components['pca'] = pickle.load(f)
        
        with open(f"{model_dir}/selected_features.pkl", 'rb') as f:
            components['selected_features'] = pickle.load(f)
        
        with open(f"{model_dir}/normalization_params.pkl", 'rb') as f:
            components['norm_params'] = pickle.load(f)
        
        print("✓ All model components loaded successfully!")
        model_components = components
        return components
        
    except Exception as e:
        print(f"Error loading model components: {str(e)}")
        return None

def preprocess_uploaded_data(file_path, selected_features):
    """Preprocess uploaded CSV data"""
    try:
        # Try different CSV reading strategies
        try:
            # First attempt: standard CSV with index
            test_counts = pd.read_csv(file_path, index_col=0)
        except:
            try:
                # Second attempt: CSV without index
                test_counts = pd.read_csv(file_path)
                test_counts = test_counts.set_index(test_counts.columns[0])
            except:
                raise ValueError("Unable to read CSV file. Please check the format.")
        
        print(f"Loaded data shape: {test_counts.shape}")
        
        # Filter by selected features (genes) and transpose
        available_genes = test_counts.index.intersection(selected_features)
        if len(available_genes) == 0:
            raise ValueError("No matching genes found in the uploaded file")
        
        print(f"Found {len(available_genes)}/{len(selected_features)} matching genes")
        
        test_counts_filtered = test_counts.loc[available_genes]
        X_test = test_counts_filtered.transpose()
        
        # Handle missing features
        missing_features = set(selected_features) - set(X_test.columns)
        if missing_features:
            print(f"Adding {len(missing_features)} missing features with zero values")
            for feature in missing_features:
                X_test[feature] = 0.0
        
        # Reorder columns to match training data
        X_test = X_test[selected_features]
        
        # Convert to numeric and handle any remaining issues
        X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return X_test
        
    except Exception as e:
        raise ValueError(f"Error preprocessing data: {str(e)}")

def make_predictions(X_test, components):
    """Make predictions on preprocessed data"""
    try:
        # Scale features
        X_test_scaled = components['scaler'].transform(X_test)
        
        # Make predictions
        predictions_encoded = components['model'].predict(X_test_scaled)
        predictions = components['label_encoder'].inverse_transform(predictions_encoded)
        
        # Get prediction probabilities
        probabilities = components['model'].predict_proba(X_test_scaled)
        
        # Calculate confidence (max probability)
        confidence = np.max(probabilities, axis=1)
        
        return predictions, probabilities, confidence
        
    except Exception as e:
        raise ValueError(f"Error making predictions: {str(e)}")

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'})
        
        # Load model components
        components = load_model_components()
        if components is None:
            return jsonify({'error': 'Model components not loaded. Please check server configuration.'})
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        print(f"Processing file: {filename}")
        
        # Preprocess data
        X_test = preprocess_uploaded_data(file_path, components['selected_features'])
        
        # Make predictions
        predictions, probabilities, confidence = make_predictions(X_test, components)
        
        # Format results
        results = []
        phase_names = components['label_encoder'].classes_
        
        for i, sample_name in enumerate(X_test.index):
            # Create probability dictionary dynamically
            prob_dict = {}
            for j, phase in enumerate(phase_names):
                prob_key = f'prob_{phase.lower().replace(" ", "_").replace("-", "_")}'
                prob_dict[prob_key] = float(probabilities[i, j])
            
            result = {
                'sample': str(sample_name),
                'predicted_phase': str(predictions[i]),
                'confidence': float(confidence[i]),
                **prob_dict
            }
            results.append(result)
        
        # Calculate summary statistics
        prediction_counts = pd.Series(predictions).value_counts().to_dict()
        summary = {
            'total_samples': len(predictions),
            **{phase.lower().replace(" ", "_").replace("-", "_"): prediction_counts.get(phase, 0) 
               for phase in phase_names}
        }
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify({
            'success': True,
            'predictions': results,
            'summary': summary,
            'model_info': {
                'total_features': len(components['selected_features']),
                'validation_accuracy': components['norm_params'].get('validation_accuracy', 'N/A')
            }
        })
        
    except Exception as e:
        # Clean up file if it exists
        try:
            if 'file_path' in locals():
                os.remove(file_path)
        except:
            pass
        
        return jsonify({'error': f'Processing error: {str(e)}'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    components = load_model_components()
    if components:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'features_count': len(components['selected_features'])
        })
    else:
        return jsonify({
            'status': 'error',
            'model_loaded': False
        }), 500

if __name__ == '__main__':
    # Load model components at startup
    components = load_model_components()
    if components is None:
        print("ERROR: Could not load model components!")
        print("Please ensure all .pkl files are in the 'models/' directory")
    else:
        print("🎉 Model components loaded successfully!")
        print(f"📊 Ready to predict with {len(components['selected_features'])} features")
        print(f"🧬 Phase classes: {', '.join(components['label_encoder'].classes_)}")
    
    # Run the Flask app
    print("\n🚀 Starting RNA-seq XGBoost Predictor...")
    print("📡 Access your app at the forwarded port URL in Codespaces")
    app.run(host='0.0.0.0', port=5000, debug=True)
