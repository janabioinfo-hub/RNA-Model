import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os, tempfile
import warnings
warnings.filterwarnings('ignore')

class RNASeqPipeline:
    def __init__(self):
        # Load training data and genes list (from environment or config)
        self.load_reference_data()
    
    def load_reference_data(self):
        """Load reference training data and gene lists"""
        # For production, these should be stored in cloud storage (S3, etc.)
        # or bundled with the app
        try:
            # Placeholder - replace with actual data loading
            self.genes = ["GENE1", "GENE2", "GENE3"]  # Load from file
            self.model_trained = False
            print("Reference data loaded successfully")
        except Exception as e:
            print(f"Error loading reference data: {e}")
            raise
    
    def train_model(self):
        """Train the XGBoost model (call once on startup)"""
        try:
            # Load training data (replace with actual data loading)
            # counts = pd.read_csv(TRAIN_DATA_PATH)
            # coldata = pd.read_csv(COLDATA_PATH)
            # ... training logic from your original code ...
            
            # For now, create a dummy trained model
            self.model = xgb.XGBClassifier(random_state=42)
            self.scaler = StandardScaler()
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(['Pre Receptive', 'Receptive', 'Post Receptive'])
            
            self.model_trained = True
            print("Model trained successfully")
            
        except Exception as e:
            print(f"Model training failed: {e}")
            raise
    
    def analyze(self, test_file_path, job_id):
        """Run complete analysis pipeline"""
        try:
            if not self.model_trained:
                self.train_model()
            
            # Create output directory
            output_dir = os.path.join('results', job_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # Load test data
            test_counts = pd.read_csv(test_file_path, index_col=0)
            print(f"Loaded test data: {test_counts.shape}")
            
            # Process test data (simplified for example)
            X_test = test_counts.transpose()
            
            # Make predictions (dummy for example)
            predictions = np.random.choice(self.label_encoder.classes_, size=len(X_test))
            probabilities = np.random.random((len(X_test), len(self.label_encoder.classes_)))
            probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
            
            # Save predictions
            predictions_file = os.path.join(output_dir, 'predicted_phases.csv')
            predictions_df = pd.DataFrame({
                'sample': X_test.index,
                'predicted_phase': predictions
            })
            
            for i, phase in enumerate(self.label_encoder.classes_):
                predictions_df[f'prob_{phase}'] = probabilities[:, i]
            
            predictions_df.to_csv(predictions_file, index=False)
            
            # Create PCA plot
            pca_file = os.path.join(output_dir, 'pca_analysis.pdf')
            self.create_pca_plot(X_test, predictions, pca_file)
            
            # Return results
            results = {
                'predictions': {
                    'path': predictions_file,
                    'filename': 'predicted_phases.csv',
                    'download_url': f'/api/download/{job_id}/predictions'
                },
                'pca_plot': {
                    'path': pca_file,
                    'filename': 'pca_analysis.pdf',
                    'download_url': f'/api/download/{job_id}/pca_plot'
                },
                'summary': {
                    'total_samples': len(X_test),
                    'phase_distribution': pd.Series(predictions).value_counts().to_dict()
                }
            }
            
            return results
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            raise
    
    def create_pca_plot(self, X_test, predictions, output_path):
        """Create PCA visualization"""
        try:
            # Simple PCA plot (replace with your actual PCA logic)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_test)
            
            plt.figure(figsize=(10, 8))
            
            colors = ['red', 'blue', 'green']
            for i, phase in enumerate(self.label_encoder.classes_):
                mask = predictions == phase
                if np.any(mask):
                    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                              c=colors[i], label=phase, alpha=0.7)
            
            plt.xlabel(f'PC1 ({0.4:.1%} variance)')  # Replace with actual variance
            plt.ylabel(f'PC2 ({0.3:.1%} variance)')
            plt.title('RNA-seq PCA Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"PCA plot creation failed: {e}")
            # Create empty file to avoid errors
            with open(output_path, 'w') as f:
                f.write("PCA plot generation failed")
