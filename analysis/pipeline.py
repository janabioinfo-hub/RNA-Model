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
import os, tempfile, requests, io
import warnings
warnings.filterwarnings('ignore')

class RNASeqPipeline:
    def __init__(self):
        """Initialize pipeline with Google Drive file IDs"""
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.pca = None
        self.genes = None
        self.model_trained = False
        
        # Google Drive file IDs - EXTRACTED FROM YOUR FULL URLS
        self.DRIVE_FILES = {
            'counts': '1FTxF7emBYL3KKA6Kh2O5Qmywuh9ebWNI',      # normalized_vst_train.csv
            'coldata': '1nt7qKj6TJwz9unP7ksjj-Z0sOrQqu3kI',     # Column_Data.csv
            'genes': '1g9ylVTqbKJ9LLNZL4eGd8qG6CW7af8SP'        # ensembl_to_gene_names.csv
        }
        
        # TEST: Verify file IDs are not placeholders
        for key, file_id in self.DRIVE_FILES.items():
            if 'YOUR_' in file_id or 'PUT_' in file_id or len(file_id) < 20:
                print(f"❌ ERROR: {key} file ID appears to be a placeholder: {file_id}")
                print("💡 Please update with actual Google Drive file IDs")
            else:
                print(f"✅ {key} file ID looks valid: {file_id[:10]}...")
        
        print("🧬 RNA-seq XGBoost Pipeline initialized with Google Drive integration")
    
    def _download_from_drive(self, file_id, filename):
        """Download file from Google Drive with large file support"""
        try:
            download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
            
            print(f"📥 Downloading {filename} from Google Drive...")
            
            session = requests.Session()
            response = session.get(download_url, stream=True)
            
            # Handle Google Drive virus scan warning for large files (>25MB)
            if response.status_code == 200:
                if 'virus scan warning' in response.text.lower() or len(response.content) < 1000:
                    # Extract confirmation token
                    import re
                    token_match = re.search(r'confirm=([^&]+)', response.text)
                    token = token_match.group(1) if token_match else 't'
                    
                    # Download with confirmation
                    confirm_url = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
                    response = session.get(confirm_url, stream=True)
            
            if response.status_code == 200:
                file_size = len(response.content) / (1024 * 1024)
                print(f"✅ Downloaded {filename} ({file_size:.1f} MB)")
                return response.content
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error downloading {filename}: {str(e)}")
            raise
    
    def load_reference_data(self):
        """Load training data with memory optimization"""
        try:
            print("🔄 Loading training data with memory optimization...")
            
            # Load genes first (needed for filtering)
            genes_data = self._download_from_drive(self.DRIVE_FILES['genes'], 'genes_list.csv')
            genes_df = pd.read_csv(io.StringIO(genes_data.decode('utf-8')), header=None)
            self.genes = genes_df.iloc[:, 0].tolist()
            print(f"Loaded {len(self.genes)} significant genes")
            
            # Download and load training counts with memory-efficient dtypes
            counts_data = self._download_from_drive(self.DRIVE_FILES['counts'], 'training_counts.csv')
            
            # Use float32 instead of float64 to halve memory usage
            self.counts = pd.read_csv(io.StringIO(counts_data.decode('utf-8')), 
                                     index_col=0, dtype='float32')
            
            # Immediately filter by genes to reduce memory footprint
            self.counts = self.counts.loc[self.counts.index.isin(self.genes)]
            print(f"Filtered to {self.counts.shape[0]} genes (memory optimization)")
            
            # Load column data
            coldata_data = self._download_from_drive(self.DRIVE_FILES['coldata'], 'column_data.csv')
            self.coldata = pd.read_csv(io.StringIO(coldata_data.decode('utf-8')), header=None)
            self.coldata.columns = ['sample', 'phase']
            self.coldata['sample'] = self.coldata['sample'].astype(str).str.strip()
            
            print(f"✅ Memory-optimized data loaded:")
            print(f"   - Counts: {self.counts.shape[0]} genes, {self.counts.shape[1]} samples")
            print(f"   - Using float32 dtype for memory efficiency")
            print(f"   - Phase distribution: {self.coldata['phase'].value_counts().to_dict()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading from Google Drive: {str(e)}")
            return False
    
    def oversample_to_majority(self, df, class_col):
        """Memory-efficient oversampling"""
        import gc
        
        class_counts = df[class_col].value_counts()
        max_count = class_counts.max()
        
        print(f"Majority class has {max_count} samples")
        print("Memory-efficient oversampling...")
        
        # Separate features and labels
        features = df.drop(columns=['sample', class_col])
        labels = df[class_col]
        
        # Use smaller sampling strategy to reduce memory usage
        over_sampler = RandomOverSampler(
            sampling_strategy='not majority',
            random_state=42
        )
        
        X_res, y_res = over_sampler.fit_resample(features, labels)
        
        # Force garbage collection
        del features, labels
        gc.collect()
        
        # Create new balanced dataframe
        balanced_df = pd.DataFrame(X_res, columns=df.drop(columns=['sample', class_col]).columns)
        balanced_df[class_col] = y_res
        
        return balanced_df
    
    def train_model(self):
        """Train XGBoost model with memory optimizations"""
        try:
            print("🤖 Training XGBoost model with memory optimization...")
            
            # Load reference data if not already loaded
            if not hasattr(self, 'counts') or self.counts is None:
                if not self.load_reference_data():
                    raise Exception("Failed to load training data from Google Drive")
            
            # Filter counts by significant genes
            counts_filtered = self.counts.loc[self.counts.index.isin(self.genes)]
            
            # Transpose to have samples as rows, genes as columns
            X_full = counts_filtered.transpose()
            
            # Ensure sample index is string type and stripped of whitespace
            X_full.index = X_full.index.astype(str).str.strip()
            
            # Merge with phase information
            data = X_full.merge(self.coldata, left_index=True, right_on='sample')
            
            print(f"Phase distribution before oversampling:")
            print(data['phase'].value_counts())
            
            # Balance the dataset using memory-efficient oversampling
            balanced_data = self.oversample_to_majority(data, 'phase')
            
            print(f"Phase distribution after oversampling:")
            print(balanced_data['phase'].value_counts())
            
            # Prepare features and labels
            X_balanced = balanced_data.drop(columns=['phase'])
            y_balanced = balanced_data['phase']
            
            # Encode phase labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y_balanced)
            
            print(f"Phase encoding:")
            for i, phase in enumerate(self.label_encoder.classes_):
                print(f"  {phase} -> {i}")
            
            # Train-Validation Split with memory consideration
            X_train, X_val, y_train, y_val = train_test_split(
                X_balanced, y_encoded,
                test_size=0.2,  # Reduced from 0.25 to save memory
                stratify=y_encoded,
                random_state=42
            )
            
            print(f"\nTraining set: {X_train.shape[0]} samples")
            print(f"Validation set: {X_val.shape[0]} samples")
            
            # Standardize features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train XGBoost with memory-efficient parameters
            self.model = xgb.XGBClassifier(
                eval_metric='mlogloss',
                random_state=42,
                n_estimators=30,        # Further reduced for memory
                max_depth=3,           # Reduced from 4
                learning_rate=0.1,
                reg_alpha=0.1,
                reg_lambda=0.1,
                tree_method='hist',    # More memory efficient
                max_bin=32,           # Fewer bins = less memory
                subsample=0.7,        # Use subset of data
                colsample_bytree=0.8  # Use subset of features
            )
            
            print("Training model...")
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate on validation set
            y_pred_val = self.model.predict(X_val_scaled)
            self.val_accuracy = accuracy_score(y_val, y_pred_val)
            
            print(f"Validation Accuracy: {self.val_accuracy:.4f}")
            print("\nValidation Classification Report:")
            print(classification_report(y_val, y_pred_val, target_names=self.label_encoder.classes_))
            
            # PCA Setup for Visualization
            self.pca = PCA(n_components=2)
            self.pca.fit(X_train_scaled)
            self.train_pca = self.pca.transform(X_train_scaled)
            self.val_pca = self.pca.transform(X_val_scaled)
            
            # Store training data for PCA plots
            self.y_train = y_train
            self.y_val = y_val
            
            print(f"PCA explained variance ratio: PC1={self.pca.explained_variance_ratio_[0]:.3f}, PC2={self.pca.explained_variance_ratio_[1]:.3f}")
            
            self.model_trained = True
            print("✅ Model trained successfully!")
            
            # Clean up memory
            import gc
            del X_train_scaled, X_val_scaled, X_balanced, balanced_data
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def analyze(self, test_file_path, job_id):
        """Run complete analysis pipeline"""
        try:
            print(f"🔬 Starting analysis for job {job_id}")
            
            # Train model if not already trained
            if not self.model_trained:
                if not self.train_model():
                    raise Exception("Model training failed")
            
            # Create output directory
            output_dir = os.path.join('results', job_id)
            os.makedirs(output_dir, exist_ok=True)
            
            print("Loading test data for prediction...")
            
            # Load test counts
            test_counts = pd.read_csv(test_file_path, index_col=0, dtype='float32')
            
            # Filter by significant genes and transpose
            test_counts_filtered = test_counts.loc[test_counts.index.isin(self.genes)]
            X_test = test_counts_filtered.transpose()
            
            print(f"Test data: {X_test.shape[0]} samples, {X_test.shape[1]} genes")
            
            # Standardize test data
            X_test_scaled = self.scaler.transform(X_test)
            
            # Make predictions
            test_predictions_encoded = self.model.predict(X_test_scaled)
            test_predictions = self.label_encoder.inverse_transform(test_predictions_encoded)
            
            # Get prediction probabilities
            test_probabilities = self.model.predict_proba(X_test_scaled)
            
            print(f"Test predictions summary:")
            print(pd.Series(test_predictions).value_counts())
            
            # Create predictions dataframe
            predictions_df = pd.DataFrame({
                'sample': X_test.index,
                'predicted_phase': test_predictions
            })
            
            # Add probability columns
            for i, phase in enumerate(self.label_encoder.classes_):
                predictions_df[f'prob_{phase}'] = test_probabilities[:, i]
            
            # Save to CSV
            predictions_file = os.path.join(output_dir, 'predicted_phases.csv')
            predictions_df.to_csv(predictions_file, index=False)
            
            print("Performing PCA analysis...")
            
            # Project test data onto training PCA space
            test_pca = self.pca.transform(X_test_scaled)
            
            # Create PCA plots
            individual_pca_file = os.path.join(output_dir, 'individual_test_pca_plots.pdf')
            summary_pca_file = os.path.join(output_dir, 'test_samples_summary_pca.pdf')
            
            self._create_individual_pca_plots(X_test, test_predictions, test_probabilities, 
                                            test_pca, individual_pca_file)
            
            self._create_summary_pca_plot(X_test, test_predictions, test_pca, summary_pca_file)
            
            # Return results
            results = {
                'predictions': {
                    'path': predictions_file,
                    'filename': 'predicted_phases.csv',
                    'download_url': f'/api/download/{job_id}/predictions'
                },
                'pca_individual': {
                    'path': individual_pca_file,
                    'filename': 'individual_test_pca_plots.pdf', 
                    'download_url': f'/api/download/{job_id}/pca_individual'
                },
                'pca_summary': {
                    'path': summary_pca_file,
                    'filename': 'test_samples_summary_pca.pdf',
                    'download_url': f'/api/download/{job_id}/pca_summary'
                },
                'summary': {
                    'total_samples': len(X_test),
                    'total_genes': len(self.genes),
                    'validation_accuracy': round(self.val_accuracy, 4),
                    'phase_distribution': pd.Series(test_predictions).value_counts().to_dict()
                }
            }
            
            print("✅ Analysis completed successfully!")
            print("="*60)
            print("ANALYSIS COMPLETE")
            print("="*60)
            print(f"Model validation accuracy: {self.val_accuracy:.4f}")
            print(f"Test samples predicted: {len(test_predictions)}")
            print("Files generated:")
            print(f"  - Predictions CSV: {predictions_file}")
            print(f"  - Individual PCA PDF: {individual_pca_file}")
            print(f"  - Summary PCA PDF: {summary_pca_file}")
            print("="*60)
            
            return results
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_individual_pca_plots(self, X_test, test_predictions, test_probabilities, test_pca, output_path):
        """Create individual PCA plots for each test sample"""
        try:
            print("Creating individual PCA plots for each test sample...")
            
            # Get training phase data
            train_phases = self.label_encoder.inverse_transform(self.y_train)
            colors = sns.color_palette("Set1", n_colors=len(self.label_encoder.classes_))
            phase_colors = {phase: colors[i] for i, phase in enumerate(self.label_encoder.classes_)}
            
            # Create multi-page PDF to save all plots
            with PdfPages(output_path) as pdf:
                
                # First page: Overview PCA plot with all training data
                plt.figure(figsize=(10, 8))
                
                # Plot training samples colored by phase
                for phase in self.label_encoder.classes_:
                    phase_mask = train_phases == phase
                    plt.scatter(self.train_pca[phase_mask, 0], self.train_pca[phase_mask, 1],
                               c=[phase_colors[phase]], s=60, alpha=0.7, label=f'{phase} (Training)')
                
                # Plot validation samples
                if hasattr(self, 'val_pca') and len(self.val_pca) > 0:
                    val_phases = self.label_encoder.inverse_transform(self.y_val)
                    for phase in self.label_encoder.classes_:
                        phase_mask = val_phases == phase
                        if np.any(phase_mask):
                            plt.scatter(self.val_pca[phase_mask, 0], self.val_pca[phase_mask, 1],
                                       c=[phase_colors[phase]], s=60, alpha=0.5,
                                       marker='s', label=f'{phase} (Validation)')
                
                plt.title("Training Data PCA Space\n(Reference for Individual Test Sample Projections)")
                plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
                plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                pdf.savefig()
                plt.close()
                
                # Individual test sample plots
                for i, sample_name in enumerate(X_test.index):
                    plt.figure(figsize=(10, 8))
                    
                    # Plot all training samples as background (lighter colors)
                    for phase in self.label_encoder.classes_:
                        phase_mask = train_phases == phase
                        plt.scatter(self.train_pca[phase_mask, 0], self.train_pca[phase_mask, 1],
                                   c=[phase_colors[phase]], s=40, alpha=0.3,
                                   label=f'{phase} (Training)')
                    
                    # Highlight the current test sample
                    test_point = test_pca[i]
                    predicted_phase = test_predictions[i]
                    
                    plt.scatter(test_point[0], test_point[1],
                               c='red', s=200, marker='*',
                               edgecolors='black', linewidth=2,
                               label=f'{sample_name}\n(Predicted: {predicted_phase})')
                    
                    # Add prediction probability info
                    prob_text = f"Prediction Probabilities:\n"
                    for j, phase in enumerate(self.label_encoder.classes_):
                        prob_text += f"{phase}: {test_probabilities[i,j]:.3f}\n"
                    
                    plt.text(0.02, 0.98, prob_text, transform=plt.gca().transAxes,
                            fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    plt.title(f"Test Sample: {sample_name}\nPredicted Phase: {predicted_phase}")
                    plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
                    plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    pdf.savefig()
                    plt.close()
            
            print(f"Individual PCA plots saved to: {output_path}")
            print(f"Total pages in PDF: {len(X_test) + 1} (1 training overview + {len(X_test)} individual test samples)")
            
        except Exception as e:
            print(f"❌ Individual PCA plot creation failed: {e}")
            # Create placeholder file
            with open(output_path, 'w') as f:
                f.write("Individual PCA plot generation failed")
    
    def _create_summary_pca_plot(self, X_test, test_predictions, test_pca, output_path):
        """Create summary plot showing all test samples together"""
        try:
            # Get training phase data
            train_phases = self.label_encoder.inverse_transform(self.y_train)
            colors = sns.color_palette("Set1", n_colors=len(self.label_encoder.classes_))
            phase_colors = {phase: colors[i] for i, phase in enumerate(self.label_encoder.classes_)}
            
            plt.figure(figsize=(12, 10))
            
            # Plot training background
            for phase in self.label_encoder.classes_:
                phase_mask = train_phases == phase
                plt.scatter(self.train_pca[phase_mask, 0], self.train_pca[phase_mask, 1],
                           c=[phase_colors[phase]], s=40, alpha=0.3, label=f'{phase} (Training)')
            
            # Plot all test samples with labels
            for i, sample_name in enumerate(X_test.index):
                test_point = test_pca[i]
                predicted_phase = test_predictions[i]
                
                plt.scatter(test_point[0], test_point[1],
                           c=[phase_colors[predicted_phase]], s=150, marker='D',
                           edgecolors='black', linewidth=1, alpha=0.9)
                
                # Add sample name as text annotation
                plt.annotate(sample_name, (test_point[0], test_point[1]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, alpha=0.8)
            
            plt.title("All Test Samples Projected onto Training PCA Space")
            plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.legend(title="Training Phases", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Summary plot of all test samples saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Summary PCA plot creation failed: {e}")
            # Create placeholder file
            with open(output_path, 'w') as f:
                f.write("Summary PCA plot generation failed")
