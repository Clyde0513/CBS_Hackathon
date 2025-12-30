"""
Multiple model ensemble with advanced features and hyperparameter optimization
"""

import numpy as np
import pandas as pd
from collections import Counter
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

print("Libraries loaded successfully!")

class AdvancedDNAFeatureExtractor:
    """Comprehensive DNA sequence feature extraction with reverse complement"""
    
    def __init__(self):
        self.nucleotides = ['A', 'C', 'G', 'T']
        self.complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    
    def reverse_complement(self, sequence):
        """Get reverse complement of sequence"""
        return ''.join([self.complement[n] for n in sequence[::-1]])
    
    def basic_composition(self, sequence):
        """Enhanced nucleotide composition features"""
        length = len(sequence)
        a_count = sequence.count('A') / length
        c_count = sequence.count('C') / length
        g_count = sequence.count('G') / length
        t_count = sequence.count('T') / length
        
        gc_content = g_count + c_count
        at_content = a_count + t_count
        gc_skew = (g_count - c_count) / (g_count + c_count + 1e-10)
        at_skew = (a_count - t_count) / (a_count + t_count + 1e-10)
        purine_content = a_count + g_count  # A, G
        pyrimidine_content = c_count + t_count  # C, T
        
        return np.array([a_count, c_count, g_count, t_count, 
                        gc_content, at_content, gc_skew, at_skew,
                        purine_content, pyrimidine_content], dtype=np.float32)
    
    def kmer_frequencies(self, sequence, k):
        """K-mer frequency features"""
        kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
        kmer_counts = Counter(kmers)
        
        all_kmers = [''.join(p) for p in product(self.nucleotides, repeat=k)]
        features = np.array([kmer_counts.get(kmer, 0) for kmer in all_kmers], dtype=np.float32)
        
        return features / (len(sequence) - k + 1)
    
    def positional_features(self, sequence, n_windows=10):
        """Extract features from sequence windows - more detailed"""
        window_size = len(sequence) // n_windows
        features = []
        
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size if i < n_windows - 1 else len(sequence)
            window = sequence[start:end]
            
            # Multiple features per window
            gc = (window.count('G') + window.count('C')) / len(window)
            purine = (window.count('A') + window.count('G')) / len(window)
            
            features.extend([gc, purine])
        
        return np.array(features, dtype=np.float32)
    
    def sequence_complexity(self, sequence):
        """Enhanced sequence complexity features"""
        length = len(sequence)
        
        # Shannon entropy
        counts = np.array([sequence.count(n) for n in self.nucleotides])
        probs = counts / length
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Linguistic complexity for different k-mers
        complexities = []
        for k in [2, 3, 4, 5]:
            kmers = set([sequence[i:i+k] for i in range(len(sequence) - k + 1)])
            complexity = len(kmers) / (len(sequence) - k + 1)
            complexities.append(complexity)
        
        # CpG islands
        cg_count = sequence.count('CG')
        cg_ratio = cg_count / (length - 1) if length > 1 else 0
        
        # Tandem repeats indicators
        max_tandem = 0
        for k in [2, 3, 4]:
            for i in range(len(sequence) - k*2):
                kmer = sequence[i:i+k]
                if sequence[i+k:i+k*2] == kmer:
                    max_tandem = max(max_tandem, k)
        
        return np.array([entropy] + complexities + [cg_ratio, max_tandem/10], dtype=np.float32)
    
    def dinucleotide_transitions(self, sequence):
        """Enhanced dinucleotide transition features"""
        transitions = {}
        for n1 in self.nucleotides:
            for n2 in self.nucleotides:
                dinuc = n1 + n2
                transitions[dinuc] = sequence.count(dinuc)
        
        features = np.array(list(transitions.values()), dtype=np.float32) / (len(sequence) - 1)
        return features
    
    def structural_features(self, sequence):
        """DNA structural property features"""
        # Runs of same nucleotide
        max_runs = {}
        avg_runs = {}
        
        for nuc in self.nucleotides:
            runs = []
            current_run = 0
            for n in sequence:
                if n == nuc:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                runs.append(current_run)
            
            max_runs[nuc] = max(runs) if runs else 0
            avg_runs[nuc] = np.mean(runs) if runs else 0
        
        max_features = np.array(list(max_runs.values()), dtype=np.float32) / len(sequence)
        avg_features = np.array(list(avg_runs.values()), dtype=np.float32) / len(sequence)
        
        return np.concatenate([max_features, avg_features])
    
    def palindrome_features(self, sequence):
        """Check for palindromic sequences (important for DNA binding)"""
        palindrome_counts = []
        
        for k in [4, 6, 8]:
            count = 0
            for i in range(len(sequence) - k + 1):
                kmer = sequence[i:i+k]
                rev_comp = self.reverse_complement(kmer)
                if kmer == rev_comp:
                    count += 1
            palindrome_counts.append(count / (len(sequence) - k + 1))
        
        return np.array(palindrome_counts, dtype=np.float32)
    
    def extract_all_features(self, sequence):
        """Extract comprehensive feature set"""
        features = []
        
        # Basic composition (10 features)
        features.append(self.basic_composition(sequence))
        
        # K-mer frequencies
        features.append(self.kmer_frequencies(sequence, k=2))  # 16
        features.append(self.kmer_frequencies(sequence, k=3))  # 64
        features.append(self.kmer_frequencies(sequence, k=4))  # 256
        features.append(self.kmer_frequencies(sequence, k=5))  # 1024
        
        # Positional features (20)
        features.append(self.positional_features(sequence, n_windows=10))
        
        # Complexity features (7)
        features.append(self.sequence_complexity(sequence))
        
        # All dinucleotide transitions (16)
        features.append(self.dinucleotide_transitions(sequence))
        
        # Structural features (8)
        features.append(self.structural_features(sequence))
        
        # Palindrome features (3)
        features.append(self.palindrome_features(sequence))
        
        return np.concatenate(features).astype(np.float32)

def load_data():
    """Load training and test data"""
    print("Loading training sequences...")
    with open('trainsequences.csv', 'r') as f:
        train_sequences = [line.strip() for line in f]
    
    print("Loading training labels...")
    train_labels = pd.read_csv('trainlabels.csv', header=None).values.ravel()
    
    print("Loading test sequences...")
    with open('testsequences.csv', 'r') as f:
        test_sequences = [line.strip() for line in f]
    
    print(f"Training samples: {len(train_sequences)}")
    print(f"Test samples: {len(test_sequences)}")
    print(f"Unique labels: {len(np.unique(train_labels))}")
    
    return train_sequences, train_labels, test_sequences

def encode_sequences(sequences, extractor):
    """Encode all sequences with enhanced features"""
    print(f"Encoding {len(sequences)} sequences...")
    encoded = []
    for i, seq in enumerate(sequences):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(sequences)} sequences")
        encoded.append(extractor.extract_all_features(seq))
    
    return np.array(encoded, dtype=np.float32)

def train_xgboost_model(X_train, y_train):
    """Train XGBoost with optimized parameters"""
    print("\nTraining XGBoost model...")
    
    y_train_xgb = y_train - 1
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=12,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.7,
        objective='multi:softmax',
        num_class=18,
        tree_method='hist',
        gamma=0.1,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbosity=1
    )
    
    xgb_model.fit(X_train, y_train_xgb)
    train_acc = xgb_model.score(X_train, y_train_xgb)
    print(f"XGBoost Training accuracy: {train_acc:.4f}")
    
    return xgb_model

def train_lightgbm_model(X_train, y_train):
    """Train LightGBM model"""
    print("\nTraining LightGBM model...")
    
    y_train_lgb = y_train - 1
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=12,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.7,
        num_class=18,
        objective='multiclass',
        num_leaves=50,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    
    lgb_model.fit(X_train, y_train_lgb)
    train_acc = lgb_model.score(X_train, y_train_lgb)
    print(f"LightGBM Training accuracy: {train_acc:.4f}")
    
    return lgb_model

def train_random_forest_model(X_train, y_train):
    """Train Random Forest with optimized parameters"""
    print("\nTraining Random Forest model...")
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        max_samples=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    rf_model.fit(X_train, y_train)
    train_acc = rf_model.score(X_train, y_train)
    print(f"Random Forest Training accuracy: {train_acc:.4f}")
    
    return rf_model

def train_extra_trees_model(X_train, y_train):
    """Train Extra Trees model"""
    print("\nTraining Extra Trees model...")
    
    et_model = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        max_samples=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    et_model.fit(X_train, y_train)
    train_acc = et_model.score(X_train, y_train)
    print(f"Extra Trees Training accuracy: {train_acc:.4f}")
    
    return et_model

def advanced_ensemble(predictions_dict):
    """
    Advanced weighted ensemble of multiple models
    """
    # Optimal weights found through experimentation
    weights = {
        'xgb': 0.30,
        'lgb': 0.30,
        'rf': 0.25,
        'et': 0.15
    }
    
    combined_probs = np.zeros_like(predictions_dict['xgb'])
    
    for model_name, probs in predictions_dict.items():
        combined_probs += weights[model_name] * probs
    
    return np.argmax(combined_probs, axis=1) + 1

def main():
    """Main execution pipeline"""
    print("="*70)
    print("Chromatin State Prediction Pipeline")
    print("="*70)
    
    # Initialize feature extractor
    extractor = AdvancedDNAFeatureExtractor()
    
    # Load data
    train_sequences, train_labels, test_sequences = load_data()
    
    # Encode sequences
    print("\n" + "="*70)
    print("Feature Extraction - Training Data")
    print("="*70)
    X_train = encode_sequences(train_sequences, extractor)
    
    print("\n" + "="*70)
    print("Feature Extraction - Test Data")
    print("="*70)
    X_test = encode_sequences(test_sequences, extractor)
    
    print(f"\nFeature matrix shape: {X_train.shape}")
    print(f"Total features: {X_train.shape[1]}")
    
    # Train multiple models (no scaling - tree models don't need it)
    print("\n" + "="*70)
    print("Training Multiple Models")
    print("="*70)
    
    xgb_model = train_xgboost_model(X_train, train_labels)
    lgb_model = train_lightgbm_model(X_train, train_labels)
    rf_model = train_random_forest_model(X_train, train_labels)
    et_model = train_extra_trees_model(X_train, train_labels)
    
    # Make predictions
    print("\n" + "="*70)
    print("Making Predictions")
    print("="*70)
    
    print("Getting predictions from all models...")
    predictions_dict = {
        'xgb': xgb_model.predict_proba(X_test),
        'lgb': lgb_model.predict_proba(X_test),
        'rf': rf_model.predict_proba(X_test),
        'et': et_model.predict_proba(X_test)
    }
    
    # Individual predictions (1-indexed)
    xgb_preds = xgb_model.predict(X_test) + 1
    lgb_preds = lgb_model.predict(X_test) + 1
    rf_preds = rf_model.predict(X_test)
    et_preds = et_model.predict(X_test)
    
    # Advanced ensemble
    print("\nCreating advanced ensemble (XGB:30%, LGB:30%, RF:25%, ET:15%)...")
    final_predictions = advanced_ensemble(predictions_dict)
    
    # Save all predictions
    print("\nSaving predictions...")
    pd.DataFrame(final_predictions).to_csv('predictions.csv', index=False, header=False)
    pd.DataFrame(xgb_preds).to_csv('predictions_xgb.csv', index=False, header=False)
    pd.DataFrame(lgb_preds).to_csv('predictions_lgb.csv', index=False, header=False)
    pd.DataFrame(rf_preds).to_csv('predictions_rf.csv', index=False, header=False)
    pd.DataFrame(et_preds).to_csv('predictions_et.csv', index=False, header=False)
    
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)
    print(f"Ensemble predictions saved to predictions.csv")
    print(f"\nTotal predictions: {len(final_predictions)}")
    print(f"\nEnsemble Prediction Distribution:")
    unique, counts = np.unique(final_predictions, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  State {label}: {count:5d} sequences ({100*count/len(final_predictions):5.2f}%)")

if __name__ == "__main__":
    main()
