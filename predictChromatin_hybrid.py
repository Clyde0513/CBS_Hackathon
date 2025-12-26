"""
Enhanced Chromatin State Prediction using Hybrid XGBoost + Random Forest
Ensemble approach with advanced feature engineering
"""

import numpy as np
import pandas as pd
from collections import Counter
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

print("Libraries loaded successfully!")

class EnhancedDNAFeatureExtractor:
    """Comprehensive DNA sequence feature extraction"""
    
    def __init__(self):
        self.nucleotides = ['A', 'C', 'G', 'T']
    
    def basic_composition(self, sequence):
        """Basic nucleotide composition features"""
        length = len(sequence)
        a_count = sequence.count('A') / length
        c_count = sequence.count('C') / length
        g_count = sequence.count('G') / length
        t_count = sequence.count('T') / length
        
        gc_content = g_count + c_count
        at_content = a_count + t_count
        gc_skew = (g_count - c_count) / (g_count + c_count + 1e-10)
        at_skew = (a_count - t_count) / (a_count + t_count + 1e-10)
        
        return np.array([a_count, c_count, g_count, t_count, 
                        gc_content, at_content, gc_skew, at_skew])
    
    def kmer_frequencies(self, sequence, k):
        """K-mer frequency features"""
        kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
        kmer_counts = Counter(kmers)
        
        all_kmers = [''.join(p) for p in product(self.nucleotides, repeat=k)]
        features = np.array([kmer_counts.get(kmer, 0) for kmer in all_kmers])
        
        # Normalize by total k-mers
        return features / (len(sequence) - k + 1)
    
    def dinucleotide_frequencies(self, sequence):
        """Dinucleotide frequency features (16 features)"""
        return self.kmer_frequencies(sequence, k=2)
    
    def trinucleotide_frequencies(self, sequence):
        """Trinucleotide (3-mer) frequency features (64 features)"""
        return self.kmer_frequencies(sequence, k=3)
    
    def tetranucleotide_frequencies(self, sequence):
        """Tetranucleotide (4-mer) frequency features (256 features)"""
        return self.kmer_frequencies(sequence, k=4)
    
    def positional_features(self, sequence, n_windows=10):
        """
        Extract features from sequence windows
        Captures positional patterns
        """
        window_size = len(sequence) // n_windows
        features = []
        
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size if i < n_windows - 1 else len(sequence)
            window = sequence[start:end]
            
            # GC content per window
            gc = (window.count('G') + window.count('C')) / len(window)
            features.append(gc)
            
            # Purine content (A, G) per window
            purine = (window.count('A') + window.count('G')) / len(window)
            features.append(purine)
        
        return np.array(features)
    
    def sequence_complexity(self, sequence):
        """Sequence complexity and entropy features"""
        length = len(sequence)
        
        # Shannon entropy
        counts = np.array([sequence.count(n) for n in self.nucleotides])
        probs = counts / length
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Linguistic complexity for different k-mers
        complexities = []
        for k in [2, 3, 4]:
            kmers = set([sequence[i:i+k] for i in range(len(sequence) - k + 1)])
            complexity = len(kmers) / (len(sequence) - k + 1)
            complexities.append(complexity)
        
        # CpG islands indicator (CG dinucleotide enrichment)
        cg_count = sequence.count('CG')
        cg_ratio = cg_count / (length - 1) if length > 1 else 0
        
        return np.array([entropy] + complexities + [cg_ratio])
    
    def dinucleotide_transitions(self, sequence):
        """Count specific dinucleotide transitions"""
        transitions = {
            'CG': sequence.count('CG'),
            'GC': sequence.count('GC'),
            'AT': sequence.count('AT'),
            'TA': sequence.count('TA'),
            'AA': sequence.count('AA'),
            'TT': sequence.count('TT'),
            'GG': sequence.count('GG'),
            'CC': sequence.count('CC')
        }
        # Normalize
        features = np.array(list(transitions.values())) / (len(sequence) - 1)
        return features
    
    def structural_features(self, sequence):
        """DNA structural property features"""
        # Runs of same nucleotide (homopolymers)
        max_runs = {}
        for nuc in self.nucleotides:
            max_run = 0
            current_run = 0
            for n in sequence:
                if n == nuc:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            max_runs[nuc] = max_run
        
        # Normalize by sequence length
        features = np.array(list(max_runs.values())) / len(sequence)
        
        return features
    
    def extract_all_features(self, sequence):
        """
        Extract comprehensive feature set
        """
        features = []
        
        # Basic composition (8 features)
        features.append(self.basic_composition(sequence))
        
        # Dinucleotides (16 features)
        features.append(self.dinucleotide_frequencies(sequence))
        
        # Trinucleotides/3-mers (64 features)
        features.append(self.trinucleotide_frequencies(sequence))
        
        # Tetranucleotides/4-mers (256 features)
        features.append(self.tetranucleotide_frequencies(sequence))
        
        # Positional features (20 features: 10 windows * 2)
        features.append(self.positional_features(sequence, n_windows=10))
        
        # Complexity features (5 features)
        features.append(self.sequence_complexity(sequence))
        
        # Dinucleotide transitions (8 features)
        features.append(self.dinucleotide_transitions(sequence))
        
        # Structural features (4 features)
        features.append(self.structural_features(sequence))
        
        return np.concatenate(features)

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
    print(f"Encoding {len(sequences)} sequences with enhanced features...")
    encoded = []
    for i, seq in enumerate(sequences):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(sequences)} sequences")
        encoded.append(extractor.extract_all_features(seq))
    
    return np.array(encoded)

def train_xgboost_model(X_train, y_train):
    """Train XGBoost classifier"""
    print("\nTraining XGBoost model...")
    
    # Convert labels to 0-indexed for XGBoost
    y_train_xgb = y_train - 1
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=12,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=18,
        tree_method='hist',
        n_jobs=-1,
        random_state=42,
        verbosity=1
    )
    
    xgb_model.fit(X_train, y_train_xgb)
    
    train_acc = xgb_model.score(X_train, y_train_xgb)
    print(f"XGBoost Training accuracy: {train_acc:.4f}")
    
    return xgb_model

def train_random_forest_model(X_train, y_train):
    """Train Random Forest classifier"""
    print("\nTraining Random Forest model...")
    
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=35,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    rf_model.fit(X_train, y_train)
    
    train_acc = rf_model.score(X_train, y_train)
    print(f"Random Forest Training accuracy: {train_acc:.4f}")
    
    return rf_model

def ensemble_predictions(xgb_probs, rf_probs, xgb_weight=0.6):
    """
    Ensemble XGBoost and Random Forest predictions
    Weighted average of probability predictions
    """
    # Weighted average
    combined_probs = xgb_weight * xgb_probs + (1 - xgb_weight) * rf_probs
    
    # Return class with highest probability (XGBoost is 0-indexed, need to add 1)
    return np.argmax(combined_probs, axis=1) + 1  # +1 to convert to 1-18

def cross_validate_models(X_train, y_train, n_splits=3):
    """
    Quick cross-validation to find optimal ensemble weights
    """
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    xgb_scores = []
    rf_scores = []
    ensemble_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\nFold {fold}/{n_splits}")
        
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # Train models
        xgb_cv = xgb.XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1,
                                    subsample=0.8, n_jobs=-1, random_state=42, verbosity=0)
        xgb_cv.fit(X_tr, y_tr)
        
        rf_cv = RandomForestClassifier(n_estimators=200, max_depth=30, n_jobs=-1, 
                                       random_state=42, verbose=0)
        rf_cv.fit(X_tr, y_tr)
        
        # Predictions
        xgb_pred = xgb_cv.predict(X_val)
        rf_pred = rf_cv.predict(X_val)
        
        # Ensemble (simple voting)
        ensemble_pred = []
        for i in range(len(X_val)):
            # Simple majority vote
            votes = [xgb_pred[i], xgb_pred[i], rf_pred[i]]  # Give XGBoost 2x weight
            ensemble_pred.append(max(set(votes), key=votes.count))
        
        xgb_acc = accuracy_score(y_val, xgb_pred)
        rf_acc = accuracy_score(y_val, rf_pred)
        ensemble_acc = accuracy_score(y_val, ensemble_pred)
        
        xgb_scores.append(xgb_acc)
        rf_scores.append(rf_acc)
        ensemble_scores.append(ensemble_acc)
        
        print(f"  XGBoost: {xgb_acc:.4f}, RF: {rf_acc:.4f}, Ensemble: {ensemble_acc:.4f}")
    
    print(f"\nMean CV Scores:")
    print(f"  XGBoost: {np.mean(xgb_scores):.4f} (+/- {np.std(xgb_scores):.4f})")
    print(f"  Random Forest: {np.mean(rf_scores):.4f} (+/- {np.std(rf_scores):.4f})")
    print(f"  Ensemble: {np.mean(ensemble_scores):.4f} (+/- {np.std(ensemble_scores):.4f})")

def main():
    """Main execution pipeline"""
    print("="*70)
    print("Enhanced Chromatin State Prediction: XGBoost + Random Forest Ensemble")
    print("="*70)
    
    # Initialize feature extractor
    extractor = EnhancedDNAFeatureExtractor()
    
    # Load data
    train_sequences, train_labels, test_sequences = load_data()
    
    # Encode sequences with enhanced features
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
    print("Feature breakdown:")
    print("  - Basic composition: 8")
    print("  - Dinucleotides (2-mers): 16")
    print("  - Trinucleotides (3-mers): 64")
    print("  - Tetranucleotides (4-mers): 256")
    print("  - Positional features: 20")
    print("  - Complexity features: 5")
    print("  - Dinucleotide transitions: 8")
    print("  - Structural features: 4")
    
    # Cross-validation (optional, comment out to save time)
    # cross_validate_models(X_train, train_labels, n_splits=3)
    
    # Train XGBoost
    print("\n" + "="*70)
    print("Training XGBoost Model")
    print("="*70)
    xgb_model = train_xgboost_model(X_train, train_labels)
    
    # Train Random Forest
    print("\n" + "="*70)
    print("Training Random Forest Model")
    print("="*70)
    rf_model = train_random_forest_model(X_train, train_labels)
    
    # Make predictions
    print("\n" + "="*70)
    print("Making Predictions")
    print("="*70)
    
    print("XGBoost predictions...")
    xgb_probs = xgb_model.predict_proba(X_test)
    xgb_preds = xgb_model.predict(X_test) + 1  # Convert back to 1-indexed
    
    print("Random Forest predictions...")
    rf_probs = rf_model.predict_proba(X_test)
    rf_preds = rf_model.predict(X_test)
    
    # Ensemble predictions (60% XGBoost, 40% Random Forest)
    print("\nEnsembling predictions (60% XGBoost + 40% Random Forest)...")
    final_predictions = ensemble_predictions(xgb_probs, rf_probs, xgb_weight=0.6)
    
    # Save predictions
    print("\nSaving predictions to predictions.csv...")
    pd.DataFrame(final_predictions).to_csv('predictions.csv', index=False, header=False)
    
    # Also save individual model predictions for comparison
    pd.DataFrame(xgb_preds).to_csv('predictions_xgb.csv', index=False, header=False)
    pd.DataFrame(rf_preds).to_csv('predictions_rf.csv', index=False, header=False)
    
    print("\n" + "="*70)
    print("COMPLETED!")
    print("="*70)
    print(f"Ensemble predictions saved to predictions.csv")
    print(f"XGBoost predictions saved to predictions_xgb.csv")
    print(f"Random Forest predictions saved to predictions_rf.csv")
    print(f"\nTotal predictions: {len(final_predictions)}")
    print(f"\nEnsemble Prediction Distribution:")
    unique, counts = np.unique(final_predictions, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  State {label}: {count:5d} sequences ({100*count/len(final_predictions):5.2f}%)")

if __name__ == "__main__":
    main()
