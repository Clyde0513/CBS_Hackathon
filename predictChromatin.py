import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Modular feature extraction imports
from motif_features import extract_motif_features
from encode_features import extract_encode_features
from advanced_kmer_features import extract_advanced_kmer_features
from context_features import extract_context_features

# Modular model training imports
from model_training import train_model, predict

# Feature caching imports
from feature_cache import save_features, load_features, cache_exists

# Configuration imports
from vars import USE_CACHE, CACHE_DIR, MODEL_TYPE

# Try to import custom model parameters if defined
try:
    from vars import RANDOM_FOREST_PARAMS
except ImportError:
    RANDOM_FOREST_PARAMS = None

try:
    from vars import LIGHTGBM_PARAMS
except ImportError:
    LIGHTGBM_PARAMS = None

"""
Chromatin State Prediction from DNA Sequences
"""

def load_data():
    """
    Load training and test data
    """
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

def encode_dataset(sequences, max_samples=None):
    """
    Encode all sequences in dataset with progress bar and time tracking
    """
    if max_samples:
        sequences = sequences[:max_samples]

    print(f"Encoding {len(sequences)} sequences...")
    start_time = time.time()
    encoded = []
    
    # Use tqdm for progress bar
    for seq in tqdm(sequences, desc="Encoding sequences", unit="seq"):
        encoded.append(extract_features(seq))
    
    elapsed_time = time.time() - start_time
    print(f"  Completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"  Average: {elapsed_time/len(sequences):.4f} seconds per sequence")

    return np.array(encoded)

"""
Chromatin State Prediction from DNA Sequences
"""

def encode_sequence_onehot(sequence): # if we need this function, feel free to use it (currently not using it)
    """
    One-hot encode DNA sequence
    A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1]
    Returns flattened array of length 800 (200bp * 4)
    """
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    encoded = []
    for nucleotide in sequence:
        encoded.extend(mapping[nucleotide])
    return np.array(encoded)

def encode_sequence_kmer(sequence, k=3):
    """
    K-mer frequency encoding
    Count frequencies of all possible k-mers
    """
    from collections import Counter
    from itertools import product
    
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    kmer_counts = Counter(kmers)
    
    # Create feature vector based on all possible k-mers (4^k possibilities)
    nucleotides = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in product(nucleotides, repeat=k)]
    
    features = []
    for kmer in all_kmers:
        features.append(kmer_counts.get(kmer, 0))
    
    return np.array(features)

def extract_features(sequence):
    """
    Extract comprehensive features from DNA sequence
    """
    # Basic composition features
    a_count = sequence.count('A')
    c_count = sequence.count('C')
    g_count = sequence.count('G')
    t_count = sequence.count('T')

    length = len(sequence)
    gc_content = (g_count + c_count) / length
    at_content = (a_count + t_count) / length

    # K-mer features (3-mers give 64 features)
    kmer_features = encode_sequence_kmer(sequence, k=3)

    # Advanced k-mer features (modular): k-mer spectrum + position-aware binning
    advanced_kmer_features = extract_advanced_kmer_features(sequence, k_range=(3, 4), num_bins=4)

    # Motif features (modular)
    motif_features = extract_motif_features(sequence)

    # ENCODE-inspired chromatin mark features (modular)
    encode_features = extract_encode_features(sequence)

    # Sequence context features (modular): dinucleotides, structure, periodicity
    context_features = extract_context_features(sequence)

    # Combine features
    basic_features = [gc_content, at_content, a_count, c_count, g_count, t_count]
    return np.concatenate([basic_features, kmer_features, motif_features, encode_features, advanced_kmer_features, context_features])

def main():
    """
    Main execution pipeline
    """
    print("="*60)
    print("Chromatin State Prediction Pipeline")
    print(f"Model: {MODEL_TYPE.upper()}")
    print("="*60)

    # Load data
    train_sequences, train_labels, test_sequences = load_data()

    # Try to load cached features
    X_train, y_train, X_test = None, None, None
    
    if USE_CACHE and cache_exists(CACHE_DIR):
        print("\n" + "="*60)
        print("Loading cached features...")
        print("="*60)
        X_train, y_train, X_test = load_features(CACHE_DIR)
        
        # Verify labels match
        if y_train is not None and not np.array_equal(y_train, train_labels):
            print("⚠️  Warning: Cached labels don't match current labels. Re-encoding...")
            X_train, y_train, X_test = None, None, None
    
    # Encode sequences if cache not available
    if X_train is None:
        print("\n" + "="*60)
        print("Encoding training data...")
        print("="*60)
        X_train = encode_dataset(train_sequences)

        print("\n" + "="*60)
        print("Encoding test data...")
        print("="*60)
        X_test = encode_dataset(test_sequences)
        
        # Save features for future runs
        if USE_CACHE:
            print("\n" + "="*60)
            print("Caching features for future runs...")
            print("="*60)
            save_features(X_train, train_labels, X_test, CACHE_DIR)

    print(f"\nTraining feature matrix shape: {X_train.shape}")
    print(f"Test feature matrix shape: {X_test.shape}")

    # Train model
    print("\n" + "="*60)
    print("Model Training")
    print("="*60)
    
    # Get model parameters if defined
    model_params = RANDOM_FOREST_PARAMS if MODEL_TYPE == 'random_forest' else LIGHTGBM_PARAMS
    
    model = train_model(X_train, train_labels, model_type=MODEL_TYPE, model_params=model_params)

    # Make predictions
    print("\n" + "="*60)
    print("Making predictions on test set...")
    print("="*60)
    predictions = predict(model, X_test, model_type=MODEL_TYPE)

    # Save predictions
    print("\nSaving predictions to predictions.csv...")
    pd.DataFrame(predictions).to_csv('predictions.csv', index=False, header=False)

    print("\n" + "="*60)
    print("COMPLETED!")
    print("="*60)
    print(f"Predictions saved to predictions.csv")
    print(f"Total predictions: {len(predictions)}")
    print(f"Predicted label distribution:")
    unique, counts = np.unique(predictions, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  State {label}: {count} sequences")

if __name__ == "__main__":
    main()
