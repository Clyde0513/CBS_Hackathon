import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

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

    # Motif features
    motif_features = extract_motif_features(sequence)

    # ENCODE-inspired chromatin mark features
    encode_features = extract_encode_features(sequence)

    # Combine features
    basic_features = [gc_content, at_content, a_count, c_count, g_count, t_count]

    return np.concatenate([basic_features, kmer_features, motif_features, encode_features])
# --- Motif Discovery Feature Engineering ---
def extract_motif_features(sequence):
    """
    Encode presence/absence and count of known regulatory motifs in the sequence.
    Returns a fixed-length feature vector.
    """
    motifs = get_known_motifs()
    features = []
    for motif in motifs:
        # Presence (binary)
        present = 1 if motif in sequence else 0
        # Count (frequency)
        count = sequence.count(motif)
        features.extend([present, count])
    return np.array(features)

def extract_encode_features(sequence):
    """
    For each (mark, cell type) pair, encode presence and count of published motif in the sequence.
    Motifs are based on published sources (see comments).
    Returns a fixed-length feature vector (9 marks × 9 cell types × 2 = 162 features).
    """
    marks = get_encode_marks()
    cell_types = get_encode_cell_types()
    mark_motifs = get_encode_mark_motifs()
    features = []
    for mark in marks:
        motif = mark_motifs[mark]
        for cell in cell_types:
            present = 1 if motif in sequence else 0
            count = sequence.count(motif)
            features.extend([present, count])
    return np.array(features)

def get_encode_marks():
    """
    Returns the 9 ENCODE histone marks + CTCF.
    """
    return [
        'H3K4me1',
        'H3K4me2',
        'H3K4me3',
        'H3K27ac',
        'H3K9ac',
        'H3K27me3',
        'H4K20me1',
        'H3K36me3',
        'CTCF',
    ]

def get_encode_cell_types():
    """
    Returns the 9 ENCODE cell types.
    """
    return [
        'HUVEC',
        'NHEK',
        'GM12878',
        'K562',
        'HepG2',
        'NHLF',
        'HMEC',
        'HSMM',
        'H1',
    ]

def get_encode_mark_motifs():
    """
    Returns a dictionary mapping each mark to a published/proxy motif.
    Motif sources are cited in comments.
    """
    return {
        # Motifs are consensus/proxy sequences from literature or JASPAR/ENCODE/Ernst et al.
        # H3K4me1: Associated with enhancers, often GATA motifs (GATA). Source: Heintzman et al., Nature Genet 2007.
        'H3K4me1': 'GATA',
        # H3K4me2: Associated with CCAAT (NF-Y) motif. Source: ENCODE, Ernst et al., Nature 2011.
        'H3K4me2': 'CCAAT',
        # H3K4me3: Associated with CpG islands (CGCG). Source: Bernstein et al., Cell 2005; ENCODE.
        'H3K4me3': 'CGCG',
        # H3K27ac: Associated with AP-1 motif (TGACTCA). Source: Creyghton et al., PNAS 2010; JASPAR MA0099.2.
        'H3K27ac': 'TGACTCA',
        # H3K9ac: Associated with SP1/GC box (GGGCGG). Source: ENCODE, JASPAR MA0079.3.
        'H3K9ac': 'GGGCGG',
        # H3K27me3: Polycomb repression, associated with GAGA motif (GAGAG). Source: Schuettengruber et al., Cell 2009.
        'H3K27me3': 'GAGAG',
        # H4K20me1: Associated with CTCF motif (CCCTC). Source: ENCODE, Wang et al., Nature 2008.
        'H4K20me1': 'CCCTC',
        # H3K36me3: Associated with SRSF2 motif (AGGAGG). Source: Kolasinska-Zwierz et al., Nature Genet 2009.
        'H3K36me3': 'AGGAGG',
        # CTCF: Canonical CTCF motif (CCCTC). Source: JASPAR MA0139.1; ENCODE.
        'CTCF': 'CCCTC',
    }
def get_known_motifs():
    """
    Returns a list of known regulatory motifs to search for.
    Extend this list as needed for biological relevance.
    """
    return [
        'TATAAA',   # TATA box (core promoter)
        'CGCG',     # CpG island (short proxy)
        'CCAAT',    # CAAT box (promoter)
        'GGGCGG',   # GC box (promoter)
        'AATAAA',   # Polyadenylation signal
        'ATG',      # Start codon (general)
        'TTGACA',   # -35 element (prokaryotic promoter, for completeness)
        'TATAAT',   # -10 element (prokaryotic, Pribnow box)
        'GATA',     # GATA motif (transcription factor binding)
        'CAGGTG',   # E-box (enhancer, bHLH TFs)
        'GGGCGG',   # SP1 binding site (redundant with GC box, but common)
        'TGACGTCA', # CRE (cAMP response element)
        'CACGTG',   # E-box (canonical)
        'AGGAGG',   # Shine-Dalgarno (prokaryotic RBS)
        'TTAA',     # AT-rich element
        'GCGCGC',   # Extended CpG island
        'CCCTC',    # CTCF binding site (short proxy)
        'GATAAG',   # GATA1 binding site
        'TCTAGA',   # XbaI restriction site (proxy for palindromic motif)
        'GAATTC',   # EcoRI site (palindromic, proxy for TFBS)
        # Add more motifs as needed for your biological context
    ]

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
    Encode all sequences in dataset
    """
    if max_samples:
        sequences = sequences[:max_samples]
    
    print(f"Encoding {len(sequences)} sequences...")
    encoded = []
    for i, seq in enumerate(sequences):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(sequences)} sequences")
        encoded.append(extract_features(seq))
    
    return np.array(encoded)

def train_model(X_train, y_train):
    """
    Train Random Forest classifier
    """
    print("\nTraining Random Forest model...")
    
    # Use a strong ensemble model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    print(f"Training accuracy: {model.score(X_train, y_train):.4f}")
    
    return model

def main():
    """
    Main execution pipeline
    """
    print("="*60)
    print("Chromatin State Prediction Pipeline")
    print("="*60)
    
    # Load data
    train_sequences, train_labels, test_sequences = load_data()
    
    # Encode sequences
    print("\n" + "="*60)
    print("Encoding training data...")
    print("="*60)
    X_train = encode_dataset(train_sequences)
    
    print("\n" + "="*60)
    print("Encoding test data...")
    print("="*60)
    X_test = encode_dataset(test_sequences)
    
    print(f"\nTraining feature matrix shape: {X_train.shape}")
    print(f"Test feature matrix shape: {X_test.shape}")
    
    # Train model
    print("\n" + "="*60)
    print("Model Training")
    print("="*60)
    model = train_model(X_train, train_labels)
    
    # Make predictions
    print("\n" + "="*60)
    print("Making predictions on test set...")
    print("="*60)
    predictions = model.predict(X_test)
    
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
