"""
Sequence context and structural features for chromatin state prediction.
DNA shape, dinucleotide properties, and structural characteristics.
"""
import numpy as np

def extract_dinucleotide_features(sequence):
    """
    Extract dinucleotide step parameters and frequencies.
    Dinucleotides have unique structural properties important for chromatin.
    """
    dinucleotides = [
        'AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT',
        'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT'
    ]
    
    features = []
    length = len(sequence) - 1
    
    # Count frequencies
    for dinuc in dinucleotides:
        count = sum(1 for i in range(length) if sequence[i:i+2] == dinuc)
        features.append(count / length if length > 0 else 0)
    
    return np.array(features)


def extract_structural_features(sequence):
    """
    Extract DNA structural properties.
    Based on dinucleotide physical properties.
    """
    # Simplified DNA shape parameters (values from DNAshape database)
    # Roll, Twist, Tilt for common dinucleotides
    shape_params = {
        'AA': [0.06, 35.6, 0.0],  'AC': [0.02, 34.4, 0.0],
        'AG': [0.08, 27.7, 0.0],  'AT': [-0.03, 31.5, 0.0],
        'CA': [0.09, 34.5, 0.0],  'CC': [0.07, 33.7, 0.0],
        'CG': [-0.15, 29.8, 0.0], 'CT': [0.08, 27.7, 0.0],
        'GA': [0.07, 36.9, 0.0],  'GC': [-0.08, 40.0, 0.0],
        'GG': [0.07, 33.7, 0.0],  'GT': [0.02, 34.4, 0.0],
        'TA': [0.0, 36.0, 0.0],   'TC': [0.09, 34.5, 0.0],
        'TG': [0.09, 34.5, 0.0],  'TT': [0.06, 35.6, 0.0]
    }
    
    roll_values = []
    twist_values = []
    
    for i in range(len(sequence) - 1):
        dinuc = sequence[i:i+2]
        if dinuc in shape_params:
            roll, twist, _ = shape_params[dinuc]
            roll_values.append(roll)
            twist_values.append(twist)
    
    if not roll_values:
        return np.array([0, 0, 0, 0, 0, 0])
    
    # Aggregate statistics
    features = [
        np.mean(roll_values),
        np.std(roll_values),
        np.mean(twist_values),
        np.std(twist_values),
        np.max(roll_values) - np.min(roll_values),  # Roll range
        np.max(twist_values) - np.min(twist_values)  # Twist range
    ]
    
    return np.array(features)


def extract_periodicity_features(sequence):
    """
    Extract periodic patterns in DNA sequence.
    10bp periodicity is important for nucleosome positioning.
    """
    length = len(sequence)
    features = []
    
    # Check for nucleotide periodicity at 10bp intervals (nucleosome wrapping)
    for nucleotide in ['A', 'C', 'G', 'T']:
        period_10 = 0
        for i in range(0, length - 10):
            if sequence[i] == nucleotide and sequence[i + 10] == nucleotide:
                period_10 += 1
        features.append(period_10 / (length - 10) if length > 10 else 0)
    
    # 5bp periodicity (half nucleosome turn)
    for nucleotide in ['A', 'T']:  # A/T positioning is key
        period_5 = 0
        for i in range(0, length - 5):
            if sequence[i] == nucleotide and sequence[i + 5] == nucleotide:
                period_5 += 1
        features.append(period_5 / (length - 5) if length > 5 else 0)
    
    return np.array(features)


def extract_purine_pyrimidine_features(sequence):
    """
    Extract purine/pyrimidine patterns.
    RY (purine-pyrimidine) patterns affect DNA structure.
    """
    purines = 'AG'
    pyrimidines = 'CT'
    
    purine_count = sum(1 for n in sequence if n in purines)
    pyrimidine_count = sum(1 for n in sequence if n in pyrimidines)
    
    # RY alternations
    ry_alternations = 0
    for i in range(len(sequence) - 1):
        if (sequence[i] in purines and sequence[i+1] in pyrimidines) or \
           (sequence[i] in pyrimidines and sequence[i+1] in purines):
            ry_alternations += 1
    
    # RR and YY runs
    rr_count = sum(1 for i in range(len(sequence)-1) 
                   if sequence[i] in purines and sequence[i+1] in purines)
    yy_count = sum(1 for i in range(len(sequence)-1) 
                   if sequence[i] in pyrimidines and sequence[i+1] in pyrimidines)
    
    length = len(sequence)
    features = [
        purine_count / length,
        pyrimidine_count / length,
        ry_alternations / (length - 1) if length > 1 else 0,
        rr_count / (length - 1) if length > 1 else 0,
        yy_count / (length - 1) if length > 1 else 0
    ]
    
    return np.array(features)


def extract_context_features(sequence):
    """
    Master function to extract all sequence context features.
    """
    dinuc_features = extract_dinucleotide_features(sequence)
    structural_features = extract_structural_features(sequence)
    periodicity_features = extract_periodicity_features(sequence)
    purine_pyr_features = extract_purine_pyrimidine_features(sequence)
    
    return np.concatenate([
        dinuc_features,
        structural_features,
        periodicity_features,
        purine_pyr_features
    ])
