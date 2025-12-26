"""
Reverse complement features for DNA sequences.
Since DNA is double-stranded, we need to consider both strands for chromatin binding.
"""
import numpy as np

def reverse_complement(sequence):
    """
    Generate the reverse complement of a DNA sequence.
    
    Args:
        sequence: DNA sequence string
    
    Returns:
        Reverse complement sequence
    """
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(complement.get(base, base) for base in reversed(sequence))


def extract_reverse_complement_features(sequence, feature_extractor_func):
    """
    Extract features from the reverse complement of a sequence.
    
    This is crucial because:
    1. DNA is double-stranded
    2. Transcription factors can bind to either strand
    3. Chromatin marks may be strand-specific
    
    Args:
        sequence: Original DNA sequence
        feature_extractor_func: Function to extract features from a sequence
    
    Returns:
        Features from reverse complement
    """
    rev_comp_seq = reverse_complement(sequence)
    return feature_extractor_func(rev_comp_seq)


def extract_strand_symmetric_features(sequence):
    """
    Extract strand-symmetric features that account for both DNA strands.
    
    These features consider that DNA binding proteins can access both strands:
    1. Palindromic motif density (same on both strands)
    2. Strand asymmetry measures
    3. GC skew (G-C)/(G+C) on forward strand
    4. AT skew (A-T)/(A+T) on forward strand
    
    Returns:
        Array of strand-aware features
    """
    rev_comp_seq = reverse_complement(sequence)
    
    # Palindromic regions (important for TF binding)
    palindrome_count = 0
    for i in range(len(sequence) - 5):
        substring = sequence[i:i+6]
        if substring == reverse_complement(substring):
            palindrome_count += 1
    palindrome_density = palindrome_count / (len(sequence) - 5) if len(sequence) > 5 else 0
    
    # Strand asymmetry: how different are the two strands?
    forward_a = sequence.count('A')
    forward_t = sequence.count('T')
    forward_g = sequence.count('G')
    forward_c = sequence.count('C')
    
    # GC skew: measures leading vs lagging strand bias
    gc_skew = (forward_g - forward_c) / (forward_g + forward_c) if (forward_g + forward_c) > 0 else 0
    
    # AT skew: purine/pyrimidine bias
    at_skew = (forward_a - forward_t) / (forward_a + forward_t) if (forward_a + forward_t) > 0 else 0
    
    # Purine skew: (A+G - T-C) / (A+G+T+C)
    purine_count = forward_a + forward_g
    pyrimidine_count = forward_t + forward_c
    purine_skew = (purine_count - pyrimidine_count) / len(sequence) if len(sequence) > 0 else 0
    
    # Check for inverted repeats (important for chromatin looping)
    inverted_repeat_score = 0
    window_size = 10
    for i in range(len(sequence) - 2 * window_size):
        window1 = sequence[i:i+window_size]
        # Look for reverse complement downstream
        for j in range(i + window_size, len(sequence) - window_size):
            window2 = sequence[j:j+window_size]
            if window1 == reverse_complement(window2):
                inverted_repeat_score += 1
    inverted_repeat_density = inverted_repeat_score / (len(sequence) - 2 * window_size) if len(sequence) > 2 * window_size else 0
    
    return np.array([
        palindrome_density,
        gc_skew,
        at_skew,
        purine_skew,
        inverted_repeat_density
    ])
