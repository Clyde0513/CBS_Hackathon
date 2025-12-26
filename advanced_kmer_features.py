import numpy as np
from collections import Counter
from itertools import product

"""
Advanced K-mer Feature Engineering for DNA Sequences
Based on best practices from:
- Min et al., Bioinformatics 2017 (chromatin accessibility with k-mer embedding)
- Position-aware k-mer binning for 200bp sequences
- Reverse complement handling for biological accuracy
"""

def extract_advanced_kmer_features(sequence, k_range=(3, 4, 5, 6), num_bins=4, normalize=True, use_reverse_complement=True):
    """
    Extract advanced k-mer features from DNA sequence.
    
    Args:
        sequence: DNA sequence string
        k_range: tuple of k values to use (default: 3-6mers)
        num_bins: number of positional bins to divide sequence (default: 4 for 50bp bins in 200bp)
        normalize: whether to normalize by total possible k-mer positions
        use_reverse_complement: whether to canonicalize k-mers with reverse complement
    
    Returns:
        numpy array of concatenated features
    """
    all_features = []
    
    # 1. Global k-mer spectrum (baseline)
    for k in k_range:
        global_kmers = extract_kmer_spectrum(sequence, k, normalize, use_reverse_complement)
        all_features.extend(global_kmers)
    
    # 2. Position-aware k-mers (binned features)
    for k in k_range:
        binned_kmers = extract_position_aware_kmers(sequence, k, num_bins, normalize, use_reverse_complement)
        all_features.extend(binned_kmers)
    
    return np.array(all_features)


def extract_kmer_spectrum(sequence, k, normalize=True, use_reverse_complement=True):
    """
    Extract k-mer frequency spectrum from entire sequence.
    
    Args:
        sequence: DNA sequence
        k: k-mer length
        normalize: normalize by total k-mer positions
        use_reverse_complement: use canonical k-mers (min of kmer and its reverse complement)
    
    Returns:
        list of k-mer counts/frequencies
    """
    # Generate all k-mers from sequence
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    
    # Canonicalize with reverse complement if requested
    if use_reverse_complement:
        kmers = [canonicalize_kmer(kmer) for kmer in kmers]
    
    kmer_counts = Counter(kmers)
    
    # Get all possible k-mers (or canonical k-mers)
    all_kmers = get_all_kmers(k, use_reverse_complement)
    
    # Build feature vector
    features = []
    for kmer in all_kmers:
        count = kmer_counts.get(kmer, 0)
        if normalize:
            # Normalize by total k-mer positions in sequence
            count = count / (len(sequence) - k + 1) if len(sequence) >= k else 0
        features.append(count)
    
    return features


def extract_position_aware_kmers(sequence, k, num_bins, normalize=True, use_reverse_complement=True):
    """
    Extract k-mer counts per positional bin to capture 'motif near start/middle/end'.
    
    Args:
        sequence: DNA sequence
        k: k-mer length
        num_bins: number of bins to divide sequence (e.g., 4 bins = 50bp each for 200bp)
        normalize: normalize counts
        use_reverse_complement: use canonical k-mers
    
    Returns:
        list of concatenated binned k-mer features
    """
    seq_len = len(sequence)
    bin_size = seq_len // num_bins
    
    all_bin_features = []
    
    for bin_idx in range(num_bins):
        start = bin_idx * bin_size
        end = start + bin_size if bin_idx < num_bins - 1 else seq_len
        bin_sequence = sequence[start:end]
        
        # Extract k-mer spectrum for this bin
        bin_features = extract_kmer_spectrum(bin_sequence, k, normalize, use_reverse_complement)
        all_bin_features.extend(bin_features)
    
    return all_bin_features


def canonicalize_kmer(kmer):
    """
    Return the canonical k-mer: min(kmer, reverse_complement(kmer)).
    This reduces feature dimensionality by half and captures biological equivalence.
    
    Args:
        kmer: DNA k-mer string
    
    Returns:
        canonical k-mer (lexicographically smaller of kmer and its reverse complement)
    """
    rev_comp = reverse_complement(kmer)
    return min(kmer, rev_comp)


def reverse_complement(seq):
    """
    Return reverse complement of DNA sequence.
    
    Args:
        seq: DNA sequence string
    
    Returns:
        reverse complement string
    """
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement.get(base, base) for base in reversed(seq))


def get_all_kmers(k, canonical=False):
    """
    Generate all possible k-mers (or canonical k-mers if requested).
    
    Args:
        k: k-mer length
        canonical: if True, return only canonical k-mers (cuts size in half)
    
    Returns:
        list of k-mers
    """
    nucleotides = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in product(nucleotides, repeat=k)]
    
    if canonical:
        # Keep only canonical k-mers (min of kmer and its reverse complement)
        canonical_set = set()
        for kmer in all_kmers:
            canonical_set.add(canonicalize_kmer(kmer))
        return sorted(canonical_set)
    
    return all_kmers


# Example usage and feature count calculation
def get_feature_count(k_range=(3, 4, 5, 6), num_bins=4, use_reverse_complement=True):
    """
    Calculate total number of features for given parameters.
    Useful for understanding feature vector dimensionality.
    
    Args:
        k_range: tuple of k values
        num_bins: number of positional bins
        use_reverse_complement: whether using canonical k-mers
    
    Returns:
        total feature count
    """
    total = 0
    for k in k_range:
        # Global k-mer spectrum
        kmer_count = len(get_all_kmers(k, use_reverse_complement))
        total += kmer_count
        
        # Position-aware k-mers (binned)
        total += kmer_count * num_bins
    
    return total


if __name__ == "__main__":
    # Example: calculate feature dimensions
    print("Advanced K-mer Feature Dimensions:")
    print(f"K-range: 3-6, 4 bins, with reverse complement")
    print(f"Total features: {get_feature_count()}")
    print(f"\nBreakdown:")
    for k in [3, 4, 5, 6]:
        canonical_count = len(get_all_kmers(k, canonical=True))
        print(f"  {k}-mers: {canonical_count} canonical kmers × (1 global + 4 bins) = {canonical_count * 5} features")
