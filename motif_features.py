import numpy as np

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
