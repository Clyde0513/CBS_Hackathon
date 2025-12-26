"""
Feature cache utilities for saving and loading encoded feature matrices.
This avoids recomputing expensive feature extraction.
"""
import numpy as np
import os
from pathlib import Path

def save_features(X_train, y_train, X_test, cache_dir='feature_cache'):
    """
    Save encoded feature matrices to disk.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        X_test: Test feature matrix
        cache_dir: Directory to save cache files
    """
    Path(cache_dir).mkdir(exist_ok=True)
    
    print(f"Saving features to {cache_dir}/...")
    np.save(os.path.join(cache_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(cache_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(cache_dir, 'X_test.npy'), X_test)
    print(f"Features saved! ({X_train.shape[1]} features)")


def load_features(cache_dir='feature_cache'):
    """
    Load encoded feature matrices from disk.
    
    Args:
        cache_dir: Directory containing cache files
    
    Returns:
        Tuple of (X_train, y_train, X_test) or (None, None, None) if cache doesn't exist
    """
    try:
        X_train = np.load(os.path.join(cache_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(cache_dir, 'y_train.npy'))
        X_test = np.load(os.path.join(cache_dir, 'X_test.npy'))
        print(f"Loaded cached features from {cache_dir}/ ({X_train.shape[1]} features)")
        return X_train, y_train, X_test
    except FileNotFoundError:
        print(f"No cache found in {cache_dir}/")
        return None, None, None


def cache_exists(cache_dir='feature_cache'):
    """
    Check if feature cache exists.
    
    Args:
        cache_dir: Directory to check
    
    Returns:
        True if all cache files exist, False otherwise
    """
    required_files = ['X_train.npy', 'y_train.npy', 'X_test.npy']
    return all(os.path.exists(os.path.join(cache_dir, f)) for f in required_files)
