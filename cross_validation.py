"""
Cross-validation module for evaluating model performance.
"""
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

def perform_cross_validation(model, X_train, y_train, cv=5, scoring='accuracy', verbose=True):
    """
    Perform cross-validation on the given model.

    Args:
        model: The model to evaluate.
        X_train: Training feature matrix.
        y_train: Training labels.
        cv: Number of cross-validation folds.
        scoring: Scoring metric for evaluation.
        verbose: Whether to print progress and results.

    Returns:
        Mean and standard deviation of cross-validation scores.
    """
    if verbose:
        print("\nPerforming cross-validation...")
        print(f"  Folds: {cv}")
        print(f"  Scoring: {scoring}")

    # Use only the meta-model for cross-validation
    if isinstance(model, dict) and 'meta_model' in model:
        model = model['meta_model']

    if verbose:
        print("Running cross-validation (this may take a few minutes)...")
    
    # Use the standard cross_val_score with progress indication
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
    
    import numpy as np
    
    if verbose:
        print("Cross-validation results:")
        print(f"  Mean accuracy: {np.mean(scores):.4f}")
        print(f"  Std deviation: {np.std(scores):.4f}")
        print(f"  Individual fold scores: {scores}")

    return np.mean(scores), np.std(scores)

    return scores.mean(), scores.std()