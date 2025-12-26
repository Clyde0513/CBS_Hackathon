"""
Ensemble training module for combining XGBoost and Random Forest.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def train_xgboost_rf_ensemble(X_train, y_train, verbose=True):
    """
    Train an ensemble model combining XGBoost and Random Forest.

    Args:
        X_train: Training feature matrix
        y_train: Training labels
        verbose: Whether to print training info

    Returns:
        Trained ensemble model (meta-classifier)
    """
    if verbose:
        print("\nTraining XGBoost + Random Forest ensemble...")

    # Split training data for meta-model
    X_base, X_meta, y_base, y_meta = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
    )

    # Adjust labels for XGBoost (0-indexed)
    y_base_adjusted = y_base - 1
    y_meta_adjusted = y_meta - 1

    # Train XGBoost model
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    xgb_model.fit(X_base, y_base_adjusted)

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_base, y_base)

    # Generate meta-features (predictions from base models)
    xgb_preds = xgb_model.predict_proba(X_meta)
    rf_preds = rf_model.predict_proba(X_meta)
    
    # Combine predictions as meta-features
    meta_features = np.hstack([xgb_preds, rf_preds])

    # Train meta-classifier
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    # Adjust labels back to original range (1 to 18) for meta-classifier
    meta_model.fit(meta_features, y_meta)

    if verbose:
        print("Ensemble training complete!")

    # Return a dictionary containing all models
    return {
        'xgb_model': xgb_model,
        'rf_model': rf_model,
        'meta_model': meta_model
    }


def predict_xgboost_rf_ensemble(ensemble, X_test):
    """
    Make predictions using the trained XGBoost + Random Forest ensemble.

    Args:
        ensemble: Dictionary containing trained base and meta models
        X_test: Test feature matrix

    Returns:
        Predicted labels
    """
    # Generate meta-features from base models
    xgb_preds = ensemble['xgb_model'].predict_proba(X_test)
    rf_preds = ensemble['rf_model'].predict_proba(X_test)
    meta_features = np.hstack([xgb_preds, rf_preds])

    # Predict using meta-classifier
    return ensemble['meta_model'].predict(meta_features)
