import numpy as np
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

"""
Modular model training for chromatin state prediction.
Supports multiple model types: Random Forest and LightGBM.
"""

def train_random_forest(X_train, y_train, verbose=True, **kwargs):
    """
    Train Random Forest classifier.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        verbose: Whether to print training info
        **kwargs: Additional parameters for RandomForestClassifier
    
    Returns:
        Trained Random Forest model
    """
    if verbose:
        print("\nTraining Random Forest model...")
    
    # Default parameters (can be overridden by kwargs from vars.py)
    default_params = {
        'n_estimators': 200,
        'max_depth': 30,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': 1 if verbose else 0
    }
    
    # Merge with custom parameters
    params = {**default_params, **kwargs}
    
    model = RandomForestClassifier(**params)
    
    model.fit(X_train, y_train)
    
    if verbose:
        print(f"Training accuracy: {model.score(X_train, y_train):.4f}")
        if hasattr(model, 'oob_score_') and model.oob_score:
            print(f"Out-of-bag score: {model.oob_score_:.4f}")
    
    return model


def train_lightgbm(X_train, y_train, verbose=True):
    """
    Train LightGBM classifier.
    More memory-efficient and faster than Random Forest for large datasets.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        verbose: Whether to print training info
    
    Returns:
        Trained LightGBM model
    """
    if verbose:
        print("\nTraining LightGBM model...")
    
    # LightGBM expects labels to be 0-indexed, but our labels are 1-18
    # Convert to 0-17 for training
    y_train_adjusted = y_train - 1
    
    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train_adjusted)
    
    # Set parameters
    params = {
        'objective': 'multiclass',
        'num_class': 18,  # 18 chromatin states
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0 if not verbose else 1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data],
        valid_names=['train'],
        callbacks=[lgb.log_evaluation(period=20 if verbose else 0)]
    )
    
    if verbose:
        # Calculate training accuracy
        y_pred = model.predict(X_train)
        y_pred_labels = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred_labels == y_train)
        print(f"Training accuracy: {accuracy:.4f}")
    
    return model


def train_model(X_train, y_train, model_type='lightgbm', verbose=True, model_params=None):
    """
    Train a model based on specified type.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        model_type: 'random_forest' or 'lightgbm'
        verbose: Whether to print training info
        model_params: Dictionary of model-specific parameters
    
    Returns:
        Trained model
    """
    if model_params is None:
        model_params = {}
    
    if model_type == 'random_forest':
        return train_random_forest(X_train, y_train, verbose, **model_params)
    elif model_type == 'lightgbm':
        return train_lightgbm(X_train, y_train, verbose)
    elif model_type == 'xgboost_rf_ensemble':
        from ensemble_training import train_xgboost_rf_ensemble
        return train_xgboost_rf_ensemble(X_train, y_train, verbose)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'random_forest' or 'lightgbm'")


def predict(model, X_test, model_type='lightgbm'):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model
        X_test: Test feature matrix
        model_type: 'random_forest' or 'lightgbm'
    
    Returns:
        Predicted labels (in original 1-18 range)
    """
    if model_type == 'random_forest':
        return model.predict(X_test)
    elif model_type == 'lightgbm':
        y_pred = model.predict(X_test)
        # LightGBM returns 0-17, convert back to 1-18
        return np.argmax(y_pred, axis=1) + 1
    elif model_type == 'xgboost_rf_ensemble':
        from ensemble_training import predict_xgboost_rf_ensemble
        return predict_xgboost_rf_ensemble(model, X_test)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_xgboost_rf_ensemble(X_train, y_train, verbose=True):
    from ensemble_training import train_xgboost_rf_ensemble
    return train_xgboost_rf_ensemble(X_train, y_train, verbose)

def predict_xgboost_rf_ensemble(ensemble, X_test):
    from ensemble_training import predict_xgboost_rf_ensemble
    return predict_xgboost_rf_ensemble(ensemble, X_test)
