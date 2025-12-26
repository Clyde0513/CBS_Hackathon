"""
Configuration variables for chromatin state prediction pipeline
"""

# Feature caching
USE_CACHE = True  # Set to False to force re-encoding features
CACHE_DIR = 'feature_cache'  # Directory to store cached feature matrices

# Model selection
MODEL_TYPE = 'xgboost_rf_ensemble'  # Options: 'random_forest', 'lightgbm', 'xgboost_rf_ensemble'

# Model hyperparameters (optional - defaults are in model_training.py)
# Uncomment and modify if you want to override defaults:
RANDOM_FOREST_PARAMS = {
    'n_estimators': 500,  # More trees = better (diminishing returns after ~500)
    'max_depth': None,  # Let trees grow deep to capture complex patterns
    'min_samples_split': 5,  # Prevent overfitting
    'min_samples_leaf': 2,  # Minimum samples per leaf
    'max_features': 'sqrt',  # Use sqrt of features at each split (good for genomics)
    'bootstrap': True,
    'oob_score': True,  # Get out-of-bag score for validation
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced'  # Handle class imbalance
}
# LIGHTGBM_PARAMS = {
#     'objective': 'multiclass',
#     'num_class': 18,
#     'num_boost_round': 200,
#     'num_leaves': 31,
#     'learning_rate': 0.1,
#     'verbose': -1
# }

# Cross-validation configuration
ENABLE_CROSS_VALIDATION = True  # Set to True to enable cross-validation
CROSS_VALIDATION_PARAMS = {
    'cv': 5,  # Number of folds
    'scoring': 'accuracy',  # Metric for evaluation
    'verbose': True  # Print progress and results
}
