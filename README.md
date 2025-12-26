# Chromatin State Prediction from DNA Sequences

## Hackathon Project Overview

This project predicts chromatin states (18 classes) from 200bp DNA sequences using machine learning. The goal is to provide chromatin state annotations for human pangenome references and variant-specific chromatin state predictions.

## Problem Statement

- **Objective:** Predict chromatin states from DNA sequences
- **Input:** 200bp DNA sequences containing only {A, C, G, T} nucleotides
- **Output:** Classification into 18 different chromatin states
- **Application:** Enable chromatin state annotations for genetically diverse individuals and predict impact of genetic variants

## Dataset

### Training Data

- **trainsequences.csv:** 286,164 sequences of 200bp length
  - Each of the 18 states present in equal amounts (15,898 sequences per state)
  - Sequences contain only {A, C, G, T} with no gaps (N)

- **trainlabels.csv:** 286,164 state labels
  - One label per line corresponding to sequence in trainsequences.csv
  - Labels are integers between 1 and 18 inclusive

### Test Data

- **testsequences.csv:** 100,008 sequences of 200bp
  - Same format as trainsequences.csv
  - Sequences for each state present in equal amounts

## Solution Approach

### Evolution of Models

#### Version 1: Baseline Random Forest (Accuracy: ~17,536/100,008)
- 70 features: Basic composition + 3-mer frequencies
- Single Random Forest model
- Training accuracy: 99.93%

#### Version 2: Hybrid XGBoost + Random Forest (Accuracy: ~17,536/100,008)
- 381 features: Enhanced with 4-mers, positional, complexity features
- 2-model ensemble (XGBoost + Random Forest)

#### Version 3: Optimized 4-Model Ensemble (Accuracy: 18,386/100,008 = 18.38%)
**Current Best Model**

### Feature Engineering (1,424 Features)

1. **Basic Composition (10 features):**
   - Nucleotide frequencies (A, C, G, T)
   - GC/AT content and skew
   - Purine/Pyrimidine content

2. **K-mer Frequencies:**
   - 2-mers (16 features)
   - 3-mers (64 features)
   - 4-mers (256 features)
   - 5-mers (1,024 features) - Captures longer sequence patterns

3. **Positional Features (20 features):**
   - Windowed GC and purine content across 10 sequence windows
   - Captures position-dependent patterns

4. **Sequence Complexity (7 features):**
   - Shannon entropy
   - Linguistic complexity (2-5 mers)
   - CpG island indicators
   - Tandem repeat detection

5. **Dinucleotide Transitions (16 features):**
   - All dinucleotide transition frequencies

6. **Structural Features (8 features):**
   - Homopolymer run analysis (max and average runs)

7. **Palindrome Features (3 features):**
   - Detection of palindromic sequences (important for DNA binding sites)

### Model Architecture - 4-Model Ensemble

**Ensemble Weights:**
- XGBoost: 30%
- LightGBM: 30%
- Random Forest: 25%
- Extra Trees: 15%

**Individual Model Configurations:**

1. **XGBoost:**
   - 300 estimators, depth 12
   - Regularization: alpha=0.1, lambda=1.0
   - Learning rate: 0.08

2. **LightGBM:**
   - 300 estimators, depth 12
   - 50 leaves, learning rate: 0.08
   - Optimized for speed and memory

3. **Random Forest:**
   - 200 estimators, depth 25
   - Bootstrap sampling (80%)
   - sqrt feature selection

4. **Extra Trees:**
   - 200 estimators, depth 25
   - More randomized splits for diversity

### Training Process

1. Load training sequences and labels (286,164 samples)
2. Extract 1,424 features from each sequence
3. Train 4 ensemble models in parallel:
   - XGBoost (gradient boosting)
   - LightGBM (fast gradient boosting)
   - Random Forest (bagging)
   - Extra Trees (randomized bagging)
4. Generate weighted ensemble predictions
5. Save predictions in required format

## Results

### Model Performance

**Best Model (Optimized 4-Model Ensemble):**
- **Test Accuracy:** 18,386/100,008 = **18.38%**
- **Training Accuracy (XGBoost):** 100%
- **Training Accuracy (LightGBM):** ~47%
- **Total Features:** 1,424
- **Processing Time:** ~1.5 hours

**Model Evolution:**
- Baseline (RF only, 70 features): ~17.5%
- Hybrid (XGB+RF, 381 features): ~17.5%
- Optimized Ensemble (4 models, 1,424 features): **18.38%**

## Output

- **File:** predictions.csv
- **Format:** One prediction per line (1-18)
- **Order:** Nth line corresponds to Nth sequence in testsequences.csv
- **Total Lines:** 100,008

## Usage

### Requirements

```bash
pip install numpy pandas scikit-learn xgboost lightgbm
```

### Running the Models

**Baseline Model (Fast, ~5 min):**
```bash
python predictChromatin.py
```

**Hybrid Model (Medium, ~10 min):**
```bash
python predictChromatin_hybrid.py
```

**Optimized Ensemble (Best accuracy, ~1.5 hours):**
```bash
python predictChromatin_optimized.py
```

Each script will:
1. Load training and test data
2. Extract features and encode sequences
3. Train model(s)
4. Generate predictions
5. Save results to predictions.csv

## Files

### Main Scripts
- `predictChromatin.py` - Baseline Random Forest (70 features)
- `predictChromatin_hybrid.py` - XGBoost + Random Forest (381 features)
- `predictChromatin_optimized.py` - **Best: 4-model ensemble (1,424 features)**

### Data Files
- `trainsequences.csv` - Training DNA sequences (286,164 samples)
- `trainlabels.csv` - Training labels 1-18
- `testsequences.csv` - Test DNA sequences (100,008 samples)

### Output Files
- `predictions.csv` - Final ensemble predictions
- `predictions_xgb.csv` - XGBoost predictions
- `predictions_lgb.csv` - LightGBM predictions
- `predictions_rf.csv` - Random Forest predictions
- `predictions_et.csv` - Extra Trees predictions

## Future Improvements

### Strategies to Reach 20k+ Accuracy

1. **Deep Learning Approaches:**
   - Convolutional Neural Networks (CNN) for motif detection
   - Bidirectional LSTM for sequence context
   - Transformer models (DNA-BERT, Nucleotide Transformer)
   - Attention mechanisms for important regions

2. **Advanced Feature Engineering:**
   - DNA shape features (minor groove width, propeller twist)
   - Reverse complement features
   - TF binding motif enrichment
   - Epigenetic signal patterns
   - Longer k-mers (6-7 mers with sparse encoding)

3. **Ensemble Improvements:**
   - Stacking with meta-learner
   - Cross-validation based ensemble weights
   - CatBoost addition to ensemble
   - Neural network + tree model hybrid

4. **Domain Knowledge Integration:**
   - Known chromatin state patterns
   - Histone modification signals
   - DNA methylation patterns
   - ATAC-seq accessibility features

5. **Data Augmentation:**
   - Reverse complement augmentation
   - Sequence perturbations
   - Semi-supervised learning

## Key Insights

- **5-mer features** (1,024 features) provide significant improvement over 3-4 mers alone
- **Ensemble diversity** is crucial - combining gradient boosting + bagging methods
- **LightGBM** underperformed on this dataset, suggesting overfitting or parameter issues
- **Feature scaling** not needed for tree-based models and can hurt performance
- **Memory constraints** limit very deep trees with high-dimensional features

## Project Information

- **Start Date:** December 24, 2025
- **Current Status:** December 25, 2025
- **Event:** Computational Biology Hackathon
- **Task:** Chromatin State Prediction Challenge
- **Best Accuracy:** 18,386/100,008 (18.38%)
- **Target:** 20,000+ (20%)
