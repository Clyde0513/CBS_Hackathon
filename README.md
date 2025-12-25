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

### Feature Engineering

The model extracts 70 features from each DNA sequence:

1. **Basic Composition Features (6 features):**
   - GC content
   - AT content
   - Individual nucleotide counts (A, C, G, T)

2. **K-mer Features (64 features):**
   - 3-mer frequency encoding
   - Counts of all possible 3-nucleotide combinations (4³ = 64 possible 3-mers)

### Model Architecture

- As of now:
- **ML Model:** Random Forest Classifier
- **Configuration:**
  - 200 decision trees
  - Max depth: 30
  - Min samples split: 5
  - Min samples leaf: 2
  - Parallel processing enabled (8 cores)

### Training Process

1. Load training sequences and labels
2. Extract features from each sequence
3. Train Random Forest model on encoded features
4. Generate predictions for test sequences
5. Save predictions in required format

## Results

### Model Performance

- **Training Accuracy:** 99.93%
- **Total Predictions:** 100,008
- **Processing Time:** ~5 minutes

### Prediction Distribution

| State | Count | State | Count | State | Count |
|-------|-------|-------|-------|-------|-------|

| 1 | 4,238 | 7 | 3,420 | 13 | 8,667 |
| 2 | 4,914 | 8 | 6,853 | 14 | 9,512 |
| 3 | 2,877 | 9 | 8,404 | 15 | 9,267 |
| 4 | 3,172 | 10 | 5,515 | 16 | 4,067 |
| 5 | 4,149 | 11 | 4,720 | 17 | 2,406 |
| 6 | 3,534 | 12 | 4,818 | 18 | 9,475 |

## Output

- **File:** predictions.csv
- **Format:** One prediction per line (1-18)
- **Order:** Nth line corresponds to Nth sequence in testsequences.csv
- **Total Lines:** 100,008

## Usage

### Requirements

```bash
pip install numpy pandas scikit-learn
```

### Running the Pipeline

```bash
python predictChromatin.py
```

The script will:

1. Load training and test data
2. Extract features and encode sequences
3. Train the Random Forest model
4. Generate predictions
5. Save results to predictions.csv

## Files

- `predictChromatin.py` - Main prediction pipeline script
- `trainsequences.csv` - Training DNA sequences
- `trainlabels.csv` - Training labels (1-18)
- `testsequences.csv` - Test DNA sequences
- `predictions.csv` - Generated predictions (output)

## Future Improvements

### Potential Enhancements

1. **Advanced Models:**
   - XGBoost or LightGBM for gradient boosting
   - Deep learning models (CNN or LSTM for sequence modeling)
   - Ensemble of multiple model types

2. **Enhanced Features:**
   - Longer k-mers (4-mers or 5-mers) for more sequence specificity
   - Dinucleotide frequencies
   - Positional features (e.g., GC content in different regions)
   - DNA shape features

3. **Validation:**
   - Cross-validation to estimate generalization performance
   - Confusion matrix analysis
   - Per-class performance metrics

4. **Optimization:**
   - Hyperparameter tuning
   - Feature selection
   - Model compression for faster inference

## Project Information

- **Starting Work Date:** December 24, 2025
- **Event:** Computational Biology Hackathon
- **Task:** Chromatin State Prediction Challenge
