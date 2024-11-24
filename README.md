# Cyber-guard

This project involves building and evaluating a hierarchical classification model for text data, with a focus on categorizing and sub-categorizing crime-related data.

## Data Preparation

1. **Null Handling**:
    - Fill `NaN` values in the `sub_category` column with "NaN".
    - Drop remaining rows with any `NaN` values.

2. **Unique Labels**:
    - Determine unique categories and sub-categories in both train and validation datasets.
    - Identify unseen categories in the validation set.

3. **Remove Unseen Categories**:
    - Exclude rows in the validation set with unseen categories that do not appear in the training set.

4. **Label Encoding**:
    - Encode `category` and `sub_category` columns with `LabelEncoder` for model compatibility.


## Tokenization and Length Analysis

1. **Tokenization**:
    - Tokenize text data using the `DistilBertTokenizer` to prepare input for DistilBERT.

2. **Token Length Distribution**:
    - Calculate token lengths for each text, analyze the mean and specific percentiles.
    - Plot the token length distribution and mark statistical lines (mean, 90th, and 95th percentiles).

3. **Statistical Summary**:
    - Provide a detailed summary of token lengths with mean and percentile information.

## Model Setup

1. **Parameters**:
    - `MAX_LENGTH = 225`
    - `BATCH_SIZE = 32`
    - `LEARNING_RATE = 5e-5`
    - `NUM_EPOCHS = 5`

2. **Category and Sub-Category Models**:
    - Two models: `CategoryModel` and `SubCategoryModel`, both based on DistilBERT with custom classification heads.

3. **Loss Functions**:
    - **CORAL Loss** for covariance alignment.
    - **MMD Loss** for distribution alignment.

## DataLoader and Dataset

1. **TextDataset Class**:
    - Dataset class to handle tokenization, padding, and label preparation.
2. **DataLoaders**:
    - `train_loader` and `val_loader` to handle batch processing for training and validation datasets.

## Optimizer and Scheduler

1. **Optimizer**:
    - `AdamP` optimizer with specific parameters for category and sub-category models.

2. **Scheduler**:
    - `ReduceLROnPlateau` scheduler for adaptive learning rate based on validation loss.

## Metrics Calculation

1. **Accuracy Calculation**:
    - Compute accuracy for each batch.

2. **Top-K Accuracy**:
    - Implement top-k accuracy for k=2, 3, and 5 to evaluate model ranking performance.

3. **Compute Metrics**:
    - Calculate precision, recall, and F1 scores.

## Validation Loop

1. **Validate Function**:
    - Evaluation loop to assess the model performance on validation data.
      
2. **Top-K Accuracy Tracking**:
    - Track top-k accuracy for category and sub-category predictions.
  

  
# Model Evaluation Summary (Best accuracy)

## Category Metrics

- **Accuracy**: 74.13%
- **Validation Loss**: 0.7548
- **Top-3 Accuracy**: 94.83%


## Sub-category Metrics

- **Accuracy**: 53.90%
- **Validation Loss**: 1.4238
- **Top-3 Accuracy**: 94.83%


## Contributors
- Sneh Shah
- Niral Shekhat

