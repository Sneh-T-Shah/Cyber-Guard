# Cyber-Guard

This project focuses on building and evaluating a **hierarchical classification model** for text data, with a primary aim to categorize and sub-categorize **crime-related data** effectively.

---

### Competition Details  
This repository is part of the **CyberGuard AI Cybercrime Prevention Hackathon**, organized by [IndiaAI](https://indiaai.gov.in/article/indiaai-launches-cyberguard-ai-cybercrime-prevention-hackathon?utm_source=newsletter&utm_medium=email&utm_campaign=The%20Heuristic%20from%20INDIAai). The competition encourages participants to develop innovative AI solutions for cybercrime prevention.

### Dataset  
The dataset for this project can be accessed [here](https://drive.google.com/drive/folders/1ojEDfAky4wiCf5rkWpxNUv6Eee9vZKw0?usp=drive_link). It contains text-based crime data for training and evaluating the hierarchical classification model.

---

## Experiments Summary  

In this project, we explored a range of models, from **traditional machine learning techniques** to **large-scale transformers**, to tackle the challenging nature of hierarchical text classification.  

1. **Traditional ML Models**:
   - Used **TF-IDF** combined with classifiers such as **LightGBM** and other machine learning algorithms.
   - Challenges: The models struggled with capturing the nuances in the text, leading to poor generalization across train and test datasets.

2. **Transformer-based Models**:
   - **GPT-2** and **T5-Base** were employed to leverage their pre-trained contextual embeddings and fine-tuning capabilities.
   - Despite their robust architecture, they tended to **overfit** the training data due to the wide variation in text length and complexity.

### Key Insights  
- The text data spans a **diverse range of lengths and styles**, making it challenging for any model to generalize well across both the training and testing datasets.
- **Overfitting** was a significant issue across all approaches, as the models failed to find a consistent decision boundary to classify both train and test data effectively.  

---

## Methodology  

### Data Preparation  
1. **Null Handling**:
    - Replaced `NaN` in `sub_category` with the string `"NaN"`.
    - Dropped rows with other missing values.  

2. **Unique Labels**:
    - Identified unique `category` and `sub_category` labels in train and validation datasets.  
    - Filtered out unseen categories in the validation set that were absent in the training set.  

3. **Label Encoding**:
    - Encoded `category` and `sub_category` labels using `LabelEncoder` for compatibility with models.  

---

### Tokenization and Length Analysis  
1. **Tokenization**:
    - Used `DistilBertTokenizer` for input preparation.  

2. **Length Distribution**:
    - Analyzed token lengths to determine a suitable `MAX_LENGTH`.  
    - Statistical insights:
      - Mean token length: **225**.
      - 90th percentile: **250 tokens**.
      - 95th percentile: **280 tokens**.

3. **Visualization**:
    - Plotted token length distribution with key percentiles and mean values for reference.  

---

### Model Setup  

1. **Parameters**:  
   - `MAX_LENGTH = 225`  
   - `BATCH_SIZE = 32`  
   - `LEARNING_RATE = 5e-5`  
   - `NUM_EPOCHS = 5`  

2. **Models**:  
   - Two DistilBERT-based models:  
     - `CategoryModel`: Predicts high-level categories.  
     - `SubCategoryModel`: Predicts sub-categories based on categories.  

3. **Loss Functions**:  
   - **CORAL Loss**: Aligns covariances of train and validation distributions.  
   - **MMD Loss**: Minimizes mean discrepancies between distributions.  

---

### Model Training and Evaluation  

1. **DataLoader**:  
   - Custom `TextDataset` class for tokenization and label preparation.  

2. **Optimizer and Scheduler**:  
   - Used **AdamP** optimizer with adaptive learning rate control via `ReduceLROnPlateau`.  

3. **Validation Loop**:  
   - Calculated top-k accuracy for **k = 2, 3, 5** to evaluate ranking performance.  
   - Metrics: **Precision**, **Recall**, and **F1-Score**.  

---

## Results  

### Best Accuracy  

#### Category Prediction:  
- **Accuracy**: 74.13%  
- **Validation Loss**: 0.7548  
- **Top-3 Accuracy**: 94.83%  

#### Sub-category Prediction:  
- **Accuracy**: 53.90%  
- **Validation Loss**: 1.4238  
- **Top-3 Accuracy**: 94.83%  

---

### Contributors  
- **Sneh Shah**  
- **Niral Shekhat**  
