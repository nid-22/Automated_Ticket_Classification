# Automated Ticket Classification

> **Classify customer support tickets into product/service categories using Topic Modelling and Supervised Learning**

---

## 📌 Table of Contents

- [Problem Statement](#problem-statement)
- [Pipeline Overview](#pipeline-overview)
- [Data Preparation](#data-preparation)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Topic Modelling](#topic-modelling)
- [Supervised Modeling & Evaluation](#supervised-modeling--evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

---

##  Problem Statement

Build a machine learning pipeline to **classify customer complaints (support tickets)** into predefined product/service categories, enabling faster routing to the correct department.

###  Target Categories (5)

- **Credit card / Prepaid card**
- **Bank account services**
- **Theft / Dispute reporting**
- **Mortgages / Loans**
- **Others**

---

##  Pipeline Overview

The project is implemented as **8 major pipeline stages**:

1. Data loading  
2. Text preprocessing  
3. Exploratory Data Analysis (EDA)  
4. Feature extraction  
5. Topic modelling (NMF)  
6. Supervised model building  
7. Model training & evaluation  
8. Model inference  

---

## 🧹 Data Preparation

Key preprocessing steps applied before topic modelling:

- Removed blank / empty complaints  
- Converted text to lowercase  
- Removed:
  - Text inside square brackets
  - Punctuation
  - Words containing numbers  
- Lemmatized tokens  
- Extracted **POS tags** and retained **only nouns (POS = NN)** to focus topics on meaningful entities  

---

##  Exploratory Data Analysis (EDA)

Performed exploratory analysis to understand complaint characteristics:

- Distribution of complaint lengths (character count)  
- Word cloud of top 40 words after preprocessing  
- Frequency analysis of:
  - Unigrams
  - Bigrams
  - Trigrams  

### 📈 N-gram Frequency Visualization

![Unigram and Bigram Frequencies](assets/images/automated_ticket_classification/unigram_bigram.png)

---

##  Topic Modelling

- Applied **Non-Negative Matrix Factorization (NMF)** on **TF-IDF features**
- Extracted **5 topics**, aligned with target departments
- Each complaint is assigned its **dominant topic**
- Topic labels are later used as targets for supervised learning

###  Topic Visualization

![Topic Modelling - NMF](assets/images/automated_ticket_classification/topic_modelling.png)

---

##  Supervised Modeling & Evaluation

After deriving topic labels, multiple supervised models were trained.

### Modeling Steps

- Vectorization using **CountVectorizer**
- Transformation to **TF-IDF**
- Train-test split
- Models trained:
  - Logistic Regression
  - Decision Tree
  - Random Forest  
  - (Naive Bayes – optional)
- Evaluation metrics:
  - Accuracy
  - ROC-AUC
  - Classification Report

---

##  Results

### Best Model: **Logistic Regression (GridSearchCV)**

- **Best CV Score:** `0.9589`
- **Best Hyperparameters:**
  ```python
  {'C': 1, 'penalty': 'l1', 'solver': 'saga'}
