# EPILEPTIC SEIZURE DETECTION USING MACHINE LEARNING

## 📚 Project Overview

This is my **Machine Learning** subject project where I compare different machine learning algorithms. I built 7 different models and tested which one works best on my dataset.

## 🎯 Goal & Objectives

### Main Goal
To find the best machine learning model for my dataset by comparing their performance.

### What I Did:
1. **Built 7 different models** using different algorithms
2. **Compared their performance** using accuracy and other metrics
3. **Tuned each model** to get the best settings
4. **Tried combining models** using ensemble methods
5. **Created charts and tables** to compare results
6. **Saved all results** for future reference

## 📊 Dataset Information

I have used the Bangalore EEG Epilepsy Dataset (BEED) provided by the UCI Machine Learning Repository. It is a collection for epileptic seizure detection and classification, recorded at a neurological research centre in Bangalore, India. Tt features high-fidelity EEG signals captured using the standard 10-20 electrode system at a 256 Hz sampling rate.
This dataset is completely clean with no missing values and no class imbalance.

### About My Data:
- **Type**: Classification (predicting categories)
- **Size**: 8000 samples with 17 features
- **What I'm predicting**: the type of seizure
- **Number of classes**: 4 (Healthy Subject(0), Generalized Seizure(1), Focal Seizure(2), Seizure Events(3))

### What I Did with the Data:
1. Analyzed the data statistics and plotted it
2. Split data into training (80%) and testing (20%)
3. Scaled the features using Standard Scaler

## 📁 Project Structure

Here's how my project files are organized:

```
EPILECTIC-SEIZURE-DETECTION-USING-ML/
│
├── data/     
│   ├── dataset/                        # Dataset storage
│   │   ├── BEED_Data.csv               # Main dataset file
│   │   └── beed_bangalore_eeg_epilepsy_dataset.zip  # Original dataset archive
│   └── training_data/                  # Processed data               
│       ├── features_labels.npz         # Extracted features and labels
│       └── splits.npz                  # Data splits for training/testing
│
├── model_training/               # Model training and evaluation
│   ├── ann_tuning/               # Artificial Neural Network tuning information
│   ├── cathpost_info/            # CatBoost information
│   ├── performance_metrics/      # Model performance results
│   │   ├── classification_reports/
│   │   ├── ensemble_classification_reports/
│   │   ├── model_comparison_summary.csv
│   │   └── model_metrics_comparison.ipynb
│   └── trained_models/           # Individual model notebooks
│       ├── 01_knn.ipynb
│       ├── 02_svc.ipynb
│       ├── 03_random_forest.ipynb
│       ├── 04_xgboost.ipynb
│       ├── 05_lgbm.ipynb
│       ├── 06_catboost.ipynb
│       ├── 07_ann.ipynb
│       └── 08_lgbm_catboost_xgboost_ensemble.ipynb
│
├── Research paper/                     # Reference papers
│
├── eda_and_data_preprocessing.ipynb    # Exploratory data analysis
├── README.md                           # Project documentation
└── .gitattributes                      # Git configuration
```

## 🤖 Models I implemented

I implemented these 7 machine learning models:

### Basic Models:
1. **KNN** - K-Nearest Neighbors
2. **SVC** - Support Vector Classifier
3. **Random Forest** - Many decision trees combined

### Advanced Models:
4. **XGBoost** - Advanced tree-based model
5. **LightGBM** - Fast tree-based model
6. **CatBoost** - Good with categorical data
7. **ANN** - Artificial Neural Network

### Ensemble Models:
8. **Voting** - Let models vote on predictions (Hard and Soft Voting)
9. **Stacking** - Use model predictions as input to another model

## 📈 How I Measured Performance

I used these metrics to compare models:

| Metric | What It Measures |
|--------|------------------|
| **Accuracy** | How many predictions were correct |
| **Precision** | How many of the "yes" predictions were actually "yes" |
| **Recall** | How many actual "yes" cases were found |
| **F1-Score** | Balance between precision and recall |

I also created:
- **ROC Curves** - Shows model's ability to distinguish classes
- **Confusion Matrices** - Shows what types of mistakes models make
- **Comparison Charts** - Easy visual comparison of all models

## 📊 Results

### Best Performing Model:
**Model Name**: LightGBM
**Accuracy**: 98.44
**Why it worked best**: its gradient boosting framework with leaf-wise tree growth efficiently captures complex patterns in high-dimensional EEG signals

### Key Findings:
1. [Finding 1 - Tree-based models worked better than linear models
2. [Finding 2 - Combining models improved accuracy making them second performant after lgbm

## 🎓 About This Project

### Course Information:
- **Course**: Machine Learning
- **Institution**: Marwadi University
- **Semester**: 5
- **Instructor**: Nishit Kotak

### What I Learned:
1. How different machine learning algorithms work
2. How to tune models for better performance
3. How to compare models properly
4. How to create useful visualizations
5. How to document a complete ML project

## 👤 Author

**Name**: Chan Evan Wesley 
**Student ID**: 92301733072  
**Email**: chanevan.std@gmail.com 
**Subject**: Machine Learning  
