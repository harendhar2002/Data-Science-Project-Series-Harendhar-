
# Breast Cancer Prediction Using Support Vector Machine (SVM)

This project demonstrates a machine learning approach for predicting breast cancer malignancy using a Support Vector Machine (SVM) model. The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset.


## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Cross-Validation](#cross-validation)
- [Results](#results)

## Project Overview

This project uses the Support Vector Machine (SVM) algorithm to classify breast tumors as benign or malignant based on a variety of features. Key steps include:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature selection and engineering
- SVM model training
- Model evaluation and tuning using GridSearchCV

## Dataset

The dataset used in this project is the [Breast Cancer Wisconsin (Diagnostic) dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data) from Kaggle.

- **Features:** 30 features computed from a digitized image of a breast mass.
- **Target:** Binary classification where 'M' stands for malignant and 'B' stands for benign.

### Columns
- **Input Features:** Various attributes such as `radius_mean`, `texture_mean`, `smoothness_mean`, etc.
- **Target Feature:** Diagnosis (`M` for malignant, `B` for benign).

## Requirements

The project requires the following libraries:

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Loading and Cleaning Data:

Data is read using Pandas from a CSV file.
- Irrelevant columns like 'id' and 'Unnamed: 32' are removed.
- The diagnosis column is mapped to binary values: malignant ('M') is 1, and benign ('B') is 0.
### Exploratory Data Analysis (EDA):
- The script explores the dataset with descriptive statistics.
- The correlation matrix is visualized using Seaborn's heatmap to show relationships between variables.
- Pairplot visualizes relationships between selected features with respect to the diagnosis.
### Feature Selection and Reduction:
- Unnecessary columns like 'worst' metrics and some 'mean', 'se' metrics are removed to reduce dimensionality.
- **Train-Test Split:**
- The dataset is split into training and testing sets with an 80-20 ratio.
- **Feature Scaling:**
- StandardScaler is used to scale the features to improve model performance.
- **SVM Model Training:**
- A basic SVM model is trained using the scaled training data.
- Predictions are made on the test set, and accuracy, confusion matrix, and classification report are generated.
### Model Tuning with GridSearchCV:
- A grid search is used to tune the SVM's hyperparameters (C, gamma, kernel) to find the best model.
- The best model is used for prediction, and evaluation metrics are reported again.

### Model Evaluation
After training the model, we evaluate its performance using:

- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report**
