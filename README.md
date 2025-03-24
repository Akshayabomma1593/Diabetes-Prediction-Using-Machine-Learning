# Diabetes Prediction Using Machine Learning

## Overview
This project implements an **ensemble-based machine learning approach** for **diabetes prediction** using Python. It utilizes multiple classifiers, including **Naive Bayes, Random Forest, Logistic Regression, and Support Vector Machine (SVM)**, combined using a **soft voting ensemble classifier** to improve prediction accuracy.

## Dataset
The model is trained on the **Pima Indians Diabetes Dataset**, which contains health-related data such as glucose levels, insulin levels, BMI, age, and other attributes that help in predicting whether a person has diabetes or not.

## Features Used
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **Blood Pressure**: Diastolic blood pressure (mm Hg)
- **Skin Thickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **Diabetes Pedigree Function**: Diabetes likelihood based on family history
- **Age**: Age of the individual
- **Outcome**: Target variable (1 = Diabetic, 0 = Non-Diabetic)

## Technologies Used
- **Python**
- **Pandas, NumPy** (Data processing)
- **Scikit-learn** (Machine learning models and evaluation metrics)
- **Matplotlib, Seaborn** (Data visualization)

## Project Structure
```
├── diabetes_prediction.py   # Main script
├── diabetes.csv             # Dataset
├── README.md                # Documentation
└── requirements.txt         # Dependencies


## Usage
1. Run the `diabetes_prediction.py` script to train the model and evaluate performance:
   ```bash
   python diabetes_prediction.py
   ```
2. The script will display:
   - Model accuracy, precision, recall, and F1-score for different classifiers
   - Confusion matrices for visualization
   - ROC curves for performance comparison

## Model Performance
| Classifier         | Accuracy | Precision | Recall | F1-score | AUC-ROC |
|--------------------|----------|------------|--------|-----------|----------|
| Naive Bayes       | 78.57%  | 68.63%   | 67.31% | 67.96%  | 84.16%  |
| Random Forest     | 79.87%  | 70.59%   | 69.23% | 69.98%  | 86.91%  |
| Logistic Regression | 79.87% | 73.33% | 63.46% | 68.84% | 86.10%  |
| SVM               | 79.22%  | 71.74%   | 63.46% | 67.35%  | 86.26%  |
| **Ensemble (Soft Voting)** | **83.18%** | **74.38%** | **68.65%** | **71.48%** | **90.90%** |

## Future Scope
- Implement **deep learning** techniques (e.g., CNNs, RNNs) to further improve accuracy.
- Develop a **web-based or mobile application** for real-time predictions.
- Incorporate **feature engineering** for enhanced model performance.

## License
This project is licensed under the **MIT License**.


