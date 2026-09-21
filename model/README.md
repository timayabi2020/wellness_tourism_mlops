
# Wellness Tourism Purchase Prediction Model

## Model Description

This model predicts whether a customer is likely to purchase a Wellness
Tourism Package.

## Target

`ProdTaken`

- 0: Customer did not purchase the package
- 1: Customer purchased the package

## Model

Gradient Boosting Classifier with a preprocessing pipeline for numerical and
categorical features.

## Model Selection

Decision Tree, Random Forest, and Gradient Boosting models were evaluated.

F1 Score was used as the primary model-selection metric because the target
variable is imbalanced.

## Test Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.9274 |
| Precision | 0.9160 |
| Recall | 0.6855 |
| F1 Score | 0.7842 |
| ROC-AUC | 0.9495 |

## Experiment Tracking

Model experiments and hyperparameters were tracked using MLflow.

## Intended Use

The model is intended to demonstrate an end-to-end MLOps workflow for
predicting customer interest in a Wellness Tourism Package.
