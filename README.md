# AI Classification Project

A machine learning classification project applying a Multi-Layer Perceptron (MLP) neural network to a real-world dataset, with a focus on handling class imbalance and evaluating model quality beyond simple accuracy.

## Overview

This project explores how different preprocessing choices affect classification performance on an imbalanced dataset. Rather than stopping at a headline accuracy number, the project specifically investigates whether the model is genuinely learning to distinguish between classes, using Cohen's Kappa as a more robust evaluation metric alongside standard accuracy, precision, recall, and F1-score.

## What This Project Does

- Performs exploratory data analysis, including class distribution checks and a correlation heatmap
- Encodes target labels and separates features from the target variable
- Addresses class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique)
- Scales features using standardisation
- Builds and trains a Multi-Layer Perceptron (MLP) neural network using TensorFlow/Keras, with batch normalisation applied between layers
- Evaluates the model using accuracy, Cohen's Kappa, a full classification report, and a confusion matrix
- Visualises training/validation accuracy over epochs and the confusion matrix

## Tech Stack

- **Python**
- **Pandas / NumPy** — data handling
- **Seaborn / Matplotlib** — visualisation
- **Scikit-learn** — preprocessing, train/test split, evaluation metrics
- **imbalanced-learn (SMOTE)** — class balancing
- **TensorFlow / Keras** — MLP model

## Key Results

- Achieved **77% test accuracy**
- Improved model reliability from a Cohen's Kappa of **0.46 to 0.54** by applying SMOTE class balancing and batch normalisation
- Balanced F1-scores of **0.76–0.77** across both classes, confirming the model wasn't simply exploiting class imbalance to inflate accuracy

## Why Cohen's Kappa Matters Here

Accuracy alone can be misleading on an imbalanced dataset — a model can score well just by favouring the majority class. Cohen's Kappa measures how much better the model performs than random chance, accounting for that imbalance. Tracking the Kappa improvement (not just accuracy) was the most important decision in this project, and it's the metric that actually confirmed the model was learning something real.

## What I'd Do Differently

Build in the accuracy-vs-Kappa comparison from the very start of a project rather than after initial results, since it fundamentally changes how every subsequent result should be read.

## Files

- `Project_AI_model.py` — full pipeline: data loading, EDA, preprocessing, SMOTE balancing, model training, and evaluation
- `wine.csv` — dataset used for this project
