# 🔧 Predictive Maintenance System

An end-to-end Machine Learning system for predicting potential machine failures using industrial sensor data.

The project combines data preprocessing, feature engineering, multiple machine learning models, hyperparameter optimization, model evaluation, probability-based risk assessment, and an interactive Streamlit dashboard.

---

## 🚀 Project Overview

Unexpected machine failures can lead to production downtime, maintenance costs, and reduced operational efficiency.

This project aims to predict whether a machine is likely to experience a failure based on operational parameters such as:

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

The trained machine learning models analyze these parameters and provide a failure prediction along with probability-based risk information.

---

## 🎯 Objectives

- Predict potential machine failures before they occur
- Compare multiple machine learning classification algorithms
- Handle class imbalance during model training
- Perform feature engineering on industrial sensor data
- Optimize model hyperparameters
- Evaluate models using appropriate classification metrics
- Generate probability-based predictions
- Support individual machine predictions
- Support batch CSV predictions
- Provide an interactive machine-health dashboard

---

## 🧠 Machine Learning Models

The project evaluates multiple classification algorithms, including:

- Random Forest
- Decision Tree
- K-Nearest Neighbors
- Logistic Regression
- Ensemble / Stacking Model

Hyperparameter optimization is performed using GridSearchCV.

---

## ⚙️ Machine Learning Pipeline

```text
Industrial Sensor Dataset
          ↓
Data Cleaning
          ↓
Exploratory Data Analysis
          ↓
Feature Engineering
          ↓
Train/Test Split
          ↓
Preprocessing Pipeline
          ↓
Hyperparameter Optimization
          ↓
Multiple ML Models
          ↓
Model Evaluation
          ↓
Ensemble Prediction
          ↓
Failure Probability
          ↓
Maintenance Risk Assessment
