# Air Quality Time-Series Regression with Neural Networks
This repository presents a comprehensive machine learning pipeline for time-series regression based on air quality sensor data. The work focuses on the design, implementation, and systematic evaluation of neural network models for predicting benzene (C₆H₆) concentration from multisensor measurements.

The project includes data preprocessing, feature scaling, sequence construction, and extensive experimental analysis of multiple neural architectures, including Multilayer Perceptrons (MLP), Convolutional Neural Networks (CNN), and Recurrent Neural Networks (LSTM/GRU). Model performance is assessed using reproducible experimental protocols, controlled hyperparameter studies, and robust evaluation metrics.

Special attention is given to computational efficiency, experiment reproducibility, and scalable evaluation strategies, such as one-at-a-time and grid-based hyperparameter analysis. The repository is structured to support both scientific experimentation and practical application in environmental monitoring systems.

## Problem Description

Air quality monitoring relies on low-cost chemical sensors that are often affected by cross-sensitivity, environmental conditions, and sensor drift. Instead of using a dedicated benzene sensor, this project formulates the task as a **regression problem**, where benzene concentration is estimated using data from multiple correlated sensors and meteorological variables.

**Problem formulation:**
- **Input:** Multisensor gas readings and environmental features  
- **Output:** Continuous benzene concentration value (µg/m³)

## Dataset

The dataset consists of **hourly averaged measurements** collected over approximately one year in a polluted urban area in Italy.

**Key characteristics:**
- ~9,300 observations
- Multisensor metal-oxide gas readings
- Temperature and humidity features
- Ground-truth reference measurements
- Missing values marked as `-200`

Missing values are handled via interpolation for input features, while invalid target values are removed to ensure reliable regression performance.

## Implemented Models

The following model families are implemented and evaluated:

- **MLP (Multilayer Perceptron)**  
  Baseline feed-forward regression model.

- **CNN (1D Convolutional Neural Network)**  
  Learns local feature interactions across sensor dimensions.

- **RNN (LSTM / GRU)**  
  Models temporal dependencies using explicit time-window sequences.

All models are trained using repeated runs with controlled random seeds and early stopping.

## Experimental Design

The repository supports two experimental strategies:

### One-at-a-Time (OAT) Hyperparameter Analysis
- Each hyperparameter is varied independently
- Other parameters are kept fixed
- Enables clear interpretation of parameter influence
- Significantly reduces computational cost

### Full Grid Evaluation
- Exhaustive exploration of parameter combinations
- Used for benchmarking and validation

Each configuration is trained multiple times, and results are aggregated using mean and standard deviation metrics.

## Evaluation Metrics

Models are evaluated on **train**, **validation**, and **test** sets using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (R²)

Results are stored incrementally in CSV files to ensure fault tolerance during long-running experiments.

