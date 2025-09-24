# Binary-Classification-with-Neural-Networks-on-the-Census-Income-Dataset
Binary Classification with Neural Networks on the Census Income Dataset

## AIM:
To build, train, and evaluate a deep learning model using PyTorch for tabular data that contains both categorical and continuous features, in order to classify income labels.

## PROCEDURE:
## 1. Import Required Libraries:
Import PyTorch for model building.

Import NumPy, Pandas for data handling.

Import Matplotlib for plotting.

Import shuffle from scikit-learn for randomizing rows.

## 2. Load and Explore Dataset:
Load the dataset (income.csv) using Pandas.

Print dataset size, column names, and class distribution of the target label.

## 3. Identify Feature Types:
Define categorical columns (e.g., sex, education, occupation).

Define continuous columns (e.g., age, hours-per-week).

Define target column (label).

## 4. Preprocess Data:
Convert categorical columns into category type.
Shuffle the dataset and reset indices.

Get the number of categories for each categorical column.

## 5. Convert Data into Tensors:
Convert categorical values into category codes.

Convert continuous values into float tensors.

Convert labels into long tensors.

Split dataset into training and testing sets.

## 6. Model Compilation
Loss Function: CrossEntropyLoss to measure prediction error for classification.

Optimizer: Adam with learning rate 0.001 for efficient parameter updates.

Random Seed: Set to ensure reproducibility of results.

## 7. Model Training
The training loop runs for 300 epochs.

Each epoch involves:

Forward pass: the model generates predictions for the training data.

Loss computation: difference between predictions and true labels is calculated.

Backpropagation: gradients are computed.

Weight updates: optimizer adjusts model parameters to reduce loss.

Training losses are recorded after each epoch to monitor model learning.

Intermediate losses are printed every 25 epochs for progress tracking.

## 8. Training Loss Visualization
A loss curve is plotted using Matplotlib to visualize how the loss decreases over epochs.

The curve provides insight into model convergence and helps identify issues such as overfitting or underfitting.

## 9. Model Evaluation
The trained model is switched to evaluation mode (disabling dropout and other training-specific behaviors).

Test data is passed through the model to generate predictions.

## Evaluation metrics computed:

Cross-Entropy Loss on test data.

## Accuracy: 
percentage of correct predictions compared to actual labels.

## Example outcome:

CE Loss: 0.39 4012 out of 5000 = 80.24% correct
