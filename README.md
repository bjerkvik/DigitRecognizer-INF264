# DigitRecognizer-INF264

This repository contains a machine learning project developed for the INF264 course. The aim of the project is to build a classifier capable of recognizing handwritten hexadecimal digits from images.

## Table of Contents
- [Framing the Problem](#framing-the-problem)
- [Data Acquisition](#data-acquisition)
- [Data Exploration](#data-exploration)
- [Data Preprocessing](#data-preprocessing)
- [Model Exploration and Selection](#model-exploration-and-selection)
- [Model Fine-tuning](#model-fine-tuning)
- [Evaluation and Results](#evaluation-and-results)

## Framing the Problem
A explanation of the problem statement, including the objectives and expected outcomes of the project.

## Data Acquisition
Dataset is loaded and split into training and test sets.

## Data Exploration
The data exploration phase begins with visualizing data samples to capture the diversity of handwritten hexadecimal digits. This is followed by an analysis of the distribution of these digits across various classes, highlighting any imbalances. Additionally, data integrity checks are performed to ensure there are no missing values or corrupt images.

## Data Preprocessing
Description of the data preprocessing steps.

## Model Exploration and Selection
The focus of this phase is to explore various machine learning models. The project experiments with a mix of traditional classifiers and Convolutional Neural Networks (CNNs).

A function `train_cnn_model` is designed to train CNN models using architectures like ResNet18 and EfficientNetV2, evaluating them based on macro F1 score and balanced accuracy. This function includes data loading, model selection, training, and evaluation steps, ensuring reproducibility and consistency.

Additionally, a `classificationScores` function is created to evaluate traditional classifiers, including XGBoost, Logistic Regression, RandomForest, k-Nearest Neighbors, and a baseline Dummy Classifier. This evaluation uses cross-validation to compute F1 macro and balanced accuracy scores.

The models are then compared based on their F1 macro scores. The code selects the best performing model among traditional classifiers and CNNs, considering the performance metrics and training time.

## Model Fine-tuning
The model fine-tuning phase involves optimizing the performance of the best-performing models identified in the exploration phase.

For the CNN model, the fine-tuning process starts with the best CNN model from the previous phase. Key enhancements include employing various callbacks such as Mixed Precision for faster training, SaveModelCallback for model saving, ReduceLROnPlateau for adaptive learning rate adjustments, and EarlyStoppingCallback to prevent overfitting. The fine-tuning is conducted for up to 15 epochs, using the learning rate determined through a learning rate finder.

A comprehensive hyperparameter tuning is performed for traditional classifiers using GridSearchCV. This process involves defining parameter grids for the models. The best traditional model undergoes grid search on the training data to find more optimal hyperparameters.

This dual approach to fine-tuning ensures both the CNN and traditional models are optimized, leveraging the strengths of each model type.

## Evaluation and Results
The evaluation phase assesses the performance of the best CNN and traditional models on the test dataset. For the CNN model, predictions and target labels are analyzed, and a confusion matrix is generated to visualize performance. Instances of misclassification are displayed, with eight images showing the true and predicted classes.

The traditional model undergoes a similar evaluation. Its predictions on the test dataset are analyzed, and a confusion matrix is created for a direct performance comparison with the CNN model. Key metrics such as precision, recall, and F1-score for each class are reported in a classification report for both models.

Finally, a comparative analysis is presented, tabulating the Macro F1 Scores and Balanced Accuracies of both models. This comparison provides a clear view of each model's overall accuracy and class-wise balance, offering valuable insights into their capabilities and areas for improvement.

## Dataset
The dataset used for this project is derived from the Extended MNIST (EMNIST) dataset. It consists of two files: `emnist_hex_images.npy` and `emnist_hex_labels.npy`, which can be downloaded from [this link](https://filesender.sikt.no/?s=download&token=01f980ab-6d18-4a21-9d81-fc2ce591123c).

- `emnist_hex_images.npy` contains 107,802 images, each of size 20×20 pixels, flattened into 1D arrays of 400 elements. The values in these arrays range from 0 (black) to 255 (white).
- `emnist_hex_labels.npy` includes corresponding labels for these images, with integers ranging from 0 to 16, where each integer encodes the class the image belongs to. The class '16' represents an empty image.
