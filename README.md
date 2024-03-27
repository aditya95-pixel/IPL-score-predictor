# IPL Cricket Score Predictor

This repository contains a deep learning model for predicting the total score in IPL (Indian Premier League) cricket matches based on various features such as venue, batting team, bowling team, batsman, and bowler.

## Overview

The model is built using TensorFlow and Keras, two popular libraries for deep learning and neural networks. It utilizes a multi-layer neural network architecture to learn the relationship between the input features and the target variable (total score). The model is trained on historical IPL match data.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- TensorFlow
- Keras
- tkinter

## Usage
Make sure you have the IPL match data file (ipl_data.csv) in the same directory as the code.

Run the Python script main.py. This script preprocesses the data, trains the neural network model, and saves the trained model to a file (model_saved.h5).

Once the model is trained and saved, you can use it to make predictions on new data.

## Model Details
The input features include venue, batting team, bowling team, batsman, and bowler.

Categorical features are encoded using LabelEncoder.

The data is split into training and testing sets.

Features are scaled using MinMaxScaler.

The neural network architecture consists of multiple dense layers with ReLU activation functions.

The loss function used for training is Huber loss, which is less sensitive to outliers compared to mean squared error.

The trained model is evaluated using mean absolute error (MAE).

## Usage Instructions

Run the provided Python script main2.py. This script launches a GUI (Graphical User Interface) for predicting cricket scores based on user input.

In the GUI window, select the venue, batting team, bowling team, batsman, and bowler from the dropdown menus.

Click on the "Predict Score" button to see the predicted score displayed below.
