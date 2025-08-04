# Task 2: Deep Learning Image Classification

company: CODTECH IT SOLUTIONS

NAME: OM KUMAR

INTERN ID: CT04DH1971

DOMAIN: DATA SCIENCE

DURATION: 4 WEEKS

MENTOR: Muzammil

DESCRIPTION OF TASK

Task 2 – Deep Learning Image Classification with TensorFlow

Project Overview

As part of my Data Science Internship at CODTECH, I successfully implemented a deep learning model for image classification using TensorFlow and Keras. This project focuses on building a Convolutional Neural Network (CNN) to classify handwritten digits from the well-known MNIST dataset, which contains 70,000 grayscale images of digits (0 to 9).

This task helped me gain hands-on experience in designing deep learning architectures, training models, evaluating performance, and visualizing results. The goal was not only to achieve high classification accuracy but also to automate a simple yet powerful image recognition task.


Technologies Used

Python 3

TensorFlow & Keras

NumPy

Matplotlib

Jupyter Notebook / Python Script

Model Details

The CNN architecture used in this project includes:

Two Convolutional Layers with ReLU activation

MaxPooling Layers to reduce dimensionality

Flattening Layer to convert 2D matrices to vectors

Dense Layers, with the final layer using softmax activation for multi-class classification

The model was compiled using the Adam optimizer and trained using Sparse Categorical Crossentropy as the loss function.

Training & Evaluation

The model was trained for 5 epochs on the training dataset.

Achieved a test accuracy of over 98% on unseen data.

After training, the model was tested on the test dataset, and sample predictions were visualized using Matplotlib.

Visualization

To make the results interpretable, one of the test images was displayed using matplotlib, alongside the predicted digit label by the model. This provides a basic but effective demonstration of the model’s prediction capability.

Files Included

image_classification_model.py – Python script to train and test the CNN model

output_sample.png – Image showing a sample prediction result

README.md – This file, providing an overview and instructions to run the project

How to Run

1. Install required libraries:

pip install tensorflow matplotlib


2. Run the script:

python image_classification_model.py


3. The model will be trained, evaluated, and a prediction image will be displayed.

Learning Outcomes

Through this task, I strengthened my understanding of:

Building CNN models with TensorFlow

Image preprocessing and normalization

Model training, evaluation, and result interpretation

Using Matplotlib for visual output of predictions


This task gave me practical exposure to deep learning workflows and will be a valuable addition to my machine learning portfolio.
