Breast Cancer Detection using Deep Learning

This repository contains the implementation of a deep learning-based system for the automated classification and segmentation of breast cancer using ultrasound images. This project was developed as part of the BME 310 course by Group 11.

🎯 Problem Definition

Breast cancer is a leading cause of mortality among women worldwide. Early and accurate diagnosis is critical for effective treatment and improving survival rates. This project aims to address the challenges of manual ultrasound interpretation by developing a highly accurate, automated detection system capable of classifying tumors as Benign, Malignant, or Normal.

🛠️ Methodology & Tech Stack

Our approach involves transitioning from a custom Convolutional Neural Network (CNN) to a powerful Transfer Learning model to achieve superior performance.

Data Assembly: Collected and structured a dataset of breast ultrasound images containing Benign, Malignant, and Normal cases.

Data Preprocessing & Augmentation: Applied necessary transformations to enhance image quality and artificially expand the training dataset to prevent model overfitting.

Model 1: Custom CNN (Deep Breast Net): Initially implemented a sequential CNN model featuring input layers, convolutional layers, flattening, and dense layers as a baseline.

Model 2: Transfer Learning (ResNet101): Implemented a deep learning approach utilizing the pre-trained ResNet101 architecture. This allowed the model to leverage complex feature extraction learned from massive datasets, fine-tuned specifically for our ultrasound images.

Segmentation Processing: Integrated segmentation techniques to not only classify but also generate predicted masks to localize the tumor regions within the ultrasound scans.

Web Application

To make the model accessible, we developed an interactive web interface. Users can easily upload an ultrasound image, and the backend model will instantly predict the most likely label (Benign, Malignant, or Normal).
