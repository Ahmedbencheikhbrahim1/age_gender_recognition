# age_gender_recognition

This project is a part of my virtual internship in Growinterne. during this project I tried to demonstrates a real-time age and gender recognition system using deep learning models implemented with OpenCV and Caffe. The system captures video from a webcam, detects faces, and predicts the age and gender of the detected faces.

# Table of Contents
Introduction
Features
Requirements
Usage
Models
Code Explanation

# Introduction
This project uses pre-trained Caffe models for detecting faces and predicting age and gender. The face detection model identifies faces in the video frames, and the age and gender models predict the respective attributes of the detected faces.

# Features
Real-time face detection.
Real-time age prediction.
Real-time gender prediction.

# Requirements
Python 3.x
OpenCV
Caffe models for face detection, age prediction, and gender prediction.

# Models
Face Detection Model: opencv_face_detector.pbtxt, opencv_face_detector_uint8.pb
Age Prediction Model: age_deploy.prototxt, age_net.caffemodel
Gender Prediction Model: gender_deploy.prototxt, gender_net.caffemodel

# Code Explanation
in age_gender_recognition.py I tried to comment every line of code so it will be clear and uselful for everyone.