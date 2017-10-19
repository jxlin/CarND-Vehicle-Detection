# Vehicle Detection

This project was cloned from [Udacity](https://github.com/udacity/CarND-Vehicle-Detection) and modified by me(olala7846@gmail.com).

## Description
Created a vehicle detection and tracking pipeline with OpenCV, histogram of oriented gradients (HOG), and support vector machines (SVM). Optimized and evaluated the model on video data from a automotive camera taken during highway driving.

### Lane Detection
* Calibrate the camera using OpenCV with chessboard images
* Detect lane lines using perspective transform and edge detection, the result capable of detecting lanes roughly 10 meters ahead and calculates the curvature of the current lane.

### Vehicle Tracking
Detect vehicles using computer vision (HOG) with machine learning (SVM) skills.
Track the vehicle using sliding window and smooth the result using temporal a heatmap.


##### Click the following image to play video

[![MY PROJECT RESULT](http://img.youtube.com/vi/5Y0BFjzWTwU/0.jpg)](http://www.youtube.com/watch?v=5Y0BFjzWTwU "Video tracking and Lane line detection")

## Setup
* To run the project, first setup environment using [Udacity Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
* To view more detail about this project, please reference [My Report](./report.md)
* To generate the combined (Vehicle detection and Lane line detection) video please see [image_pipeline.py](./image_pipeline.py)



