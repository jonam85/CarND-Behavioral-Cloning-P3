# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository is as part of my Submission to the Project 3: Behavioral Cloning Project for the Udacity Self Driving Car Nano Degree Program.

In this project, deep neural networks and convolutional neural networks are used to clone driving behavior. The model is trained, validated and tested using Keras. The model will output a steering angle to an autonomous vehicle with the Udacity provided simulator. 

First we need to steer a car around a tracks for data collection. Then we'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the tracks.

### Review Set

To meet specifications, the project will required submitting five files: 

* [model.py](model.py) (script used to create and train the model)
* [drive.py](drive.py) (script to drive the car)
* a model file, [model_track_1.h5](model_track_1.h5) (a trained Keras model)
* a report [writeup](writeup_report) file
* a [video](model_track_1__run.mp4) file recording of the vehicle driving autonomously around the track for at least one full lap 

As an optional exercise the model is extended to run in track 2 jungle theme which is slightly complex compared to track 1, with few ramps and downs, shadows and with sharp turns.

The same model.py script is used to train the model, [model_track_both.h5](BothTracks/model_track_both.h5) and this [video2](BothTracks/model_track_both_run.mp4) provides the simulation corresponding to the trained model parameters on both the tracks.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The simulator can be downloaded from the classroom. In the classroom, sample data have also been provided that can optionally be used to help train the model.

More information about this project and the usage of the files are in [writeup_report](writeup_report) file.

