#**Behavioral Cloning** 

Author : Manoj Kumar Subramanian



## Overview

This repository is as part of my Submission to the Project 3: Behavioral Cloning Project for the Udacity Self Driving Car Nano Degree Program.

In this project, deep neural networks and convolutional neural networks are used to clone driving behavior. The model is trained, validated and tested using Keras. The model will output a steering angle to an autonomous vehicle with the Udacity provided simulator. 

First we need to steer a car around a tracks for data collection. Then we'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the tracks.

### Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report




## Rubric Points
####1. Submission includes all required files and can be used to run the simulator in autonomous mode

### Review Set

To meet specifications, the project required submitting five files: 

- [model.py](model.py) (script used to create and train the model)
- [drive.py](drive.py) (script to drive the car)
- a model file, [model_track_1.h5](model_track_1.h5) (a trained Keras model)
- a report [writeup](writeup_report) file
- a [video](model_track_1__run.mp4) file recording of the vehicle driving autonomously around the track for at least one full lap 

As an optional exercise the model is extended to run in track 2 jungle theme which is slightly complex compared to track 1, with few ramps and downs, shadows and with sharp turns.

The same model.py script is used to train the model, [model_track_both.h5](BothTracks/model_track_both.h5) and this [video2](BothTracks/model_track_both_run.mp4) provides the simulation corresponding to the trained model parameters on both the tracks.

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 [--speed 12]
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection. The optional argument speed is the target set speed for the simulator model to run. The drive.py model uses a PI control logic to adhere to the set speed.

For the model file model_track_1.h5, the track 1 runs with the set speed from 8 to 16 without going off road.

For the generated model file model_track_both.h5, both the tracks run with the set speed as 9 to 12 without going off road.

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model, inspired by both the NVIDIA and LeNet5 model architecture, consists of layers that can be visualized as extended LeNet5 model but a reduced NVIDIA model. 



The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. 

```
dict_keys(['loss', 'val_loss'])

[0.021351781623013835, 0.011106089238755987, 0.011046232438950551, 0.010437979799137905, 0.010400431804672474]

[0.011114087152796296, 0.011068746433731649, 0.011129142931447579, 0.0092389183477140386, 0.010090619466258305]
```

The converging values between the training loss and the validation loss suggests that the model is not overfitting the data.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer with the default parameters, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

Initially, I captured about 2 laps of the first track data on to a folder.

I have followed the same sequence of code that was mentioned in the lecture videos. By making the dataset samples from the csv file, splitting the set into training and validation samples, using a generator to batch process the samples, as it is from the lecture videos, with the setting of steering corrections of 0.2 for the left and right images, I was able to move around the vehicle in the autonomous mode.

My first step was to use a convolution neural network model similar to the LeNet model but since the size of the image to compare is quite high 160x320 compared to the 32x32 image set discussed in LeNet. So I added 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a 4 convolution layers (2 5x5 and 2 3x3 filter sizes) in the network, each having a rigorous Max pooling layer to cut down the input size with increase number of filters followed by 3 dense layers. I have chosen the number of filters to be 16, 32, 48 and 64 as a random but progressive but the first run itself proven that the model is pretty fine with the initial level of predictions. So, I retuned the base architecture to suit the dataset with only additions of drop out layers. The default, (lecture referenced) , Mean square error loss function and ADAM optimizer without any tweaking in the parameters were used.

Here is a visualization of the architecture.



####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded about two laps on track one using center lane driving. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back itself when it approaches the road edge. This is enhanced by the left and right images are they provide a good offset for the data samples. Then the behavior at the sharp corners were recorded by carefully driving at a slow pace multiple times.

To augment the data sat, I also flipped images and angles.

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

