import csv
import cv2
import numpy as np
import os
import sklearn
import argparse
from sklearn.model_selection import train_test_split

# Load data from the csv file
def load_data(csv_file):
    samples = []
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

# Using the generator to batch the samples
def generator(samples, batch_size=32, range_val=3):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        #shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                
                for i in range(range_val):
                    # Use the split function to get the filename from the path
                    name = './IMG/'+batch_sample[i].split('\\')[-1]
                    image = cv2.imread(name)
                    #center_angle = float(batch_sample[3])
                    
                    if i==1: #Left image
                        angle = float(float(batch_sample[3]) + 0.2)
                    elif i==2: # Right image
                        angle = float(float(batch_sample[3]) - 0.2)      
                    else: # Center image
                        angle = float(batch_sample[3])

                    # Append the data to dataset
                    images.append(image)
                    angles.append(angle)
                    
                    
                    # Flip the image and add to the dataset
                    image_flipped = cv2.flip(image,1)
                    angle_flipped = (-angle)
                    images.append(image_flipped)
                    angles.append(angle_flipped)

            # Convert the images to arrays using numpy
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# Keras related imports
from keras.models import Sequential
from keras.layers import Flatten,Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.models import load_model

# Function that creates model 
def create_Model():

    # Creating a sequential model
    model = Sequential()
    
    # Use lambda to normalize the values
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

    # trim image to only see section with road
    model.add(Cropping2D(cropping = ((60,15),(0,0))))

    # adding convolution layers with rigorous maxpooling with elu activation function
    model.add(Convolution2D(16,5,5,activation="elu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(32,5,5,activation="elu"))

    # Dropout layer 1 to reduce overfitting
    model.add(Dropout(.25))
    model.add(MaxPooling2D())
    model.add(Convolution2D(48,3,3,activation="elu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64,3,3,activation="elu"))
    model.add(MaxPooling2D())

    # Flatten
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dropout(.25))
    model.add(Dense(60))
    model.add(Dense(1))

    # Use meas square error loss function and adam optimizer
    model.compile(loss = 'mse', optimizer = 'adam')
    return model

# Function to train model
def train_Model(model,train_samples, validation_samples,range_val,batch_size,epoch_size):
    # Train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size, range_val= range_val)
    validation_generator = generator(validation_samples, batch_size=batch_size, range_val = range_val)
    
    history_object = model.fit_generator(train_generator, samples_per_epoch = (range_val*len(train_samples)), validation_data = validation_generator, nb_val_samples = (range_val*len(validation_samples)), nb_epoch = epoch_size, verbose = 1)

    ### print the keys contained in the history object
    print(history_object.history.keys())
    print(history_object.history['loss'])
    print(history_object.history['val_loss'])
    return model

# Function to save model
def save_Model(model,save_filename):
    model.save(save_filename)
    return

# Use parser args to easily config parameters from cmd prompt
def main():
    parser = argparse.ArgumentParser(description='Training the Behaviour model')
    parser.add_argument(
        '--csv',
        type=str,
        default='driving_log.csv',
        help='Path to the csv file containing training set.'
    )
    parser.add_argument(
        '--s',
        type=str,
        default='model.h5',
        help='File name to save model.')
    parser.add_argument(
        '--l',
        type=str,
        default='',
        help='File name to load a saved model.')
    parser.add_argument(
        '--u',
        type=str,
        default='y',
        help='Use Left and Right images. y/n')
    parser.add_argument(
        '--e',
        type=int,
        default=5,
        help='Total number of epochs to be used')
    parser.add_argument(
        '--b',
        type=int,
        default=32,
        help='Batch sample size')
    args = parser.parse_args()

    csv_file = args.csv
    samples = load_data(csv_file)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    
    use_left_right = args.u
    if (use_left_right == 'y') or (use_left_right == 'Y'):
        range_val = 3
    else:
        range_val = 1

    if args.l == '':
        model = create_Model()
    else:
        model = load_model(args.l)
    
    model = train_Model(model,train_samples, validation_samples, range_val,args.b,args.e)
    save_Model(model,args.s)


if __name__ == '__main__':
    main()
