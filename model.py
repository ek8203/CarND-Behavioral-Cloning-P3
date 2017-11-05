# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 23:07:37 2017

@author: nmkekrop
"""

"""
This file containing the python script to create and train a model to clone driving behavior.
"""
# Import modules
import csv
import cv2
import numpy as np
import sklearn

# Use csv lib to read the measurements from the driving_log.csv file
datalog_dir = "data/"
lines = []
with open(datalog_dir + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
def read_image(src_path):
    """
    Read an image frame from file
    Input: src_path - full name (string) of the image file
    Returns: image data in RGB format
    """    
    # update the dir path of the image file
    filename = src_path.split('\\')[-1]
    current_path = datalog_dir +'IMG/' + filename    
    # read a frame and add to the list
    image = cv2.imread(current_path)
    # convert BGR to RGB since the test simulator (drive.py) is using RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

"""
Below are two methods of training the models:
Method 1: training on the dataset of the entire training data
Method 2: training on the batch data - a subset of the training data
"""
use_generator = True

# Method 1
def load_all_data(lines):
    """
    1. Load all images and measurements at once. 
    2. Build augmented images and measurements
    3. Put all together in one training dataset
    Returns Numpy arrays of the training dataset 
    """    
    # Placeholders
    images = []
    measurements = []
    images_left = []
    measurements_left = []
    images_right = []
    measurements_right = []
    aug_images = []
    aug_measurements = []

    # Load all collected data
    for line in lines:
        steering_center = float(line[3])
        # create adjusted steering measurements for the side camera images
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction
    
        # center camera image
        image = read_image(line[0])
        images.append(image)    
        measurements.append(steering_center)        

        # left camera image
        image = read_image(line[1])
        images_left.append(image)    
        measurements_left.append(steering_left)        

        # right camera image
        image = get_image_angle(line[2])
        images_right.append(image)    
        measurements_right.append(steering_right)

    # Build augmented center camera images
    for image, steer_ang in zip(images, measurements):
        # augmented images - flipped vertically (flipCode = 1)
        aug_images.append(cv2.flip(image, 1))
        # reverse the angles
        aug_measurements.append(steer_ang * (-1.))

    # Put everything together in one training dataset
    images.extend(aug_images)
    images.extend(images_left)
    images.extend(images_right)
    measurements.extend(aug_measurements)
    measurements.extend(measurements_left)
    measurements.extend(measurements_right)
    
    # Put the training dataset in Numpy arrays
    X_train = np.array(images)
    y_train = np.array(measurements)
    
    return X_train, y_train

# Method 2
def generator(samples, correction = 0.2, batch_size=32):
    """
    Input batch data generator
    """
    num_samples = (len(samples)//batch_size)*batch_size
    
    while True: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            
            for batch_sample in batch_samples:
                # get and adjust steering angles
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                # center camera image
                image = read_image(batch_sample[0])
                images.append(image)    
                angles.append(steering_center)
                # add augmented images - flipped vertically (flipCode = 1)
                images.append(cv2.flip(image, 1))
                # reverse the angles
                angles.append(steering_center * (-1.))

                # left camera image
                image = read_image(batch_sample[1])
                images.append(image)    
                angles.append(steering_left)        
                # add augmented images - flipped vertically (flipCode = 1)
                images.append(cv2.flip(image, 1))
                # reverse the angles
                angles.append(steering_left * (-1.))

                # right camera image
                image = read_image(batch_sample[2])
                images.append(image)    
                angles.append(steering_right)        
                # add augmented images - flipped vertically (flipCode = 1)
                images.append(cv2.flip(image, 1))
                # reverse the angles
                angles.append(steering_right * (-1.))
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
   
# Build a simple Keras model
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Dropout, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def grayscale_image(input):
    from keras.backend import tf as ktf
    return ktf.image.rgb_to_grayscale(input)

def resize_image(input):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (64, 64))

def InputNormalized(shape=(160,320,3)):
    """
    Model input and normalization layers.
    """
    from keras.backend import tf as ktf
    # Sequential(shape=(batch_size, height, width, channels))
    model = Sequential()
    # Cropping the hood from bottom and the background from top of the image    
    model.add(Cropping2D(cropping=((50,25), (0,0)), input_shape=shape))
    # Resize the image
    model.add(Lambda(resize_image))
    # Grayscale the image
    model.add(Lambda(grayscale_image))
    # Image normalization. That lambda layer could take each pixel in an image and run it through the formulas:
    # pixel_normalized = pixel / 255
    # pixel_mean_centered = pixel_normalized - 0.5
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    return model
    
def LeNet(shape=(160,320,3), keep_prob=0.5):
    """
    LeNet-5 model architecture:
    INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
    """
    # INPUT:
    model = InputNormalized(shape)
    # CONV->ACT: 6 filters, 5x5 kernel, valid padding and ReLU activation.
    model.add(Convolution2D(16, 5, 5, activation='relu'))
    # POOL: 2x2 max pooling layer immediately following your convolutional layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # CONV->ACT: 16 filters, 5x5 kernel, valid padding and ReLU activation.
    model.add(Convolution2D(32, 5, 5, activation='relu'))
    # POOL: 2x2 max pooling layer immediately following your convolutional layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # FLATTEN: 800
    model.add(Flatten())
    model.add(Dropout(keep_prob))
    # FC: 400 and ReLU activation 
    model.add(Dense(400, activation='relu'))
    # FC: 120 and ReLU activation
    model.add(Dense(120, activation='relu'))
    # FC: 1
    model.add(Dense(1))
    return model
    
def nVidia(shape=(160,320,3), keep_prob=1.0):
    """
    nVidea model architecture:
    INPUT -> 3x(CONV -> ACT) -> 2x(CONV -> ACT)-> FLATTEN -> 4xFC
    """
    model = InputNormalized(shape)
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    #model.add(Dropout(keep_prob))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    #model.add(Dropout(keep_prob))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    #model.add(Dropout(keep_prob))
    model.add(Convolution2D(64,3,3, activation='relu'))
    #model.add(Dropout(keep_prob))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(keep_prob))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

model = LeNet(keep_prob=0.3)
#model = nVidia(keep_prob=0.05)

# Use Adam optimizer and MSE loss function because it is a regression network. 
# The model has to minimize the error between the predicted steering measurements and the true measurements
model.compile(optimizer='adam', loss='mse')

print("Number of measurement lines:\t{}".format(len(lines)))

if use_generator:
    """
    Train the model using the generator function
    """
    # Split collected data into a train and validation datasets
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(lines*6, test_size=0.2)
    print("Number of train samples:\t{}".format(len(train_samples)))
    print("Number of validation samples:\t{}".format(len(validation_samples)))

    BATCH_SIZE = 32
    EPOCHS = 8
    
    train_generator         = generator(train_samples, batch_size = BATCH_SIZE)
    validation_generator    = generator(validation_samples, batch_size = BATCH_SIZE)

    # test the generator
    #for i in range(3):
    #    X,y = next(train_generator)
    #    print(X.shape, y.shape)
    #exit()         

    history = model.fit_generator(train_generator, samples_per_epoch = len(train_samples), 
                    validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = EPOCHS)
else:
    # Split the data for train and validation sets and suffle the data
    model.fit(X_train, y_train, nb_epoch=n_epoch, validation_split=0.2, shuffle=True)

print(model.summary())
                    
#print("Done")
#exit()
            
# Save the trained model for the test run with the simulator
model.save("model.h5")  
print("Model saved")
   
    
    
    
    
