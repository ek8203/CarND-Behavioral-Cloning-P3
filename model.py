# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 23:07:37 2017

@author: nmkekrop
"""

"""
TODO: Complete the header with description
    
This file containing the python script to create and train the model 
to clone driving behavior.
"""
# import modules
import csv
import cv2
import numpy as np

# TODO: Use csv lib to read the driving_log.csv file
# Read the measurements from the log file
datalog_dir = "data/"
lines = []
with open(datalog_dir + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        #print(line)

        
# TODO: read the image frames and the steering angles

def get_image_angle(src_path):
    
    # update the dir path of img files
    filename = src_path.split('\\')[-1]
    current_path = datalog_dir +'IMG/' + filename
    
    # read a frame and add to the list
    image = cv2.imread(current_path)
    
    return image


images = []
measurements = []
images_left = []
measurements_left = []
images_right = []
measurements_right = []

# the first line is the header - to be excluded
for line in lines:
    steering_center = float(line[3])

    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    # center camera image
    image = get_image_angle(line[0])
    images.append(image)    
    measurements.append(steering_center)        

    # left camera image
    image = get_image_angle(line[1])
    images_left.append(image)    
    measurements_left.append(steering_left)        

    # right camera image
    image = get_image_angle(line[2])
    images_right.append(image)    
    measurements_right.append(steering_right)        
    
# Add augmented center camera images to the training dataset
aug_images = []
aug_measurements = []

for image, steer_ang in zip(images, measurements):
    # add augmented images - flipped vertically (flipCode = 1)
    aug_images.append(cv2.flip(image, 1))
    # reverse the angles
    aug_measurements.append(steer_ang * (-1.))

# add images and angles to the center camera dataset
images.extend(aug_images)
images.extend(images_left)
images.extend(images_right)
measurements.extend(aug_measurements)
measurements.extend(measurements_left)
measurements.extend(measurements_right)    

# Add angles and images to Numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)

n_train = len(X_train)

# set to False when training
debug = False

if debug: 
    print("Total: {}".format(n_train))
    print("Original angle: {}".format(y_train[500]))
    print("Flipped angle:  {}".format(y_train[501]))
    cv2.imshow( "Original", X_train[500] )
    cv2.imshow( "Flipped", X_train[501] )
    cv2.waitKey(0)
    exit()

# Build a simple Keras model
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Input shape for the model
img_shape = X_train.shape[1:]
print(img_shape)
# One output (steering angle) to directly predict the steering angle
num_classes = 1

model = Sequential()

# Image normalization. That lambda layer could take each pixel in an image and run it through the formulas:
# pixel_normalized = pixel / 255
# pixel_mean_centered = pixel_normalized - 0.5
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=img_shape))
# Cropping the hood from bottom and the background from top of the image
model.add(Cropping2D(cropping=((70,25), (0,0))))
# Implement Lenet5 model architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
# CONV->ACT: 6 filters, 5x5 kernel, valid padding and ReLU activation.
model.add(Convolution2D(6, 5, 5, activation='relu'))
# POOL: 2x2 max pooling layer immediately following your convolutional layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# CONV->ACT: 6 filters, 5x5 kernel, valid padding and ReLU activation.
model.add(Convolution2D(6, 5, 5, activation='relu'))
# POOL: 2x2 max pooling layer immediately following your convolutional layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# FLATTEN: 400
model.add(Flatten())
model.add(Dropout(0.5))

# FC: 120 and ReLU activation 
model.add(Dense(120, activation='relu'))

# FC: 84 and ReLU activation
model.add(Dense(84, activation='relu'))

# FC: 1
model.add(Dense(num_classes))
#exit()

n_epoch=6
# Use Adam optimizer and MSE loss function because it is a regression network. 
#The model has to minimize the error between the predicted steering measurements and the true measurements
model.compile(optimizer='adam', loss='mse')
# Split the data for train and validation sets and suffle the data
# Train on 3 epochs to test
model.fit(X_train, y_train, nb_epoch=n_epoch, validation_split=0.2, shuffle=True)

# Save the trained model for the test run with the simulator
model.save("model.h5")  
#exit()
   
    
    
    
    
