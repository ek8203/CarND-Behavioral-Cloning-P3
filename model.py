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
 
# TODO: read the image frames and the steering angles
def isfloat(value):
  try:
    float(value)
    return True
  except:
    return False

images = []
measurements = []
# the first line is the header - to be excluded
for line in lines[1:]:
    
    # update the dir path of img files
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = datalog_dir +'IMG/' + filename

    # read a frame and add to the list
    image = cv2.imread(current_path)
    
    if isfloat(line[3]):
        steer_ang = float(line[3])
        measurements.append(steer_ang)
        images.append(image)
    else:
        print(line[3])
        print(filename)
        
# Copy angles and images to Numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)

#exit()

# Build a simple Keras model
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Input shape for the model
img_shape = X_train.shape[1:]
# One output (steering angle) to directly predict the steering angle
num_classes = 1

model = Sequential()

# Image normalization. That lambda layer could take each pixel in an image and run it through the formulas:
# pixel_normalized = pixel / 255
# pixel_mean_centered = pixel_normalized - 0.5
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=img_shape))

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
# FC: 120 and ReLU activation 
model.add(Dense(120, activation='relu'))
# FC: 84 and ReLU activation
model.add(Dense(84, activation='relu'))
# FC: 1
model.add(Dense(num_classes))
#exit()

# Use Adam optimizer and MSE loss function because it is a regression network. 
#The model has to minimize the error between the predicted steering measurements and the true measurements
model.compile(optimizer='adam', loss='mse')
# Split the data for train and validation sets and suffle the data
# Train on 3 epochs to test
model.fit(X_train, y_train, nb_epoch=6, validation_split=0.2, shuffle=True)

# Save the trained model for the test run with the simulator
model.save("model.h5")  
#exit()
   
    
    
    
    
