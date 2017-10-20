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


# Build a simple Keras model
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense

# Input shape for the model
img_shape = X_train.shape[1:]
# One output (steering angle) to directly predict the steering angle
num_classes = 1

model = Sequential()

model.add(Flatten(input_shape=img_shape))
model.add(Dense(num_classes))

# Use Adam optimizer and MSE loss function because it is a regression network. 
#The model has to minimize the error between the predicted steering measurements and the true measurements
model.compile(optimizer='adam', loss='mse')
# Split the data for train and validation sets and suffle the data
# Train on 3 epochs to test
model.fit(X_train, y_train, nb_epoch=6, validation_split=0.2, shuffle=True)

# Save the trained model for the test run with the simulator
model.save("model.h5")  
   
    
    
    
    
