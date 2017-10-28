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

# Use csv lib to read the measurements from the driving_log.csv file
datalog_dir = "data/"
lines = []
with open(datalog_dir + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print("loaded")
        
def read_image(src_path):
    """
    Read the image frames
    """    
    # update the dir path of img files
    filename = src_path.split('\\')[-1]
    current_path = datalog_dir +'IMG/' + filename    
    # read a frame and add to the list
    image = cv2.imread(current_path)
    # convert BGR to RGB since the test simulator (drive.py) is using RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resize
    #image = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
    return image
    
import sklearn
def generator(samples, correction = 0.2, batch_size=32):
    """
    Input batch data generator
    """
    num_samples = (len(samples)//batch_size)*batch_size
    
    while True: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        
        #X_batch = np.zeros((batch_size, 160, 320, 3), dtype=np.float32)
        #y_batch = np.zeros((batch_size,), dtype=np.float32)
        
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
                image = read_image(line[0])
                images.append(image)    
                angles.append(steering_center)
                # add augmented images - flipped vertically (flipCode = 1)
                images.append(cv2.flip(image, 1))
                # reverse the angles
                angles.append(steering_center * (-1.))

                # left camera image
                image = read_image(line[1])
                images.append(image)    
                angles.append(steering_left)        
                # add augmented images - flipped vertically (flipCode = 1)
                images.append(cv2.flip(image, 1))
                # reverse the angles
                angles.append(steering_left * (-1.))

                # right camera image
                image = read_image(line[2])
                images.append(image)    
                angles.append(steering_right)        
                # add augmented images - flipped vertically (flipCode = 1)
                images.append(cv2.flip(image, 1))
                # reverse the angles
                angles.append(steering_right * (-1.))

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

   
# compile and train the model using the generator function
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)

# test the generator
for i in range(3):
    X,y = next(train_generator)
    print(X.shape, y.shape)
print("done")
#exit()
            

validation_generator = generator(validation_samples, batch_size=32)

print(len(train_samples), len(validation_samples))

"""

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
"""

# Build a simple Keras model
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Dropout, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

shape = (160,320,3)
#shape = (32,32,3)

def resize_image(input):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (64, 64))


def InputNormalized(shape):
    """
    Model input and normalization layers.
    """
    #Sequential(shape=(batch_size, height, width, channels))
    model = Sequential()
    # Cropping the hood from bottom and the background from top of the image    
    model.add(Cropping2D(cropping=((50,25), (0,0)), input_shape=shape))
    model.add(Lambda(resize_image))
    # Image normalization. That lambda layer could take each pixel in an image and run it through the formulas:
    # pixel_normalized = pixel / 255
    # pixel_mean_centered = pixel_normalized - 0.5
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))#, input_shape=shape))
    return model
    
def LeNet(shape=(160,320,3), keep_prob=1.0):
    """
    LeNet-5 model architecture:
    INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
    """
    # INPUT:
    model = InputNormalized(shape)
    # CONV->ACT: 6 filters, 5x5 kernel, valid padding and ReLU activation.
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    # POOL: 2x2 max pooling layer immediately following your convolutional layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # CONV->ACT: 16 filters, 5x5 kernel, valid padding and ReLU activation.
    model.add(Convolution2D(16, 5, 5, activation='relu'))
    # POOL: 2x2 max pooling layer immediately following your convolutional layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # FLATTEN: 400
    model.add(Flatten())
    model.add(Dropout(keep_prob))
    # FC: 120 and ReLU activation 
    model.add(Dense(120, activation='relu'))
    # FC: 84 and ReLU activation
    model.add(Dense(84, activation='relu'))
    # FC: 1
    model.add(Dense(1))
    return model
    
def nVidia(shape=(160,320,3)):
    """
    nVidea model architecture:
    INPUT -> 3x(CONV -> ACT) -> 2x(CONV -> ACT)-> FLATTEN -> 4xFC
    """
    model = InputNormalized(shape)
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

#exit()

#model = LeNet(shape)
model = nVidia()

n_epoch=6
# Use Adam optimizer and MSE loss function because it is a regression network. 
#The model has to minimize the error between the predicted steering measurements and the true measurements
model.compile(optimizer='adam', loss='mse')
# Split the data for train and validation sets and suffle the data
#model.fit(X_train, y_train, nb_epoch=n_epoch, validation_split=0.2, shuffle=True)

history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), 
                    validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=n_epoch)

#history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)/batch_size,
#                        validation_data=validation_generator, nb_val_samples=len(validation_samples),nb_epoch=7)

#print(model.summary())
                    
#print("done")
#exit()
            
# Save the trained model for the test run with the simulator
model.save("model.h5")  
#exit()
   
    
    
    
    
