
### Sefl-Driving Car Nanodegree Program. Term 1
<img style="float: left;" src="https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg">

## Project 3: Behavioral Cloning

### Overview
This project demostrates usage of deep neural networks and convolutional neural networks to clone driving behavior. The project uses [Keras Deep Learning library](https://keras.io/) to train and test the model. 

The driving data is collected by driving a car around the first track in the [simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip) for Windows. The collected image data and steering angles are used to train a neural network. The output of the model is used to drive the car autonomously around the track.

The project demonstrates usage of two Convolutional Neural Network architectures: [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) and [NVIDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). Both models demonstarated similar training performance on the same dataset.  [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) model was taken for from [CarND-Traffic-Sign-Classifier-Project](https://github.com/ek8203/CarND-Traffic-Sign-Classifier-Project) with minor modifications of the regularisation parameters. It was choosen for final submission because it is lighter, faster and consuming less memory resources.  

The project includes following procedures:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator.
* Summarize the results with a written report

A detail witeup of the projects can be found in [writeup_report.md](writeup_report.md) document.

### Project directory content:

* [README.md](README.md) - This file.
* [model.py](model.py) - The script used to create and train the model.
* [drive.py](drive.py) - The script (original) to drive the car.
* [model.h5](model.h5) - The saved model in HDF5 format.
* [writeup_report.md](writeup_report.md) - The project writeup - a markdown file that explains the structure of the network and training approach.
* [video.mp4](video.mp4) - A video recording of the vehicle driving autonomously one lap around the track.

### Project Environment

The project environment was created with [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit). The model was executed, trained and tested on [AWS](https://aws.amazon.com/) GPU-enabled Linux instance.
