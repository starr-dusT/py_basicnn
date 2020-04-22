#  _         _                     
# | |_  ___ | |_  __ _  _ __  _ __ 
# | __|/ __|| __|/ _` || '__|| '__|
# | |_ \__ \| |_| (_| || |   | |   
#  \__||___/ \__|\__,_||_|   |_|   
# 
# Description: Example neural network training with hand written digits
#              from dataset in Andrew Ng's Coursera machine learning course.
# Author: Tyler Starr
# Date Created: 21 April 2020
# https://github.com/starr-dusT/*need to add*

# Import needed libraries
from scipy.io import loadmat
import py_basicnn as nn
import numpy as np
import random

# Import dataset
dat = loadmat('exdata/ex4data1.mat')
# Import input features 
x = dat['X']
# Import truth outputs
y = dat['y']

# Randomly shuffle x,y while making sure rows match
rand_seed = random.randint(0, 5000)
np.random.seed(rand_seed)
np.random.shuffle(x)
np.random.seed(rand_seed)
np.random.shuffle(y)

# Split data into training and verification sections
x_train, x_ver = np.split(x,[int(.8*len(x))])
y_train, y_ver = np.split(y,[int(.8*len(y))])

# Get number of input features from input data
n = x.shape[1]
# Define the desired number units in hidden layer
hl = 25
# Define number of bins in output
k = 10
# Define regularization parameter
lambda_reg = 0.5

# Initalize thetas to random values
Theta1 = np.random.rand(n+1,hl)
Theta2 = np.random.rand(hl+1,k)
# Roll up for passing to functions
nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F')))

# Define arguements and options for training function
myargs = (n, hl, k, x_train, y_train, lambda_reg)
options = {'disp': True, 'maxiter':100}
# Train neural network
results = nn.nnTrain(nn_params, myargs, options, method="L-BFGS-B", jac=True)

# Extract parameters from results
nn_params_trained = results["x"]
# Unroll parameters for pass to prediction function
Theta1 = np.reshape(nn_params_trained[:hl * (n + 1)], (hl, n + 1), order='F')
Theta2 = np.reshape(nn_params_trained[hl * (n + 1):], (k, hl + 1), order='F')

# Predict outputs for verification portion of dataset
pred = nn.nnPredict(Theta1, Theta2, x_ver)
# Print the percentage which are correct
print('Training Set Accuracy: {:f}'.format((np.mean(pred == y_ver.T)*100)))
