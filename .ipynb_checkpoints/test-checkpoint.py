from scipy.io import loadmat
from scipy.optimize import minimize
import py_nn as nn
import numpy as np
import checkNNGradients as cnng
import predict as pr

m = 5000
n = 400
hl = 25
k = 10

dat = loadmat('ex4data1.mat')
weight = loadmat('ex4weights.mat')
# Fill inputs with random data for inital test (with one training example)
x = dat['X']
# Example training output
y = dat['y']
# Fill thetas with random numbers
Theta1 = np.random.rand(n+1,hl)
Theta2 = np.random.rand(hl+1,k)
#Theta1 = weight['Theta1']
#Theta2 = weight['Theta2']
# Add regularization term
nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F')))
maxiter = 1000
lambda_reg = 0.1
myargs = (n, hl, k, x, y, lambda_reg)
results = minimize(nn.nnCostFunction, x0=nn_params, args=myargs, options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)

nn_params_t = results["x"]


Theta1_t = np.reshape(nn_params_t[:hl * (n + 1)], (hl, n + 1), order='F')
Theta2_t = np.reshape(nn_params_t[hl * (n + 1):], (k, hl + 1), order='F')

pred = pr.predict(Theta1_t, Theta2_t, x)
print('Training Set Accuracy: {:f}'.format( ( np.mean(pred == y.T)*100 ) ) )
