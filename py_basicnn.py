#  _         _                     
# | |_  ___ | |_  __ _  _ __  _ __ 
# | __|/ __|| __|/ _` || '__|| '__|
# | |_ \__ \| |_| (_| || |   | |   
#  \__||___/ \__|\__,_||_|   |_|   
# 
# Description: Implementation of basic 1 hidden layer neural network with
#              regularization. This was completed to complement Andrew Ng's
#              course on machine learning. See example.py for use on handwritten digit
#              dataset from the course.
# Author: Tyler Starr
# Date Created: 21 April 2020
# https://github.com/starr-dusT/*need to add*

# Import needed libraries
import numpy as np
from scipy.optimize import minimize

def sigmoid(z):
    """Compute sigmoid function.
    Parameters
    ----------
    z : np.array (float) or float
        inputs to function

    Returns
    -------
    s : np.array (float) or float
        return sigmoid function of input
    """
    s = 1/(1 + np.exp(-z))
    return s

def dersigmoid(z):
    """Compute dervative of sigmoid function.
    Parameters
    ----------
    z : np.array (float) or float
        inputs to function

    Returns
    -------
    g : np.array (float) or float
        return dervative of gradient
    """
    g = sigmoid(z)*(1-sigmoid(z))
    return g

def nnTrain(nn_params, args, options, method, jac):
    """Train neural network over training data.
    Parameters
    ----------
    nn_params : np.array (float)
        input thetas
    args : vector of arguments
        arguements to pass to trainer
    options : vector of options
        options for trainer
    method : str
        method to train with
    jac : bool
        whether to use jac or not

    Returns
    -------
    results : np.array (float)
        trained thetas
    """
    results = minimize(nnCostFunction, x0=nn_params, args=args, \
              options=options, method=method, jac=jac)
    return results

def nnPredict(Theta1, Theta2, x):
    """Predict outputs given data and trained Thetas.
    Parameters
    ----------
    Theta1 : np.array (float)
        Trained Theta for input layer
    Theta2 : np.array (float)
        Trained Theta for hidden layer

    Returns
    -------
    p+1 : np.array (int)
        predicted bins
    """
    # Change 1D array to 2D
    if x.ndim == 1:
        x = np.reshape(X, (-1,x.shape[0]))
    # Find num training examples and output bins
    m = x.shape[0]
    num_labels = Theta2.shape[0]
    # Compute predicted results
    p = np.zeros((m,1))
    h1 = sigmoid(np.dot(np.column_stack((np.ones((m,1)), x)), Theta1.T))
    h2 = sigmoid(np.dot(np.column_stack((np.ones((m,1)), h1)), Theta2.T))
    p = np.argmax(h2, axis=1)
    return p+1

def nnCostFunction(nn_params, n, hl, k, x, y, reglambda):
    """Compute cost function and gradient for given params.

    Parameters
    ----------
    nn_params : np.array (float)
        neural network theta rolled up
    n : int:
        input layer size
    hl : int
        hidden layer size
    k : int
        output layer size
    x : np.array (float)
        input features of truth data
    y : np.array (float)
        output truth data
    reglambda : float
        regularization lambda parameter

    Returns
    -------
    J : float
        cost of neural network with given params.
    grad : np.array (float)
        gradient of neural network with given params.
    """
    # Get number of training examples
    m = len(y)
    # unroll nn_params into Theta1 (input layer) and Theta2 (hidden layer)
    Theta1 = np.reshape(nn_params[:hl * (n + 1)], (hl, n + 1), order='F')
    Theta2 = np.reshape(nn_params[hl * (n + 1):], (k, hl + 1), order='F')
    # Neural network feed forward
    a1 = np.append(np.ones((m,1)), x, axis=1)
    z2 = a1.dot(np.transpose(Theta1))
    a2 = np.append(np.ones((m,1)), sigmoid(z2), axis=1)
    z3 = a2.dot(np.transpose(Theta2))
    a3 = sigmoid(z3)
    # Convert bin number to vector binary representation
    ymod = np.zeros((m,k))
    for i,row in enumerate(ymod):
        ymod[i,:] = (np.linspace(1,k,k) == y[i])
    # Find cost without regularization
    J = (-1/m)*np.sum(ymod*np.log(a3) + (1-ymod)*np.log(1-a3))
    # Add in regularization
    regTheta1 = np.square(Theta1[:,1:])
    regTheta2 = np.square(Theta2[:,1:])
    J = J + (reglambda/(2*m))*np.sum(np.sum(regTheta1) + np.sum(regTheta2))
    # Initalize gradient terms to zero
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    # Loop over all training data
    for i in range(m):
        # Get needed paramters from feed forward for on training example
        a3i = a3[i,:]
        a2i = a2[i,:]
        a1i = a1[i,:]
        ymodi = ymod[i,:]
        z2i = np.append(1, z2[i,:])
        # compute delta3 and delta2
        delta3 = a3i - ymodi
        delta2 = (np.transpose(Theta2).dot(delta3))*dersigmoid(z2i)
        # sum delta terms up to create Delta
        Theta2_grad = Theta2_grad + np.outer(delta3, a2i)
        Theta1_grad = Theta1_grad + np.outer(delta2[1:], a1i)
    # Divide by m to complete non-regularized gradient
    Theta2_grad = (1/m)*Theta2_grad
    Theta1_grad = (1/m)*Theta1_grad
    # Add in regularization to gradient
    Theta2_grad[:,1:] = Theta2_grad[:,1:] + (reglambda/m)*Theta2[:,1:]
    Theta1_grad[:,1:] = Theta1_grad[:,1:] + (reglambda/m)*Theta1[:,1:]
    # Roll gradient up before return
    grad = np.concatenate((Theta1_grad.reshape(Theta1_grad.size, order='F'), Theta2_grad.reshape(Theta2_grad.size, order='F')))
    # Return cost and gradient given input params
    return J, grad
