import numpy as np

# Sigmoud Function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def dersigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    g = g*(1-g)
    return g


# input layer size, hidden layer size, input, output, reglambda, theta(rolled)
def nnCostFunction(nn_params, n, hl, k, x, y, reglambda):
    m = len(y)
    Theta1 = np.reshape(nn_params[:hl * (n + 1)], (hl, n + 1), order='F')
    Theta2 = np.reshape(nn_params[hl * (n + 1):], (k, hl + 1), order='F')
    
    # Feed Forward
    a1 = np.append(np.ones((m,1)), x, axis=1)
    z2 = a1.dot(np.transpose(Theta1))
    a2 = np.append(np.ones((m,1)), sigmoid(z2), axis=1)
    z3 = a2.dot(np.transpose(Theta2))
    a3 = sigmoid(z3)

    # Bin number to vector form
    ymod = np.zeros((m,k))
    for i,row in enumerate(ymod):
        ymod[i,:] = (np.linspace(1,k,k) == y[i])
    
    # Find cost without regularization
    J = (-1/m)*np.sum(ymod*np.log(a3) + (1-ymod)*np.log(1-a3))

    # Add in regularization
    regTheta1 = np.square(Theta1[:,1:])
    regTheta2 = np.square(Theta2[:,1:])
    J = J + (reglambda/(2*m))*np.sum(np.sum(regTheta1) + np.sum(regTheta2))
    
    # Implement Gradient
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    for i in range(m):
        a3i = a3[i,:]
        a2i = a2[i,:]
        a1i = a1[i,:]
        ymodi = ymod[i,:]
        z2i = np.append(1, z2[i,:])
       
        delta3 = a3i - ymodi
        delta2 = (np.transpose(Theta2).dot(delta3))*dersigmoid(z2i)

        delta3_re = delta3.reshape(delta3.shape[0],1)
        z2i_re = z2i.reshape(z2i.shape[0],1)
        Theta2_grad = Theta2_grad + np.outer(delta3, a2i)

        delta2_re = delta2.reshape(delta2.shape[0],1)
        a1i_re = a1i.reshape(a1i.shape[0],1)
        Theta1_grad = Theta1_grad + np.outer(delta2[1:], a1i)

    Theta2_grad = (1/m)*Theta2_grad
    Theta1_grad = (1/m)*Theta1_grad
    Theta2_grad[:,1:] = Theta2_grad[:,1:] + (reglambda/m)*Theta2[:,1:]
    Theta1_grad[:,1:] = Theta1_grad[:,1:] + (reglambda/m)*Theta1[:,1:]
    
    grad = np.concatenate((Theta1_grad.reshape(Theta1_grad.size, order='F'), Theta2_grad.reshape(Theta2_grad.size, order='F')))

    return J, grad