from turtle import shape
import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    ##########################
    b = np.zeros(shape=(out_size,))
    W = np.random.uniform(low=-np.sqrt(6/(in_size + out_size)), high=np.sqrt(6/(in_size + out_size)), size=in_size*out_size)
    W = np.reshape(W, (in_size, out_size))

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################
    # x = x.astype(np.float64)
    res = 1/(1+np.exp(-x))

    # # optimise the below code later
    # if x > 0:
    #     res = 1/(1+np.exp(-x))
    # elif x < 0:
    #     res = np.exp(x)/(1+np.exp(x))

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]


    ##########################
    ##### your code here #####
    ##########################

    pre_act = np.matmul(X,W) + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################

    num = np.exp(x-np.max(x, axis=1)[:, np.newaxis])
    # num = np.exp(x)
    res = num/num.sum(1)[:, np.newaxis]

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    ##########################

    loss = -np.multiply(y,np.log(probs)).sum()

    # obtain y_pred from the probs
    y_pred = np.zeros_like(probs)
    max_indices = np.argmax(probs, axis=1)
    y_pred[np.arange(probs.shape[0]), max_indices] = 1

    acc = np.sum(np.multiply(y,y_pred))/y.shape[0]

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    ##########################
    ##### your code here #####
    ##########################

    delta = delta*activation_deriv(post_act)

    grad_W = np.matmul(np.transpose(X), delta)
    grad_b = delta.sum(0)
    grad_X = np.matmul(delta, np.transpose(W))

    # print(f"delta: {delta.shape} || X: {X.shape} || activation_deriv(pre_act): {activation_deriv(pre_act).shape}")

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b

    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    ##########################

    N = x.shape[0]
    num_batches = N//batch_size

    shuffled_inds = np.arange(x.shape[0])
    np.random.shuffle(shuffled_inds)

    x_shuffled = x[shuffled_inds]
    y_shuffled = y[shuffled_inds]


    for i in range(num_batches):
        x_batch = x_shuffled[i*batch_size: (i+1)*batch_size,:]
        y_batch = y_shuffled[i*batch_size: (i+1)*batch_size,:]
        batches.append((x_batch, y_batch))


    return batches
