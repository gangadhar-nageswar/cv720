import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

train_loss_history = []
train_acc_history = []

valid_loss_history = []
valid_acc_history = []

train_model = True
save_model = True


def update_momentum(params, learning_rate):
    params['m_Wlayer1'] = 0.9*params['m_Wlayer1'] - learning_rate*params['grad_Wlayer1']
    params['m_Wlayer2'] = 0.9*params['m_Wlayer2'] - learning_rate*params['grad_Wlayer2']
    params['m_Wlayer3'] = 0.9*params['m_Wlayer3'] - learning_rate*params['grad_Wlayer3']
    params['m_Woutput'] = 0.9*params['m_Woutput'] - learning_rate*params['grad_Woutput']
    params['m_blayer1'] = 0.9*params['m_blayer1'] - learning_rate*params['grad_blayer1']
    params['m_blayer2'] = 0.9*params['m_blayer2'] - learning_rate*params['grad_blayer2']
    params['m_blayer3'] = 0.9*params['m_blayer3'] - learning_rate*params['grad_blayer3']
    params['m_boutput'] = 0.9*params['m_boutput'] - learning_rate*params['grad_boutput']


def update_weights(params):
    params['Wlayer1'] += params['m_Wlayer1']
    params['Wlayer2'] += params['m_Wlayer2']
    params['Wlayer3'] += params['m_Wlayer3']
    params['Woutput'] += params['m_Woutput']
    params['blayer1'] += params['m_blayer1']
    params['blayer2'] += params['m_blayer2']
    params['blayer3'] += params['m_blayer3']
    params['boutput'] += params['m_boutput']

def update_weights_GD(params, learning_rate):

    params['Wlayer1'] -= learning_rate*params['grad_Wlayer1']
    params['Wlayer2'] -= learning_rate*params['grad_Wlayer2']
    params['Wlayer3'] -= learning_rate*params['grad_Wlayer3']
    params['Woutput'] -= learning_rate*params['grad_Woutput']
    params['blayer1'] -= learning_rate*params['grad_blayer1']
    params['blayer2'] -= learning_rate*params['grad_blayer2']
    params['blayer3'] -= learning_rate*params['grad_blayer3']
    params['boutput'] -= learning_rate*params['grad_boutput']


def model_forward(test_x, params):
    test_x = np.expand_dims(test_x, axis=1).T
    test_img = np.reshape(test_x, (32,32))
    plt.imshow(test_img.T)
    plt.show()

    layer1_output = forward(test_x,params,'layer1',relu)
    layer2_output = forward(layer1_output,params,'layer2',relu)
    layer3_output = forward(layer2_output,params,'layer3',relu)
    recons_x = forward(layer3_output,params,'output',sigmoid)

    recons_img = np.reshape(recons_x, (32,32)).T

    plt.imshow(recons_img)
    plt.show()

    return recons_img




if train_model == True:
    params = Counter()
    # params = {}

    # Q5.1 & Q5.2
    # initialize layers here
    ##########################
    ##### your code here #####
    ##########################

    layers = [1024, 32, 32, 32, 1024]
    acts = [relu, relu, relu, sigmoid]

    initialize_weights(layers[0],layers[1],params,'layer1')
    initialize_weights(layers[1],layers[2],params,'layer2')
    initialize_weights(layers[2],layers[3],params,'layer3')
    initialize_weights(layers[3],layers[4],params,'output')

    # initialise all the momentums to be zero
    # params['m_Wlayer1'] = np.zeros_like(params['Wlayer1'])
    # params['m_Wlayer2'] = np.zeros_like(params['Wlayer2'])
    # params['m_Wlayer3'] = np.zeros_like(params['Wlayer3'])
    # params['m_Woutput'] = np.zeros_like(params['Woutput'])

    # params['m_blayer1'] = np.zeros_like(params['blayer1'])
    # params['m_blayer2'] = np.zeros_like(params['blayer2'])
    # params['m_blayer3'] = np.zeros_like(params['blayer3'])
    # params['m_boutput'] = np.zeros_like(params['boutput'])


    # should look like your previous training loops
    for itr in range(max_iters):
        total_loss = 0
        for xb,_ in batches:
            # training loop can be exactly the same as q2!
            # your loss is now squared error
            # delta is the d/dx of (x-y)^2
            # to implement momentum
            #   just use 'm_'+name variables
            #   to keep a saved value over timestamps
            #   params is a Counter(), which returns a 0 if an element is missing
            #   so you should be able to write your loop without any special conditions

            ##########################
            ##### your code here #####
            ##########################
            layer1_output = forward(xb,params,'layer1',relu)
            layer2_output = forward(layer1_output,params,'layer2',relu)
            layer3_output = forward(layer2_output,params,'layer3',relu)
            recons_img = forward(layer3_output,params,'output',sigmoid)

            # print(f"shape of loss : {(xb-recons_img).shape}")

            loss = ((xb-recons_img)**2).sum()
            total_loss += loss

            delta0 = 2*(recons_img-xb)
            delta1 = backwards(delta0,params,'output',sigmoid_deriv)
            delta2 = backwards(delta1,params,'layer3',relu_deriv)
            delta3 = backwards(delta2,params,'layer2',relu_deriv)
            backwards(delta3,params,'layer1',relu_deriv)

            # update_weights_GD(params, learning_rate)

            update_momentum(params, learning_rate)
            update_weights(params)

        train_loss_history.append(total_loss)

        layer1_output = forward(valid_x,params,'layer1',relu)
        layer2_output = forward(layer1_output,params,'layer2',relu)
        layer3_output = forward(layer2_output,params,'layer3',relu)
        valid_recons_img = forward(layer3_output,params,'output',sigmoid)

        # print(f"shape of loss : {(xb-recons_img).shape}")

        valid_loss = ((valid_x-valid_recons_img)**2).sum()
        valid_loss_history.append(valid_loss)

        if itr % 2 == 0:
            print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))

        if itr % lr_rate == lr_rate-1:
            learning_rate *= 0.9
            

    # visualize training loss
    import matplotlib.pyplot as plt

    epoch_history = [i for i in range(max_iters)]

    fig, ax = plt.subplots()
    ax.plot(epoch_history, train_loss_history, label='train acc', color='blue')

    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.legend()

    plt.show()

    epoch_history = [i for i in range(max_iters)]

    fig, ax = plt.subplots()
    ax.plot(epoch_history, valid_loss_history, label='train acc', color='blue')

    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.legend()

    plt.show()


    import pickle
    if save_model == True:
        saved_params = {k:v for k,v in params.items() if '_' not in k}
        with open('q5_weights.pickle', 'wb') as handle:
            pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)





# load weights
import pickle
params = pickle.load(open('q5_weights.pickle','rb'))


# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
##########################
##### your code here #####
##########################

test_x = valid_x[510,:]
recons_img = model_forward(test_x, params)

# test_x = np.expand_dims(test_x, axis=1).T
# test_img = np.reshape(test_x, (32,32))
# plt.imshow(test_img.T)
# plt.show()

# layer1_output = forward(test_x,params,'layer1',relu)
# layer2_output = forward(layer1_output,params,'layer2',relu)
# layer3_output = forward(layer2_output,params,'layer3',relu)
# recons_x = forward(layer3_output,params,'output',sigmoid)

# recons_img = np.reshape(recons_x, (32,32))

# plt.imshow(recons_img)
# plt.show()


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
##########################
##### your code here #####
##########################

layer1_output = forward(valid_x,params,'layer1',relu)
layer2_output = forward(layer1_output,params,'layer2',relu)
layer3_output = forward(layer2_output,params,'layer3',relu)
recons_valid_x = forward(layer3_output,params,'output',sigmoid)

# psnr_matrix = 20*np.log10(np.max(recons_valid_x, axis=1)) - 10*np.log10(np.mean(recons_valid_x, axis=1))
# print(f"psnr_matrix: {np.mean(psnr_matrix)}")


psnr_value = 0
for i in range(valid_x.shape[0]):
    psnr_value += peak_signal_noise_ratio(recons_valid_x[i], valid_x[i])

psnr_value /= valid_x.shape[0]

print(psnr_value)