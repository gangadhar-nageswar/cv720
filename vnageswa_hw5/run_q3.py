from cgi import test
from warnings import filters
import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 36
learning_rate = 1e-03
hidden_size = 64
##########################
##### your code here #####
##########################

print(f"train size={train_x.shape[1]}")
print(f"train size={train_x.shape[0]}")

input_size = train_x.shape[1]
output_size = train_y.shape[1]

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################
##### your code here #####
##########################
initialize_weights(input_size,hidden_size,params,'layer1')
initialize_weights(hidden_size,output_size,params,'output')

# first_act = relu
# first_act_deriv = relu_deriv

first_act = sigmoid
first_act_deriv = sigmoid_deriv

train_loss_history = []
train_acc_history = []

valid_loss_history = []
valid_acc_history = []

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0

    for xb,yb in batches:


        ### testing ###

        if False: # view the data
            for crop in xb:
                import matplotlib.pyplot as plt
                plt.imshow(crop.reshape(32,32).T, cmap='gray')
                plt.show()





        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################
        # forward
        layer1_output = forward(xb,params,'layer1',first_act)
        probs = forward(layer1_output,params,'output',softmax)


        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        avg_acc += acc

        # backward
        delta1 = probs
        # print(f"before delta1:{delta1.shape} || yb: {yb.shape}")
        delta1[np.arange(probs.shape[0]), yb.argmax(axis=1)] -= 1
        # delta1 -= yb
        # print(f"after --> delta1 shape: {delta1.shape}")

        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',first_act_deriv)

        # apply gradient
        params['Woutput'] -= learning_rate*params['grad_Woutput']
        params['Wlayer1'] -= learning_rate*params['grad_Wlayer1']

        params['blayer1'] -= learning_rate*params['grad_blayer1']
        params['boutput'] -= learning_rate*params['grad_boutput']

    avg_acc /= batch_num
    total_loss /= batch_num*xb.shape[0]

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))
    
    train_loss_history.append(total_loss)
    train_acc_history.append(avg_acc)

    # get validation loss and accuracy
    valid_layer1_output = forward(valid_x,params,'layer1',first_act)
    valid_probs = forward(valid_layer1_output,params,'output',softmax)

    valid_loss, valid_acc = compute_loss_and_acc(valid_y, valid_probs)

    valid_loss /= valid_y.shape[0]

    valid_loss_history.append(valid_loss)
    valid_acc_history.append(valid_acc)



# run on validation set and report accuracy! should be above 75%
# valid_acc = None

##########################
##### your code here #####
##########################


# plots for 3.1
import matplotlib.pyplot as plt

epoch_history = [i for i in range(max_iters)]

print(f"epoch history: {len(epoch_history)}")
print(f"train acc history: {len(train_acc_history)}")

fig, ax = plt.subplots()

ax.plot(epoch_history, train_acc_history, label='train acc', color='blue')
ax.plot(epoch_history, valid_acc_history, label='valid acc', color='red')

ax.set_xlabel('epochs')
ax.set_ylabel('accuracy')
ax.legend()

plt.show()


# loss vs epochs
fig, ax = plt.subplots()

ax.plot(epoch_history, train_loss_history, label='train loss', color='blue')
ax.plot(epoch_history, valid_loss_history, label='valid loss', color='red')

ax.set_xlabel('epochs')
ax.set_ylabel('loss')
ax.legend()

plt.show()


if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T, cmap='gray')
        plt.show()


import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
##########################
##### your code here #####
##########################


# Reshape each column into 32x32 matrices
num_filters = params['Wlayer1'].shape[1]

weight_filters = [params['Wlayer1'][:, i].reshape(32, 32) for i in range(num_filters)]

fig = plt.figure(figsize=(8, 8))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(int(np.ceil(np.sqrt(num_filters))), int(np.ceil(np.sqrt(num_filters)))),  # Grid size
                    axes_pad=0.1,  # Pad between images
                    )

# Add images to the grid
for ax, img in zip(grid, weight_filters):
    ax.imshow(img)
    ax.axis('off')

plt.show()



########### visualise the final weights ##########
# Reshape each column into 32x32 matrices
num_filters = params['Woutput'].shape[1]
filter_size = int(np.sqrt(params['Woutput'].shape[0]))

weight_filters = [params['Woutput'][:, i].reshape(filter_size,filter_size) for i in range(num_filters)]

fig = plt.figure(figsize=(np.sqrt(num_filters), np.sqrt(num_filters)))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(int(np.ceil(np.sqrt(num_filters))), int(np.ceil(np.sqrt(num_filters)))),  # Grid size
                    axes_pad=0.1,  # Pad between images
                    )

# Add images to the grid
for ax, img in zip(grid, weight_filters):
    ax.imshow(img)
    ax.axis('off')

plt.show()




# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
##########################
##### your code here #####
##########################

layer1_output = forward(test_x,params,'layer1',first_act)
probs = forward(layer1_output,params,'output',softmax)

# obtain y_pred from the probs
test_y_pred = np.zeros_like(probs)
max_indices = np.argmax(probs, axis=1)
# true_max_indices = np.argmax(test_y, axis=1)
test_y_pred[np.arange(probs.shape[0]), max_indices] = 1

for ind in range(test_y_pred.shape[0]):
    y_pred_ind = np.argmax(test_y_pred[ind,:])
    y_true_ind = np.argmax(test_y[ind,:])
    
    confusion_matrix[y_pred_ind, y_true_ind] += 1


import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()