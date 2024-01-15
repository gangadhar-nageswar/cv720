import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from skimage import data, io
from matplotlib import pyplot as plt

from nn import *
from q4 import *

import cv2


# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    # im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',"04_deep.jpg")))
    
    
    bboxes, bw = findLetters(im1)
    bw_invert = 1-bw

    plt.imshow(bw_invert, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()


    # find the rows using..RANSAC, counting, clustering, etc.

    pad = 100
    num_boxes = len(bboxes)
    rows = []

    bboxes_arr = np.array(bboxes)
    row_sorted_indices = np.argsort(bboxes_arr[:, 0])
    bbs = bboxes_arr[row_sorted_indices]

    while num_boxes > 0:
        ymin = bbs[0,0]
        col_indices = np.where((bbs[:, 0] >= ymin) & (bbs[:, 0] <= ymin + pad))[0]
        rem_indices = np.where((bbs[:, 0] < ymin) | (bbs[:, 0] > ymin + pad))[0]

        col_bboxes = bbs[col_indices]
        rem_bboxes = bbs[rem_indices]

        col_sorted_indices = np.argsort(col_bboxes[:, 1])
        sorted_col_bbs = col_bboxes[col_sorted_indices]
        
        rows.append(sorted_col_bbs)

        row_sorted_indices = np.argsort(rem_bboxes[:, 0])
        bbs = rem_bboxes[row_sorted_indices]

        num_boxes = bbs.shape[0]



    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))


    for row in rows:

        if img == "02_letters.jpg":
            horizontal_pad = 10
        else:
            hpads = []
            for ind_col in range(1,row.shape[0]):
                hpads.append(row[ind_col][1] - row[ind_col-1][1])
            
            hpads = np.array(hpads)
            hspace_median = np.median(hpads)
            hspace_eps = np.std(hpads)

            horizontal_pad = hspace_median + 1.3*hspace_eps


        prev_minc = row[0][1]

        for ind_col in range(row.shape[0]):
            ind_letter = row[ind_col]
            minr, minc, maxr, maxc = ind_letter[0], ind_letter[1], ind_letter[2], ind_letter[3]
            test_img = bw[minr-5:maxr+5, minc-5:maxc+5]

            if abs(minc-prev_minc) > horizontal_pad:
                print(" ", end="")
                
            
            prev_minc = minc

            test_img = np.pad(test_img, 30, 'constant', constant_values=0.0)
            # plt.imshow(test_img, cmap='gray')
            # plt.show()

            test_img = skimage.transform.resize(test_img, (32, 32))
            # test_img = skimage.morphology.erosion(test_img, skimage.morphology.square(1))

            # plt.imshow(test_img, cmap='gray')
            # plt.show()

            # test_img = 1-test_img
            test_img = skimage.util.invert(test_img)


            test_img = test_img.transpose()
            test_x = test_img.flatten()
            test_x = np.reshape(test_x, (1024,1))
            test_x = test_x.transpose()

            # test_x = 1-test_x

            layer1_output = forward(test_x,params,'layer1',sigmoid)
            probs = forward(layer1_output,params,'output',softmax)

            # obtain y_pred from the probs
            test_y_pred = np.zeros_like(probs)
            pred_ind = np.argmax(probs, axis=1)

            pred_letter = letters[pred_ind]

            print(pred_letter[0], end="")

            # fig, ax = plt.subplots()
            # ax.imshow(test_img.transpose())
            # ax.set_title(pred_letter)
            # plt.show()
        
        print("\n", end="")

    print("\n\n", end="")
















    '''
    Some extra functions and code snippets
    
    # Visualise the detected boxes:
    
    for row in rows:
        print("new row -- ")
        for ind_col in range(row.shape[0]):
            plt.imshow(bw)
            ind_letter = row[ind_col]
            minr, minc, maxr, maxc = ind_letter[0], ind_letter[1], ind_letter[2], ind_letter[3]
            let_img = bw[minr:maxr, minc:maxc]
            rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.show()
    
    
    

    for row in rows:
        for ind_col in range(row.shape[0]):
            ind_letter = row[ind_col]
            minr, minc, maxr, maxc = ind_letter[0], ind_letter[1], ind_letter[2], ind_letter[3]
            let_img = bw[minr:maxr, minc:maxc]

            let_img = np.pad(let_img, ((20, 20), (20, 20))

            plt.imshow(let_img)
            plt.show()

    

    for bbox in bboxes[:20]:
        test_coords = bbox
        test_img = bw[test_coords[0]-5:test_coords[2]+5, test_coords[1]-5:test_coords[3]+5]

        test_img = np.pad(test_img, 29, 'constant', constant_values=0.0)
        plt.imshow(test_img, cmap='gray')
        plt.show()

        test_img = skimage.transform.resize(test_img, (32, 32))
        # test_img = skimage.morphology.erosion(test_img, skimage.morphology.square(1))

        plt.imshow(test_img, cmap='gray')
        plt.show()

        # test_img = 1-test_img
        test_img = skimage.util.invert(test_img)


        test_img = test_img.transpose()
        test_x = test_img.flatten()
        test_x = np.reshape(test_x, (1024,1))
        test_x = test_x.transpose()

        # test_x = 1-test_x

        layer1_output = forward(test_x,params,'layer1',sigmoid)
        probs = forward(layer1_output,params,'output',softmax)

        # obtain y_pred from the probs
        test_y_pred = np.zeros_like(probs)
        pred_ind = np.argmax(probs, axis=1)

        pred_letter = letters[pred_ind]

        fig, ax = plt.subplots()
        ax.imshow(test_img.transpose())
        ax.set_title(pred_letter)
        plt.show()
    
    
    
    '''