from curses import REPORT_MOUSE_POSITION
from distutils.log import Log
import os, multiprocessing
from os.path import join, isfile
from random import random
# from socket import ALG_OP_DECRYPT
from tkinter import image_names

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
from sklearn.cluster import KMeans
from scipy.spatial import distance


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : optionș
    * img     : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    filter_scales = opts.filter_scales

    # ----- TODO -----
    H, W, C = img.shape
    F = 4
    Ns = len(filter_scales)
    
    response_map = np.ndarray(shape=(H,W,Ns*F*C))
    
    # convert the grayscale images to 3 channels
    if len(img.shape) < 3:
        img = np.stack([img,img,img], axis=0)
        img = np.moveaxis(img, 0, -1)
    
    lab_img = skimage.color.rgb2lab(img)
    
    for i in range(Ns):
        for j in range(C):
            response_map[:,:, i*F*C + j]      = scipy.ndimage.gaussian_filter(lab_img[:,:,j], sigma=filter_scales[i])
            response_map[:,:, i*F*C + j + C]  = scipy.ndimage.gaussian_filter(lab_img[:,:,j], sigma=filter_scales[i], order=2)
            response_map[:,:, i*F*C + j + 2*C]  = scipy.ndimage.gaussian_filter(lab_img[:,:,j], sigma=[filter_scales[i],0], order=1)
            response_map[:,:, i*F*C + j + 3*C]  = scipy.ndimage.gaussian_filter(lab_img[:,:,j], sigma=[0,filter_scales[i]], order=1)

    return response_map
    
    
def compute_dictionary_one_image(opts, data_dir, img_path):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----

    alpha = opts.alpha
    feat_dir = opts.feat_dir

    cat_name = img_path.split('/')[0]
    img_name = img_path.split('/')[1].split('.')[0]

    img = Image.open(join(opts.data_dir, img_path))
    img = np.array(img).astype(np.float32)/255

    H,W,C = img.shape
    img_responses = extract_filter_responses(opts, img)

    x_rand_inds = np.random.randint(low=0, high=H, size=alpha)
    y_rand_inds = np.random.randint(low=0, high=W, size=alpha)
    
    for k in range(alpha):
        pix_response = img_responses[x_rand_inds[k], y_rand_inds[k], :]
        np.save(join(feat_dir, f'{cat_name}_{img_name}_{k}_filter_response.npy'), pix_response)
    
    return None


def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----
    
    # for img_path in train_files:
    #     compute_dictionary_one_image(opts, data_dir, img_path)

    img_args = [(opts, data_dir, file) for file in train_files]
    pool = multiprocessing.Pool(n_worker)
    responses = pool.starmap(compute_dictionary_one_image, img_args)

    response_paths = os.listdir(feat_dir)
    if '.DS_Store' in response_paths:
        response_paths.remove('.DS_Store')

    filter_responses = []
    for res_file_name in response_paths:
        res = np.load(join(feat_dir, res_file_name), allow_pickle=True)
        filter_responses.append(res)
    
    # filter_responses = np.array(filter_responses)
        
    
    kmeans = KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_

    np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    H, W, C = img.shape
    img_filter_response = extract_filter_responses(opts, img)
    wordmap = np.ndarray(shape=(H,W))

    for i in range(H):
        for j in range(W):
            pix_response = np.expand_dims(img_filter_response[i,j,:], axis=0)
            dist_mat = distance.cdist(pix_response, dictionary, 'cosine')

            pix_cat = np.argmin(dist_mat)
            
            wordmap[i][j] = pix_cat

    return wordmap


    

