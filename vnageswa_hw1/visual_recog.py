from ctypes import resize
import os, math, multiprocessing
from os.path import join
from copy import copy
from pyexpat import features
from re import I
from time import sleep
import time

import numpy as np
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    H, W = wordmap.shape

    # ----- TODO -----

    reshaped_img = wordmap.reshape(H*W)
    hist, bin_edges = np.histogram(reshaped_img, bins=K, range=(0,K), density=True)

    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
    
    K = opts.K
    L = opts.L
    # ----- TODO -----

    H, W = wordmap.shape

    hist_all = np.ndarray(shape=(0,))
    hist_all = create_level_histograms(opts, wordmap, 0, hist_all)
    hist_all *= 2**(-1)

    for l in range(L-1,-1,-1):
        step_size = 2**(L-l) * 2**(L-l) * K
        for i in range(2**l * 2**l):
            cell_hist = hist_all[step_size*i: step_size*(i+1)]
            cell_hist = np.reshape(cell_hist, (2**(L-l) * 2**(L-l), K))
            cell_hist = np.sum(cell_hist, axis=0)
            hist_all = np.concatenate((hist_all, cell_hist), axis=0)
        
        if l == 0:
            level_weight = 2**(-L)
        else:
            level_weight = 2**(l-L-1)
        
        hist_all *= level_weight
            
    return hist_all


def create_level_histograms(opts, wordmap, l, hist_all):

    if l==opts.L:
        return get_feature_from_wordmap(opts, wordmap)

    
    # divide the wordmap into four parts
    H, W = wordmap.shape
    ni, nj = 2, 2

    wordmap_tl = wordmap[0:H//ni, 0:W//nj]
    wordmap_tr = wordmap[0:H//ni, W//nj:]
    wordmap_br = wordmap[H//ni:, W//nj:]
    wordmap_bl = wordmap[H//ni:, 0:W//nj]

    hist_tl = create_level_histograms(opts, wordmap_tl, l+1, hist_all)
    hist_tr = create_level_histograms(opts, wordmap_tr, l+1, hist_all)
    hist_br = create_level_histograms(opts, wordmap_br, l+1, hist_all)
    hist_bl = create_level_histograms(opts, wordmap_bl, l+1, hist_all)

    hist_all = np.concatenate((hist_all, hist_tl, hist_tr, hist_br, hist_bl), axis=0)

    return hist_all


def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    # ----- TODO -----
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    if len(img.shape) < 3:
        img = np.stack([img,img,img], axis=0)
        img = np.moveaxis(img, 0, -1)
    
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    hist_all = get_feature_from_wordmap_SPM(opts, wordmap)

    return hist_all

def pool_build_system(opts, img_path):
    global dictionary
    img_feature = get_image_feature(opts, img_path, dictionary)

    return img_feature


def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    global dictionary
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----
    training_size = len(train_files)
    # training_size = 10
    print(f"training size = {training_size}")

    img_args = [(opts, join(data_dir, train_files[i]), dictionary) for i in range(training_size)]

    pool = multiprocessing.Pool(n_worker)
    # features = pool.starmap(pool_build_system, img_args)
    features = pool.starmap(get_image_feature, img_args)
    features = np.array(features)
    
    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    T, hist_len = histograms.shape

    # word_hist = np.stack([word_hist for _ in range(T)], axis=1)
    # word_hist = np.moveaxis(word_hist, 0, -1)


    word_hist = np.array([word_hist for _ in range(T)])
    hist_similarity = 1- np.minimum(word_hist, histograms).sum(axis=1)

    return hist_similarity

def distance_to_class(word_hist, class_histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    T, hist_len = class_histograms.shape

    # word_hist = np.stack([word_hist for _ in range(T)], axis=1)
    # word_hist = np.moveaxis(word_hist, 0, -1)


    word_hist = np.array([word_hist for _ in range(T)])
    hist_similarity = 1- np.minimum(word_hist, class_histograms).sum(axis=1)

    return hist_similarity.sum()/T


def pool_testing(opts, img_path):
        global dictionary, train_features, train_labels

        test_feature = get_image_feature(opts, img_path, dictionary)
        dist_to_train = distance_to_set(test_feature, train_features)
        ind = np.argmin(dist_to_train)
        pred_label = train_labels[ind]

        kit_ind = [427,576]
        laun_ind = [577,726]

        kit_hists = train_features[427:577,:]
        laun_hists = train_features[578:727,:]

        if pred_label == 3 or pred_label == 4:
            sim_kit = distance_to_class(test_feature, kit_hists)
            sim_laund = distance_to_class(test_feature, laun_hists)

            if sim_kit < sim_laund:
                pred_label = 3
            elif sim_kit < sim_laund:
                pred_label = 4

        return pred_label

def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))

    global dictionary, train_features, train_labels
    dictionary = trained_system['dictionary']
    train_labels = trained_system['labels']
    train_features = trained_system['features']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----

    test_size = len(test_files)
    # test_size = 10
    print(f"test_size: {test_size}")
    testing_args = [(opts, join(data_dir, test_files[i])) for i in range(test_size)]

    pool = multiprocessing.Pool(n_worker)
    preds = pool.starmap(pool_testing, testing_args)

    num_classes = 8
    confusion_matrix = np.zeros(shape=(num_classes,num_classes))
    for i in range(test_size):
        confusion_matrix[preds[i]][test_labels[i]] += 1
    
    accuracy = np.trace(confusion_matrix)/confusion_matrix.sum()

    return confusion_matrix, accuracy



    















'''
############## below codes are just for any future reference ##########


# def get_feature_from_wordmap_SPM(opts, wordmap):
    
#     K = opts.K
#     L = opts.L
#     # ----- TODO -----

#     H, W = wordmap.shape

#     hist_all = np.ndarray(shape=(0,))

#     # histogram for Lth level
#     ni, nj = H//2**L, W//2**L
#     for i in range(2**L):
#         for j in range(2**L):
#             cell = wordmap[ni*i: ni*(i+1), nj*j: nj*(j+1)]
#             hist_all = np.concatenate((hist_all, get_feature_from_wordmap(opts, cell)), axis=0)

#     base_hist = np.copy(hist_all)

#     hist_all = create_level_histograms(opts, wordmap, 0, hist_all, base_hist)

#     for l in range(L):
#         print(f"level -- {l}")
#         wordmap_temp = np.copy(wordmap)

#         nrows, ncols = 2**l, 2**l
#         cells = wordmap_temp.reshape(H//nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols)

#         for cell in cells:
#             print("computing hist of cells")
#             hist_all = np.concatenate((hist_all, get_feature_from_wordmap(opts, cell)), axis=0)
        
#         if l == 0:
#             level_weight = 2**(-L)
#         else:
#             level_weight = 2**(l-L-1)
        
#         hist_all *= level_weight

#     hist_all /= hist_all.sum()
#     hist_all.shape

#     return hist_all


'''