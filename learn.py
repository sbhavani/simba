from os.path import exists, isdir, basename, join, splitext
from glob import glob
from pickle import dump, load, HIGHEST_PROTOCOL
import argparse
import os

from numpy import zeros, resize, hstack, vstack, savetxt, zeros_like, uint8, histogram
import scipy.cluster.vq as vq
import cv2
import matplotlib.pyplot as plt

# ML libraries
from skimage.feature import local_binary_pattern

import settings


def parse_arguments():
    parser = argparse.ArgumentParser(description='train a visual bag of words model')
    parser.add_argument('-d', help='path to the dataset', required=False, default=settings.DATASETPATH)
    args = parser.parse_args()
    return args


def get_categories(datasetpath):
    cat_paths = [files
                 for files in glob(datasetpath + "/*")
                 if isdir(files)]
    cat_paths.sort()
    cats = [basename(cat_path) for cat_path in cat_paths]
    return cats


def get_imgfiles(path):
    all_files = []
    all_files.extend([join(path, basename(fname))
                      for fname in glob(path + "/*")
                      if splitext(fname)[-1].lower() in settings.EXTENSIONS])
    return all_files


def get_detector(my_detector):
    if my_detector == "ORB":
        return cv2.ORB_create()
    if my_detector == "BRISK":
        return cv2.BRISK_create()
    if my_detector == "KAZE":
        return cv2.KAZE_create()
    if my_detector == "AKAZE":
        return cv2.AKAZE_create()


def get_gray_img(img_fname):
    image_name, extension = os.path.splitext(img_fname)
    print(img_fname)
    if extension == ".tif":
        color_img = plt.imread(img_fname)
        gray_img = uint8(color_img)
    else:
        img = cv2.imread(img_fname)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray_img


def process_image(my_feature, imgfname):
    gray = get_gray_img(imgfname)

    if my_feature == "LBP":
        # settings for LBP
        radius = 5
        n_points = 8 * radius
        lbp_method = 'uniform'
        lbp = local_binary_pattern(gray, n_points, radius, lbp_method)
        n_bins = lbp.max() + 1
        featurevec, _ = histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
    else:
        detector = get_detector(my_feature)
        locs, featurevec = detector.detectAndCompute(gray, None)

    return featurevec


def extract_features(input_files, my_feature):
    print("extracting " + my_feature + " features")
    all_features_dict = {}
    for i, fname in enumerate(input_files):
        features_fname = os.path.join(settings.FEATUREPATH, os.path.basename(fname) + '.' + my_feature)
        if not exists(features_fname):
            open(features_fname, 'w').close()
            print("calculating " + my_feature + " features for", fname)
            img_features = process_image(my_feature, fname)
            with open(features_fname, 'wb') as f:
                dump(img_features, f, protocol=HIGHEST_PROTOCOL)
        else:
            print("gathering " + my_feature + " features for", fname)
            with open(features_fname, 'rb') as f:
                img_features = load(f)

        all_features_dict[fname] = img_features

    return all_features_dict


def dict2numpy_kps(my_dict):
    nkeys = len(my_dict)
    nwords = list(my_dict.values())[0].shape[1]
    if nwords > 64:
        nwords = 64
    array = zeros((nkeys * settings.PRE_ALLOCATION_BUFFER, nwords))
    pivot = 0
    for key in list(my_dict.keys()):
        value = my_dict[key]
        if value is None:
            print(key)
            continue
        nelements = value.shape[0]
        while pivot + nelements > array.shape[0]:
            padding = zeros_like(array)
            array = vstack((array, padding))
        array[pivot:pivot + nelements] = value
        pivot += nelements
    array = resize(array, (pivot, nwords))
    return array


def dict2numpy_featurevec(my_dict):
    nkeys = len(my_dict)
    array = vstack(tuple(list(my_dict.values())))
    return array


def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = histogram(code,
                                              bins=list(range(codebook.shape[0] + 1)),
                                              normed=True)
    return histogram_of_words


def writeHistogramsToFile(nwords, labels, fnames, all_word_histgrams, features_fname):
    data_rows = zeros(nwords + 1)  # +1 for the category label
    for fname in fnames:
        histogram = all_word_histgrams[fname]
        if (histogram.shape[0] != nwords):  # scipy deletes empty clusters
            nwords = histogram.shape[0]
            data_rows = zeros(nwords + 1)
            print('nclusters have been reduced to ' + str(nwords))
        data_row = hstack((labels[fname], histogram))
        data_rows = vstack((data_rows, data_row))
    data_rows = data_rows[1:]
    fmt = '%i '
    for i in range(nwords):
        fmt = fmt + str(i) + ':%f '
    savetxt(features_fname, data_rows, fmt)