from os.path import exists, isdir, basename, join, splitext
from glob import glob
from numpy import zeros, resize, sqrt, hstack, vstack, savetxt, zeros_like, uint8, histogram
import scipy.cluster.vq as vq
import libsvm
from pickle import dump, load, HIGHEST_PROTOCOL
import argparse
import cv2
import os
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

EXTENSIONS = [".jpeg", ".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
DATASETPATH = 'testdata_female'
FEATUREPATH = 'feature_files'
PRE_ALLOCATION_BUFFER = 1000  # for ORB
HISTOGRAMS_FILE = 'trainingdata.svm'
K_THRESH = 1  # early stopping threshold for kmeans originally at 1e-5, increased for speedup
CODEBOOK_FILE = 'codebook.file'
DETECTOR = "ORB"  # set feature detector type

def parse_arguments():
    parser = argparse.ArgumentParser(description='train a visual bag of words model')
    parser.add_argument('-d', help='path to the dataset', required=False, default=DATASETPATH)
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
                      if splitext(fname)[-1].lower() in EXTENSIONS])
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
        features_fname = os.path.join(FEATUREPATH, os.path.basename(fname) + '.' + my_feature)
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
    array = zeros((nkeys * PRE_ALLOCATION_BUFFER, nwords))
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


if __name__ == '__main__':
    print("---------------------")
    print("## loading the images and extracting the " + DETECTOR + " features")
    args = parse_arguments()
    datasetpath = args.d
    cats = get_categories(datasetpath)
    ncats = len(cats)
    print("searching for folders at " + datasetpath)
    if ncats < 1:
        raise ValueError('Only ' + str(ncats) + ' categories found. Wrong path?')
    print("found following folders / categories:")
    print(cats)
    print("---------------------")
    all_files = []
    all_files_labels = {}
    all_features = {}
    cat_label = {}
    for cat, label in zip(cats, list(range(ncats))):
        cat_path = join(datasetpath, cat)
        cat_files = get_imgfiles(cat_path)
        cat_features = extract_features(cat_files, DETECTOR)
        all_files = all_files + cat_files
        all_features.update(cat_features)
        cat_label[cat] = label
        for i in cat_files:
            all_files_labels[i] = label

    print("---------------------")
    print("## computing the visual words via k-means")
    if DETECTOR == "LBP":
        all_features_array = dict2numpy_featurevec(all_features)
    else:
        all_features_array = dict2numpy_kps(all_features)
    nfeatures = all_features_array.shape[0]
    nclusters = int(sqrt(nfeatures))
    codebook, distortion = vq.kmeans(all_features_array,
                                     nclusters,
                                     thresh=K_THRESH)

    with open(datasetpath + CODEBOOK_FILE, 'wb') as f:

        dump(codebook, f, protocol=HIGHEST_PROTOCOL)

    print("---------------------")
    print("## compute the visual words histograms for each image")
    all_word_histgrams = {}
    for imagefname in all_features:
        word_histgram = computeHistograms(codebook, all_features[imagefname])
        all_word_histgrams[imagefname] = word_histgram

    print("---------------------")
    print("## write the histograms to file to pass it to the svm")
    writeHistogramsToFile(nclusters,
                          all_files_labels,
                          all_files,
                          all_word_histgrams,
                          datasetpath + HISTOGRAMS_FILE)

    print("---------------------")
    print("## train svm")
    c, g, rate, model_file = libsvm.grid(datasetpath + HISTOGRAMS_FILE,
                                         png_filename='grid_res_img_file.png')

    print("--------------------")
    print("## outputting results")
    print("model file: " + datasetpath + model_file)
    print("codebook file: " + datasetpath + CODEBOOK_FILE)
    print("category      ==>  label")
    for cat in cat_label:
        print('{0:13} ==> {1:6d}'.format(cat, cat_label[cat]))
