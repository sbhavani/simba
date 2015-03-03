from os.path import exists, isdir, basename, join, splitext
from glob import glob
from pickle import dump, load, HIGHEST_PROTOCOL
import argparse
import os
import numpy

# ML libraries
from skimage.feature import (
    greycomatrix, greycoprops, hog, local_binary_pattern, daisy
)
import scipy.cluster.vq as vq

# CV libraries
import cv2
from PIL import Image, ImageOps

# user-defined libraries
import settings


# private functions
def _get_detector(my_detector):
    if my_detector == "ORB":
        return cv2.ORB_create()
    if my_detector == "BRISK":
        return cv2.BRISK_create()
    if my_detector == "KAZE":
        return cv2.KAZE_create()
    if my_detector == "AKAZE":
        return cv2.AKAZE_create()


# public functions
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


def is_kp_descriptor(my_detector):
    return my_detector in settings.detector_types


def compute_feats_daisy(file_hash):
    my_gray_img = get_gray_img(file_hash)

    # TODO: change settings to reduce dimensionality
    descs, descs_img = daisy(my_gray_img, step=180, radius=58, rings=2, histograms=6,
                             orientations=8, visualize=True)

    return descs


def compute_feats_hog(file_hash):
    """
    Extract HOG features
      my_imgfname: image filename

    Reference: http://www.vlfeat.org/overview/hog.html
    """
    my_gray_img = get_gray_img(file_hash)

    featvec = hog(my_gray_img,
                  orientations=12,
                  pixels_per_cell=(6, 6),
                  cells_per_block=(3, 3),
                  visualise=False,
                  normalise=True)

    featvec, _ = numpy.histogram(featvec, bins=50)

    return featvec


def compute_feats_detector(img_fname, my_feature):
    """
    Extract keypoint detector features given an image
      my_img: numpy matrix representing an image

    Reference: http://www.vlfeat.org/api/sift.html
    """

    if my_feature == 'DAISY':
        return compute_feats_daisy(img_fname)

    # convert to grayscale to drop the 3rd dim
    my_gray_img = get_gray_img(img_fname)

    detector = _get_detector(my_feature)
    kps, descriptors = detector.detectAndCompute(my_gray_img, None)

    return descriptors


def compute_feats_lbp(img_fname):
    """
    Extract local binary pattern (LBP) features
      my_imgfname: image filename

    """
    my_gray_img = get_gray_img(img_fname)

    # settings for LBP
    radius = 8
    n_points = 8 * radius
    lbp_method = 'uniform'
    lbp = local_binary_pattern(my_gray_img, n_points, radius, lbp_method)
    n_bins = lbp.max() + 1
    featurevec, _ = numpy.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))

    return featurevec


def compute_feats_glcm(imgfname):
    """

    Extract GLCM features
    Reference: http://www.fp.ucalgary.ca/mhallbey/tutorial.htm
    """

    grayimg = get_gray_img(imgfname)

    num_feats = 6
    featvec = numpy.zeros((1, num_feats))

    glcm = greycomatrix(
        grayimg, [1, 2], [0, numpy.pi / 2],
        levels=256,
        symmetric=True,
        normed=True)

    featvec[0, 0] = greycoprops(glcm, 'contrast')[0][0]
    featvec[0, 1] = greycoprops(glcm, 'energy')[0][0]
    featvec[0, 2] = greycoprops(glcm, 'homogeneity')[0][0]
    featvec[0, 3] = greycoprops(glcm, 'correlation')[0][0]
    featvec[0, 4] = greycoprops(glcm, 'ASM')[0][0]
    featvec[0, 5] = greycoprops(glcm, 'dissimilarity')[0][0]

    return featvec[0, :]


def get_gray_img(img_fname):
    input = Image.open(img_fname)
    output = ImageOps.grayscale(input)

    return output


def process_image(my_feature, imgfname):
    gray = get_gray_img(imgfname)

    if my_feature == "LBP":
        featurevec = compute_feats_lbp(imgfname)
    elif my_feature == 'GLCM':
        featurevec = compute_feats_glcm(imgfname)
    elif my_feature == 'HOG':
        featurevec = compute_feats_hog(imgfname)
    else:
        detector = _get_detector(my_feature)
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
    array = numpy.zeros((nkeys * settings.PRE_ALLOCATION_BUFFER, nwords))
    pivot = 0
    for key in list(my_dict.keys()):
        value = my_dict[key]
        if value is None:
            continue
        nelements = value.shape[0]
        while pivot + nelements > array.shape[0]:
            padding = numpy.zeros_like(array)
            array = numpy.vstack((array, padding))
        array[pivot:pivot + nelements] = value
        pivot += nelements
    array = numpy.resize(array, (pivot, nwords))
    return array


def dict2numpy_featurevec(my_dict):
    nkeys = len(my_dict)
    array = numpy.vstack(tuple(list(my_dict.values())))
    return array


def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = numpy.histogram(code,
                                                    bins=list(range(codebook.shape[0] + 1)),
                                                    normed=True)
    return histogram_of_words