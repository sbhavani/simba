import argparse
from pickle import load

import libsvm
from learn import extract_features, computeHistograms, writeHistogramsToFile
import settings

def parse_arguments():
    parser = argparse.ArgumentParser(description='classify images with a visual bag of words model')
    parser.add_argument('-c', help='path to the codebook file', required=False, default=settings.CODEBOOK_FILE)
    parser.add_argument('-m', help='path to the model  file', required=False, default=settings.MODEL_FILE)
    parser.add_argument('input_images', help='images to classify', nargs='+')
    args = parser.parse_args()
    return args


def classify_lion(fnames):
    print("---------------------")
    print("## extract Sift features")
    all_files = []
    all_files_labels = {}
    all_features = {}

    model_file = settings.MODEL_FILE
    codebook_file = settings.CODEBOOK_FILE
    all_features = extract_features(fnames, settings.DETECTOR)
    for i in fnames:
        all_files_labels[i] = 0  # label is unknown

    print("---------------------")
    print("## loading codebook from " + codebook_file)
    with open(codebook_file, 'rb') as f:
        codebook = load(f)

    print("---------------------")
    print("## computing visual word histograms")
    all_word_histgrams = {}
    for imagefname in all_features:
        word_histgram = computeHistograms(codebook, all_features[imagefname])
        all_word_histgrams[imagefname] = word_histgram

    print("---------------------")
    print("## write the histograms to file to pass it to the svm")
    nclusters = codebook.shape[0]
    writeHistogramsToFile(nclusters,
                          all_files_labels,
                          fnames,
                          all_word_histgrams,
                          settings.HISTOGRAMS_FILE)

    print("---------------------")
    print("## test data with svm")
    result = libsvm.test(settings.HISTOGRAMS_FILE, model_file)

    return settings.int2lion[result[0]]


if __name__ == '__main__':

    print("---------------------")
    print("## extract Sift features")
    all_files = []
    all_files_labels = {}
    all_features = {}

    args = parse_arguments()
    model_file = args.m
    codebook_file = args.c
    fnames = args.input_images
    all_features = extract_features(fnames, settings.DETECTOR)
    for i in fnames:
        all_files_labels[i] = 0  # label is unknown

    print("---------------------")
    print("## loading codebook from " + codebook_file)
    with open(codebook_file, 'rb') as f:
        codebook = load(f)

    print("---------------------")
    print("## computing visual word histograms")
    all_word_histgrams = {}
    for imagefname in all_features:
        word_histgram = computeHistograms(codebook, all_features[imagefname])
        all_word_histgrams[imagefname] = word_histgram

    print("---------------------")
    print("## write the histograms to file to pass it to the svm")
    nclusters = codebook.shape[0]
    writeHistogramsToFile(nclusters,
                          all_files_labels,
                          fnames,
                          all_word_histgrams,
                          settings.HISTOGRAMS_FILE)

    print("---------------------")
    print("## test data with svm")
    result = libsvm.test(settings.HISTOGRAMS_FILE, model_file)
    print('Predicted Lion: \n' + int2lion[result[0]])
