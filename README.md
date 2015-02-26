Minimal Bag of Visual Words Image Classifier
============================================

Implementation of a content based image classifier using the [bag of visual words model][1] in Python. 

As the name suggests, this is only a minimal example to illustrate the general workings of such a system. For a more advanced system that provides better performance check: https://github.com/shackenberg/phow_caltech101.py

The approach consists of two major steps called learning and classifying, represented in the files [`learn.py`][3] and [`classify.py`][4].

The script `learn.py` will generate a visual vocabulary and train a classifier using a user provided set of already classified images.
After the learning phase `classify.py` will use the generated vocabulary and the trained classifier to predict the class for any image given to the script by the user.

The learning consists of:

1. Extracting local features of all the dataset images
2. Generating a codebook of visual words with clustering of the features
3. Aggregating the histograms of the visual words for each of the traning images
4. Feeding the histograms to the classifier to train a model

The classification consists of:

1. Extracting local features of the to be classified image
2. Aggregating the histograms of the visual words for the image using the prior generated codebook
4. Feeding the histogram to the classifier to predict a class for the image

This code relies on:

 - SIFT features for local features
 - k-means for generation of the words via clustering
 - SVM as classifier using the LIBSVM library

### Example use:
  
You train the classifier for a specific dataset with: 

    python learn.py -d path_to_folders_with_images

To classify images use:

    python classify.py -c path_to_folders_with_images/codebook.file -m path_to_folders_with_images/trainingdata.svm.model images_you_want_to_classify

The dataset should have following structure, where all the images belonging to one class are in the same folder:

    .
    |-- path_to_folders_with_images
    |    |-- class1
    |    |-- class2
    |    |-- class3
    ...
    |    └-- classN

The folder can have any name. One example dataset would be the [Caltech 101 dataset][2].

### Prerequisites:

Results
=====================================
75% for females
61% for males
53% for combined