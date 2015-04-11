import os

import cv2


# SemanticMD libraries
from utilities_lib import get_imgfiles, get_categories
from find_objects import process_image
import settings


def show_results(img, center, radius):
    """
    This demo process images and tries to find salient objects in them in very
    naive way. The algorithm is the following:
    Phase 1.
        Try to find any faces by using classifier
        (namely a cascade of boosted classifiers working with haar-like features)
    Phase 2.
        If no faces were found, try to find the most "distinctable" object by using
        ORB features. The demo is looking for a single object only and will
        group set of features into an object shape by using k-means algorithm.
    Use Space to navigate to next result and Esc to exit.
    """
    green_color = (0, 255, 0)
    # outlines found object
    # cv2.circle(img, center, radius, green_color, 1, 8, 0)
    x1 = center[0] - radius
    y1 = center[1] + radius
    x2 = center[0] + radius
    y2 = center[1] - radius
    top_left = (x1, y1)
    bottom_right = (x2, y2)
    cv2.rectangle(img, top_left, bottom_right, green_color, 1, 8, 0)
    cv2.imshow('Naive Salient Object Detection', img)


if __name__ == '__main__':

    # iterate through all the categories (each folder in datasetpath)
    for cat in get_categories(settings.DATASETPATH):
        cat_dir = os.path.join(settings.DATASETPATH, cat)
        # cropped images saved here
        cat_dir_local = os.path.join(settings.RESULTSPATH, cat)
        if not os.path.exists(cat_dir_local):
            os.makedirs(cat_dir_local)
        # iterate through all images for a category
        for image in get_imgfiles(cat_dir):
            img_fname = 'c_' + os.path.basename(image) + '.jpg'

            img, center, radius, zoom_level = process_image(image, None)

            # crop image
            x1 = int(center[0] - radius)
            y1 = int(center[1] - radius)
            x2 = int(center[0] + radius)
            y2 = int(center[1] + radius)

            roi = img[y1:y2, x1:x2, :]

            cv2.imwrite(os.path.join(cat_dir_local, img_fname), roi)