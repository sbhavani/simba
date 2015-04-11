import os
import cv2
from utilities_lib import get_imgfiles
from crop_img_lib.find_objects import process_image


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
    for image in get_imgfiles('test_crops/Oyayai/'):

        img_fname = 'c_' + os.path.basename(image) + '.jpg'

        img, center, radius, zoom_level = process_image(image, None)

        # crop image
        x1 = int(center[0] - radius)
        y1 = int(center[1] - radius)
        x2 = int(center[0] + radius)
        y2 = int(center[1] + radius)

        roi = img[y1:y2, x1:x2, :]

        cv2.imwrite(os.path.join('output', img_fname), roi)