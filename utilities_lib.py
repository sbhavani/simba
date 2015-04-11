__author__ = 'sb'
from os.path import isdir, basename, join, splitext
from glob import glob
from random import choice

# supported image extensions
EXTENSIONS = [".jpeg", ".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]


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


def multi_delete(list_, index_list):
    indexes = sorted(index_list, reverse=True)
    for index in indexes:
        del list_[index]
    return list_


def test_random_img(dataset_path):
    all_cats = get_categories(dataset_path)

    cat_index = choice(range(len(all_cats)))
    cat_path = join(dataset_path, all_cats[cat_index])

    cat_files = get_imgfiles(cat_path)

    img_index = choice(range(len(cat_files)))

    return cat_files[img_index]
