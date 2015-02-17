__author__ = 'sb'

EXTENSIONS = [".jpeg", ".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
DATASETPATH = '/home/sb/git_projects/simba/testdata_male'
FEATUREPATH = '/home/sb/git_projects/simba/feature_files'
RESULTSPATH = '/home/sb/git_projects/simba/results'

HISTOGRAMS_FILE = 'testdata_maletrainingdata.svm'
CODEBOOK_FILE = 'testdata_malecodebook.file'
MODEL_FILE = 'testdata_maletrainingdata.svm.model'

DETECTOR = "ORB"  # set feature detector type

int2lion = {0: 'Ambogga', 3: 'Masusu', 4: 'Oyayai_(Lorpolosie)', 2: 'Maringa', 1: 'Kasayio'}
