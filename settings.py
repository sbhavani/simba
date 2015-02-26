__author__ = 'sb'

EXTENSIONS = [".jpeg", ".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
# original dataset
DATASETPATH = '/home/sb/git_projects/simba/testdata_male'
FEATUREPATH = '/home/sb/git_projects/simba/feature_files'
RESULTSPATH = '/home/sb/git_projects/simba/results'

ALL_PATHS = [DATASETPATH, FEATUREPATH]


HISTOGRAMS_FILE = 'testdata_maletrainingdata.histograms'
CODEBOOK_FILE = 'testdata_malecodebook.file'
MODEL_FILE = 'testdata_maletrainingdata.model'

DETECTOR = "LBP"  # set feature detector type

int2lion = {0: 'Ambogga', 3: 'Masusu', 4: 'Oyayai_(Lorpolosie)', 2: 'Maringa', 1: 'Kasayio'}

K_THRESH = 10  # early stopping threshold for kmeans originally at 1e-5, increased for speedup

PRE_ALLOCATION_BUFFER = 1000  # for ORB
