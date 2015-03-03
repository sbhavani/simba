__author__ = 'sb'

# General libraries
from os.path import join
from pickle import dump, HIGHEST_PROTOCOL
from numpy import sqrt
import scipy.cluster.vq as vq

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

# user-defined libraries
import settings
from learn import get_categories, get_imgfiles, extract_features, \
    dict2numpy_featurevec, dict2numpy_kps, is_kp_descriptor


print("---------------------")
print("## loading the images and extracting the " + settings.DETECTOR + " features")
datasetpath = settings.DATASETPATH
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
    cat_features = extract_features(cat_files, settings.DETECTOR)
    all_files = all_files + cat_files
    all_features.update(cat_features)
    cat_label[cat] = label
    for i in cat_files:
        all_files_labels[i] = label

print("---------------------")
print("## computing the visual words via k-means")
if is_kp_descriptor(settings.DETECTOR):
    all_features_array = dict2numpy_kps(all_features)
else:
    all_features_array = dict2numpy_featurevec(all_features)

nfeatures = all_features_array.shape[0]
nclusters = int(sqrt(nfeatures))
codebook, distortion = vq.kmeans(all_features_array,
                                 nclusters,
                                 thresh=settings.K_THRESH)

with open(join(settings.RESULTSPATH, datasetpath + settings.CODEBOOK_FILE), 'wb') as f:
    dump(codebook, f, protocol=HIGHEST_PROTOCOL)

print("---------------------")
print("## train randomforest")
clf = RandomForestClassifier(n_jobs=2)

all_files_labels_list = list(all_files_labels.values())
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    all_features_array,
    all_files_labels_list,
    test_size=0.1,
    random_state=0
)
clf.fit(X_train, y_train)

num_folds = 10
scores = cross_validation.cross_val_score(
    clf,
    all_features_array,
    all_files_labels_list,
    cv=num_folds
)

acc = round((sum(scores) / num_folds) * 100, 2)
print("Average " + str(num_folds) + "-fold accuracy: " + str(acc))

print("---------------------")
print("## store best classifier")
best_clf = clf
with open(settings.MODEL_FILE, 'wb') as f:
    dump(best_clf, f, protocol=HIGHEST_PROTOCOL)

print("--------------------")
print("## outputting results")
print("model file: " + datasetpath + settings.MODEL_FILE)
