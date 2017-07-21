#!/usr/bin/python

import os
import scipy
import scipy.misc
from scipy import misc
import image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from sklearn import preprocessing
import numpy as np
import numpy.linalg as lin

# total count of faces
L = 154

# list file directories
basedir = './yalefaces/'
flist = os.listdir(basedir)

# create zero array
arr = np.zeros((L, 1600))

num = 0

# load data matrix out of PGM files
for f in flist:
    # construct target
    tfile = basedir + str(f)

    print("Opening:", tfile)

    im = misc.imread(tfile, f)

    image = scipy.misc.imresize(im, (40,40))
    flat_image = image.flatten()

    arr[num] = flat_image
    num += 1

X_scaled = preprocessing.StandardScaler().fit_transform(arr)

covariant_matrix = np.cov(X_scaled.T)

eigen_values, eigen_vectors = np.linalg.eig(covariant_matrix)

eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]

w = np.hstack((eigen_pairs[0][1] [:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))

X_train_pca = X_scaled.dot(w)
