from __future__ import print_function
from caffe.model_libs import *
import pickle
import numpy as np
from sklearn import svm
from sklearn.externals import joblib

def extractTopK(A, k):
    conf = A[:, 1]
    index = np.argsort(-conf)
    if len(index) > k:
        output = A[index[0:k], 0:6]
    else:
        o1 = A[index, 0:6]
        o2 = np.zeros([k - A.shape[0], 6])
        output = np.concatenate((o1, o2), axis=0)
    # flatten
    instance = np.squeeze(np.reshape(output, [1, output.shape[0] * output.shape[1]]))
    return instance

# ------------------------------------------------------
if __name__ == '__main__':
    # Read indices of easy and hard images
    output = open("output_files/trainval/inds_300_rfcn_dif.txt", 'r')
    fid = pickle.load(output)
    output.close()

    num_train = 16551
    num_test = 4952
    labels = np.ones([num_train, ])
    labels[fid] = 0

    # Load proposals from Tiny-Yolo
    fid = open("AF/tiny_yolo/detection_0712trainval_top50_tiny_yolo.txt", "r")
    fv = pickle.load(fid)
    fid.close()
    topK = 25
    use_weight = True
    ofv = fv[:, 0:dim*topK]

    num_classes = 20
    dim = 5
    nfv = np.zeros([ofv.shape[0], num_classes + dim * topK])
    for i in range(fv.shape[0]):
        for j in range(topK):
            if int(ofv[i, 6 * j]) != 0:
                nfv[i, int(ofv[i, 6 * j] - 1)] += 1
            nfv[i, dim * j + 20] = ofv[i, 6 * j + 1]
            nfv[i, (dim * j + 20 + dim - 4):(dim * j + 20 + dim)] = ofv[i, (6 * j + 2)], ofv[i, (6 * j + 3)], \
                                                                    ofv[i, (6 * j + 4)] - ofv[i, (6 * j + 2)], ofv[
                                                                        i, (6 * j + 5)] - ofv[i, (6 * j + 3)]

    fv = nfv

    # Load train/val SPLIT for AF classifier
    fid = open("AF/indices/train_indices.txt", "r")
    train_index = pickle.load(fid)
    fid.close()
    fid = open("AF/indices/test_indices.txt", "r")
    test_index = pickle.load(fid)
    fid.close()

    xtrain = fv[train_index, :]
    ytrain = labels[train_index]
    xtest = fv[test_index, :]
    ytest = labels[test_index]

    """We will train the Adaptive Feeding classifier from now"""

    # SET SAMPLING WEIGHT
    sample_weight = {0: 1.0, 1: 3.0}

    clf = svm.LinearSVC(verbose=True, dual=False, class_weight=sample_weight).fit(xtrain, ytrain)
    ypred = clf.predict(xtest)
    # Save AF classifier
    joblib.dump(clf, ('AF/SVM/300_rfcn_VOC' + '.m'))
