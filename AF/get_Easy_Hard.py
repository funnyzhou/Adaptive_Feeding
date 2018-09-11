from __future__ import print_function
from caffe.model_libs import *
from google.protobuf import text_format
import caffe
import pickle
from caffe import layers as L
from caffe import params as P
import math
import os
import numpy as np
import time
import shutil
import stat
import subprocess
import sys
import time
import matplotlib.pyplot as plt
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
from caffe.model_libs import *
import argparse

'''
This file aims to split the dataset into two parts: Easy vs. Hard.
'''

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for '
          '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)


def extract(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    detection_out = np.reshape(np.array([None] * 7), [1, 7])
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        dout = np.zeros([dets.shape[0], 7])
        dout[:, 1] = cls_ind
        dout[:, 2] = dets[:, 4]
        dout[:, 3] = dets[:, 0] / im.shape[1]
        dout[:, 4] = dets[:, 1] / im.shape[0]
        dout[:, 5] = dets[:, 2] / im.shape[1]
        dout[:, 6] = dets[:, 3] / im.shape[0]
        detection_out = np.concatenate((detection_out, dout), axis=0)
    detection_out = np.delete(detection_out, 0, axis=0)
    detection_out = np.reshape(detection_out, [1, 1, detection_out.shape[0], detection_out.shape[1]])
    return detection_out


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def ismember(A, B):
    count = 0
    for a in A:
        if a == B:
            count += 1
    return count


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


if '__name__' == '__main__':
    # Set your home path
    home_path = '/home/zhouhy/'

    # Set gpu id
    gpu_device_id = 0
    caffe.set_device(gpu_device_id)
    caffe.set_mode_gpu()

    labelmap_file = os.path.join(home_path, 'caffe_ssd/data/VOC0712/labelmap_voc.prototxt')
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    # Read image paths
    img_path = []
    home_path = os.path.join(home_path, 'data/VOCdevkit/')
    f = open('data/VOC0712/trainval.txt', 'r')
    done = 0
    while not done:
        line = f.readline()
        if (line != ''):
            img_path.append(home_path + line.split(' ')[0])
        else:
            done = 1

    # Load SSD and R-FCN
    rfcn_prototxt = 'AF/R-FCN/test_agnostic.prototxt'
    rfcn_weights = 'AF/R-FCN/resnet101_rfcn_final.caffemodel'

    prototxt1 = "AF/SSD300/split.prototxt"
    prototxtm = "AF/cal_mAP/split.prototxt"

    weights1 = "AF/SSD300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel"
    weightsm = "AF/SSD300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel"

    ssd = caffe.Net(prototxt1, weights1, caffe.TEST)
    rfcn = caffe.Net(rfcn_prototxt, rfcn_weights, caffe.TEST)
    
    '''Note: netm is used to process detection results from R-FCN to keep consistent with SSD'''
    netm = caffe.Net(prototxtm, weightsm, caffe.TEST)

    num_test2007 = 4952
    num_train = 16551
    conf_thresh = 0.01

    all_map1 = np.zeros([num_train, ])
    all_map2 = np.zeros([num_train, ])

    for i in range(num_train):
        proposals = extract(rfcn, img_path[i])

        netm.blobs['merged_proposals'].reshape(proposals.shape[0], proposals.shape[1], \
                                               proposals.shape[2], proposals.shape[3])
        netm.blobs['merged_proposals'].data[...] = proposals

        netm.forward()

        ssd.forward()

        label1 = ssd.blobs['label'].data
        eval1 = ssd.blobs['detection_eval'].data
        label2 = netm.blobs['label'].data
        eval2 = netm.blobs['detection_eval'].data

        mAP1 = np.zeros([20, ])
        mAP2 = np.zeros([20, ])

        # reshape
        label1 = np.reshape(label1, [label1.shape[2], label1.shape[3]])
        label2 = np.reshape(label2, [label2.shape[2], label2.shape[3]])
        eval1 = np.reshape(eval1, [eval1.shape[2], eval1.shape[3]])
        eval2 = np.reshape(eval2, [eval2.shape[2], eval2.shape[3]])
        eval1 = eval1[20:eval1.shape[0], :]
        eval2 = eval2[20:eval2.shape[0], :]
        if np.array_equal(np.unique(label1[:, 1]), np.unique(label2[:, 1])):
            label = label1
            unique_label = np.unique(label1[:, 1])
        else:
            ss = "error"
            print("%s" % (ss))

        for j in range(unique_label.shape[0]):
            all_pos_indices = np.where(label[:, 1] == unique_label[j])
            easy_indices = np.where(label[all_pos_indices[0][:], 7] == 0)  # easy or difficult
            npos = easy_indices[0].size
            indices1 = np.where(eval1[:, 1] == unique_label[j])
            if indices1[0].size and (npos != 0):
                BB = eval1[indices1[0][:], :]
                sort_ind = np.argsort(-BB[:, 2])
                tp = BB[sort_ind, 3]
                fp = BB[sort_ind, 4]
                fp = np.cumsum(fp)
                tp = np.cumsum(tp)
                rec = tp / float(npos)
                prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
                mAP1[j] = voc_ap(rec, prec, True)
                if np.isnan((mAP1[j])):
                    all_map1[i] += 0.0
                else:
                    all_map1[i] += mAP1[j]

            else:
                mAP1[j] = 0
        all_map1[i] /= float(unique_label.shape[0])
        for j in range(unique_label.shape[0]):
            all_pos_indices = np.where(label[:, 1] == unique_label[j])
            easy_indices = np.where(label[all_pos_indices[0][:], 7] == 0)
            npos = easy_indices[0].size
            indices2 = np.where(eval2[:, 1] == unique_label[j])
            if indices2[0].size and (npos != 0):
                BB = eval2[indices2[0][:], :]
                sort_ind = np.argsort(-BB[:, 2])
                tp = BB[sort_ind, 3]
                fp = BB[sort_ind, 4]
                fp = np.cumsum(fp)
                tp = np.cumsum(tp)
                rec = tp / float(npos)
                prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
                mAP2[j] = voc_ap(rec, prec, True)
                if np.isnan(mAP2[j]):
                    all_map2[i] += 0.0
                else:
                    all_map2[i] += mAP2[j]
            else:
                mAP2[j] = 0
        all_map2[i] /= float(unique_label.shape[0])
        print(i)

    # Write image-level mAP to files
    output = open("output_files/trainval/300_rfcn_map300", "w")
    pickle.dump(all_map1, output)
    output.close()
    output = open("output_files/trainval/300_rfcn_maprfcn", "w")
    pickle.dump(all_map2, output)
    output.close()
    output = open("output_files/trainval/300_rfcn_map300", "r")
    all_map1 = pickle.load(output)
    output.close()
    output = open("output_files/trainval/300_rfcn_maprfcn", "r")
    all_map2 = pickle.load(output)
    output.close()

    ratio = np.zeros([all_map1.shape[0], ])
    dif = np.zeros([all_map2.shape[0], ])
    for k in range(all_map1.shape[0]):
        ratio[k] = (all_map2[k] - all_map1[k]) / (all_map1[k] + all_map2[k] + np.finfo(np.float64).eps)
        dif[k] = all_map2[k] - all_map1[k]

    # Visualize using histograms
    plt.hist(dif, bins=100, range=(-1.0, 1.0), normed=0)
    plt.show()
    ids_dif_0 = np.squeeze(np.array(np.where((dif < 0.0))))
    ids_dif0 = np.squeeze(np.array(np.where((dif == 0.0))))
    ids_dif0_ = np.squeeze(np.array(np.where((dif > 0.0))))
    ids = np.squeeze(np.array(np.where((dif <= 0.0))))

    # Write image index
    output = open("output_files/trainval/inds_300_rfcn_dif.txt", "w")
    pickle.dump(ids, output)
    output.close()
    output = open("output_files/trainval/inds_300_rfcn_dif_0.txt", "w")
    pickle.dump(ids_dif_0, output)
    output.close()
    output = open("output_files/trainval/inds_300_rfcn_dif0.txt", "w")
    pickle.dump(ids_dif0, output)
    output.close()
    output = open("output_files/trainval/inds_300_rfcn_dif0_.txt", "w")
    pickle.dump(ids_dif0_, output)
    output.close()
