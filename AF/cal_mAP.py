from __future__ import print_function
from caffe.model_libs import *
from google.protobuf import text_format
import pickle
from sklearn.externals import joblib
from liblinearutil import *
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import caffe, os, cv2
from caffe.model_libs import *
import argparse

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'ResNet-101': ('ResNet-101',
                  'resnet101_rfcn_final.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_final.caffemodel')}

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
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
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

    detection_out = np.reshape(np.array([None]*7), [1,7])
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
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

def ismember(A,B):
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
        o2 = np.zeros([k-A.shape[0] ,6])
        output = np.concatenate((o1, o2), axis=0)
    # flatten
    instance = np.squeeze(np.reshape(output, [1, output.shape[0]*output.shape[1]]))
    return instance

if __name__ == '__main__':
	home_path = '/home/zhouhy'

    CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    num_classes = 20

    # Load PASCAL VOC labels
    labelmap_file = os.path.join(home_path, 'data/VOC0712/labelmap_voc.prototxt')
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    gpu_device_id = 0
    caffe.set_device(gpu_device_id)
    caffe.set_mode_gpu()

    img_path = []
    img_id = []

    # Initialize all_eval and labels
    all_eval = np.reshape(np.array([None]*5), [1,1,1,5])
    labels = np.reshape(np.array([None]*8), [1,1,1,8])

    # Load VOC2007 test image paths
    f = open(os.path.join(home_path, 'data/VOC0712/test.txt'), 'r')
    done = 0
    while not done:
      line = f.readline().strip()
      if(line != ''):
        img_path.append(home_path + line.split(' ')[0])
      else:
        done = 1

    rfcn_prototxt = 'AF/R-FCN/test_agnostic.prototxt'
    rfcn_weights = 'AF/R-FCN/resnet101_rfcn_final.caffemodel'

    # SSD
    prototxt1 = "AF/SSD300/test.prototxt"
    prototxtm = "AF/cal_mAP/test.prototxt"

    weights1 = "AF/SSD300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel"
    weightsm = "AF/SSD300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel"

    num_test = 4952
    topK = 25
    ssd = caffe.Net(prototxt1, weights1, caffe.TEST)
    rfcn = caffe.Net(rfcn_prototxt, rfcn_weights, caffe.TEST)

    # To keep consistent between R-FCN and SSD
    netm = caffe.Net(prototxtm, weightsm, caffe.TEST)

    # Load proposals of VOC 2007 test images which were generated from Tiny-YOLO
    fid = open("AF/tiny_yolo/detection_2007test_top50_tiny_yolo.txt", "r")
    fv = pickle.load(fid)
    fid.close()
    topK = 25
    num_classes = 20
    use_weight = True
    ofv = fv[:, 0:dim*topK]

    dim = 5
    count = 0
    nfv = np.zeros([ofv.shape[0], num_classes+dim*topK])

    # Organize detection results into hand-crafted features
    for i in range(fv.shape[0]):
        for j in range(topK):
            if int(ofv[i,dim*j])!=0:
                nfv[i, int(ofv[i, dim * j] - 1)] += 1
            nfv[i, dim*j+num_classes] = ofv[i,dim*j+1]
            nfv[i, (dim * j + num_classes + dim - 4):(dim * j + num_classes + dim)] = ofv[i, (6 * j + 2)], ofv[i, (6 * j + 3)], \
                                                                    ofv[i, (6 * j + 4)] - ofv[i, (6 * j + 2)], ofv[
                                                                        i, (6 * j + 5)] - ofv[i, (6 * j + 3)]
    fv = nfv

    clf = joblib.load('AF/SVM/300_rfcn_VOC.m')

    all_eval = np.reshape(np.array([None]*5), [1,1,1,5])
    labels = np.reshape(np.array([None]*8), [1,1,1,8])
    all_map1 = np.zeros([num_test,])
    all_map2 = np.zeros([num_test,])
    count = 0

    # Evaluate the combination of R-FCN and SSD300
    for i in range(num_test):
        ypred = clf.predict(np.reshape(fv[i, :], [1, num_classes + dim * topK]))
        if np.array(ypred) == 1:
            proposals = extract(rfcn, img_path[i])
            netm.blobs['merged_proposals'].reshape(proposals.shape[0], proposals.shape[1], \
                                                   proposals.shape[2], proposals.shape[3])
            netm.blobs['merged_proposals'].data[...] = proposals

            netm.forward()
            labels = np.concatenate((labels, netm.blobs['label'].data), axis=2)
            detection_eval = netm.blobs['detection_eval'].data
            all_eval = np.concatenate((all_eval, detection_eval[:, :, 20:detection_eval.shape[2], :]), axis=2)

            count += 1
            print(count)
        else:
            proposals = np.ones([1, 1, 3, 7])
            netm.blobs['merged_proposals'].reshape(proposals.shape[0], proposals.shape[1], \
                                                   proposals.shape[2], proposals.shape[3])
            netm.blobs['merged_proposals'].data[...] = proposals
            netm.forward()
            labels = np.concatenate((labels, ssd.blobs['label'].data), axis=2)
            detection_eval = ssd.blobs['detection_eval'].data
            all_eval = np.concatenate((all_eval, detection_eval[:, :, 20:detection_eval.shape[2], :]), axis=2)
        print(i)

    labels = np.delete(labels, 0, axis = 2)
    labels = np.squeeze(labels)
    all_eval = np.delete(all_eval, 0, axis = 2)
    all_eval = np.squeeze(all_eval)
    mAP = np.zeros([num_classes,])
    for i in range(num_classes):
        all_pos_indices = np.where(labels[:, 1] == (i+1))
        easy_indices = np.where(labels[all_pos_indices[0][:], 7] == 0)
        npos = easy_indices[0].size
        indices = np.where(all_eval[:, 1] == (i+1))
        if indices[0].size:
            BB = all_eval[indices[0][:], :]
            sort_ind = np.argsort(-BB[:, 2])
            tp = BB[sort_ind, 3]
            fp = BB[sort_ind, 4]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            prec = tp / np.maximum(tp+fp, np.finfo(np.float64).eps)
            mAP[i] = voc_ap(rec, prec, True)
        else:
            mAP[i] = 0
    print('Final mAP: {:f}'.format(np.mean(mAP)))