from __future__ import print_function
from caffe.model_libs import *
from google.protobuf import text_format
import pickle
from sklearn.externals import joblib
from liblinearutil import *
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import matplotlib.pyplot as plt
import numpy as np
import caffe, os, sys, cv2
from caffe.model_libs import *

###################### Making a video demo ##################

CLASSES = ('__background__',
           'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant'
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
           'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
           'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
           'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
           'sofa', 'pottedplant',
           'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster',
           'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

def vis_detections(im, class_name, dets, thresh=0.6, color=(255,255,255), tag='R-FCN'):
    """Visualize and Return Images with Label Tags"""

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return im
    h, w, _ = im.shape
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(im,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),color,2)

        cv2.putText(im,'{:s} {:.3f}'.format(class_name, score),(int(bbox[0]),int(bbox[1]-2)),
                    cv2.FONT_HERSHEY_COMPLEX,.70,color, 2)
        cv2.putText(im, '{:s}'.format(tag), (int(w/2-70), int(50)),
                    cv2.FONT_HERSHEY_COMPLEX, 1.70, color, 5)
    return im

def get_labelname(labelmap, labels):
    """Get Label Name from LabelMap"""

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

def extractTopK(A, k):
    '''
    Extract Top-K Proposals
    (either from Tiny-Yolo or simply the Basic Model (ex., ssd300)
    '''
    dim = 6
    conf = A[:, 1]
    index = np.argsort(-conf)
    if len(index) > k:
        output = A[index[0:k], 0:dim]
    else:
        o1 = A[index, 0:dim]
        o2 = np.zeros([k-A.shape[0] ,dim])
        output = np.concatenate((o1, o2), axis=0)

    # flatten
    instance = np.squeeze(np.reshape(output, [1, output.shape[0]*output.shape[1]]))
    return instance

def plot_rfcn(net, image_name, name):
    """Get Detection Results from R-FCN"""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)
    h, w ,_ = im.shape
    scores, boxes = im_detect(net, im)

    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # As we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue
        for i in inds:
            im = vis_detections(im, cls, dets, thresh=CONF_THRESH, color=(255,255,255))

    # Write detection results to folder skyfall_output/
    cv2.imwrite(os.path.join('skyfall_output', str(name) + '.jpg'), im)
    print('rfcn finished')
    plt.close()

def plot_ssd(net, img_path, transformer, image_resize, name):
    """Get Detection Results from SSD300 (SSD500 is also okay)"""

    net.blobs['data'].reshape(1, 3, image_resize, image_resize)
    image = caffe.io.load_image(img_path)
    im = cv2.imread(img_path)
    h, w, _ = image.shape
    CONF_THRESH = 0.6
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    # Forward pass.
    detections = net.forward()['detection_out']

    # Parse the outputs.
    det_label = detections[0, 0, :, 1]
    det_conf = detections[0, 0, :, 2]
    det_xmin = detections[0, 0, :, 3]
    det_ymin = detections[0, 0, :, 4]
    det_xmax = detections[0, 0, :, 5]
    det_ymax = detections[0, 0, :, 6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= CONF_THRESH]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    dets = np.zeros((top_conf.shape[0], 5))
    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = top_labels[i]
        dets[i, :4] = xmin, ymin, xmax, ymax
        dets[i, -1] = score
        im = vis_detections(im, label_name, dets, thresh=CONF_THRESH, color=(255, 255, 0), tag="SSD")

    cv2.imwrite(os.path.join('skyfall_output', str(name) + '.jpg'), im)
    print('ssd is finished')

##----------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Set your home path
    home_path = '/home/zhouhy/'

    # Set gpu id
    gpu_device_id = 0
    caffe.set_device(gpu_device_id)
    caffe.set_mode_gpu()

    # Read path of images
    # Note that you should first split the video into frames
    img_path = []
    home_path = os.path.join(home_path, 'data/VOCdevkit/')

    f = open('skyfall_path.txt', 'r') # 'skyfall_path.txt' stores frames
    done = 0
    while not done:
      line = f.readline().strip('\n')
      if(line != ''):
        img_path.append(line)
      else:
        done = 1

    # Load MSCOCO labels
    labelmap_file = '/home/zhouhy/caffe_ssd/data/coco/labelmap_coco.prototxt'
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

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

    # Load Detectors
    rfcn_prototxt = 'test_agnostic.prototxt'
    rfcn_weights = 'resnet101_rfcn_ohem_iter_160000.caffemodel'

    ssd_prototxt = "ssd300_deploy.prototxt"
    rfcn_weights = "VGG_coco_SSD_300x300_iter_240000.caffemodel"

    ssd = caffe.Net(ssd_prototxt,      # defines the structure of the model
                    rfcn_weights,  # contains the trained weights
                    caffe.rfcn_prototxt)     # use test mode (e.g., don't perform dropout)

    rfcn = caffe.Net(rfcn_prototxt, rfcn_weights, caffe.TEST)

    # Define transformer
    transformer = caffe.io.Transformer({'data': ssd.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104,117,123])) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    # Load Adaptive Feeding Classifier (trained on MSCOCO)
    clf = joblib.load('iccv_exp2/300_rfcn_skyfall_balance.m')

    # Load Proposal Files from Tiny-YOLO (using pickle to load)
    num_classes = 20
    dim = 6
    fid = open("detection_skyfall_top50.txt", "r")
    fv = pickle.load(fid)
    fid.close()
    topK = 25
    ofv = fv[:, 0:dim*topK]

    dim_2 = 5 # remove the class num stored in ofv
    nfv = np.zeros([ofv.shape[0], num_classes+dim_2*topK])

    # Organize detection results into hand-crafted features
    
    for i in range(fv.shape[0]):
        for j in range(topK):
            if int(ofv[i,dim*j]) != 0:
                nfv[i, int(ofv[i, dim*j] - 1)] += 1
            nfv[i, dim*j+num_classes] = ofv[i, dim*j+1]
            nfv[i, (dim*j+dim-(dim_2-1)):(dim*j+num_classes+dim)] = ofv[i,(dim*j+2)], ofv[i,(dim*j+3)], \
                                                      ofv[i,(dim*j+4)]-ofv[i,(dim*j+2)], ofv[i,(dim*j+5)]-ofv[i,(6*j+3)]
    fv = nfv

    num_frames = 1969
    for i in range(num_frames):
        ypred = np.array(clf.predict(np.reshape(fv[i,:], [1, num_classes+dim*topK])))
        if ypred == 1:
            plot_rfcn(rfcn, img_path[i], i)
        else:
            image_resize = 300
            plot_ssd(ssd, img_path[i], transformer, image_resize, i)

        print('Frame {:d} is done\n'.format(i))