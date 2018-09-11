### Adaptive Feeding: Achieving Fast and Accurate Detections by Adaptively Combining Object Detectors 

Code in Python 2.7 for paper in ICCV 2017.

**Note**

This code aims at reproducing the experimental results when combining R-FCN and SSD300 reported in the paper. If you want to replace R-FCN with SSD500, the procedure can be quite simple.

#### Step 1 

Since we use SSD and R-FCN in our experiments, it is necessary to merge branches in [Single Shot Multibox Detector](https://github.com/weiliu89/caffe/tree/ssd) and [R-FCN](https://github.com/YuwenXiong/py-R-FCN). You can either do this by yourself or simply compile our code on the basis of [Single Shot Multibox Detector](https://github.com/weiliu89/caffe/tree/ssd), which means you should download the SSD CAFFE BRANCH before running our code and configure the DATASET (especially Pascal VOC) following the instruction written by [Wei Liu](https://github.com/weiliu89/caffe/tree/ssd)

#### Step 2

In this step you may need to download our code and merge it into aforementioned caffe folder. Now your folder should be structured like this:

> caffe-ssd
>
> > cmake
> >
> > AF
> >
> > output_files
> >
> > data
> >
> > docker
> >
> > docs
> >
> > examples
> >
> > include
> >
> > matlab
> >
> > python
> >
> > models
> >
> > scripts
> >
> > src
> >
> > tools
> >
> > .......

Our code lies in *AF/* and we also provide indices of images in *output_files/*. Please do remember to download our [models](https://drive.google.com/open?id=1hM0ceZ9l-Spc0lKYDGvBVueKpXKGUk9g) and put them under *AF/*. Now you might have such a structure

> AF
>
> > cal_mAP
> >
> > indices
> >
> > R-FCN
> >
> > SSD300
> >
> > SVM
> >
> > tiny_yolo
> >
> > ......

#### Step 3

After configure the dataset path and caffe branch, you are able to run our code (besides, you might still need to modify some variables in the code, like *home_path* in all four python files)

```python
cd caffe-ssd/
# Get Easy vs. Hard split
python AF/get_Easy_Hard.py
# Train an AF classifier
python AF/train_AF.py
# Get the mAP of the combination
python AF/cal_mAP.py
```

For your convenience, we already provided extracted proposals from Tiny-YOLO (under AF/tiny_yolo). If you want to speed up the detection process, implementing the YOLO under caffe is necessary. Please refer to [caffe-YOLO](https://github.com/xingwangsfu/caffe-yolo).

#### Miscellaneous

In addition to the official experimental code, we also provided a file named *video_demo.py* under *AF/* in order to help you make a demo like [this](http://zhouhy.org/videos/skyfall_piece.mp4). To produce such a video, you need to follow the instructions in code and employ [ffmpeg](https://www.ffmpeg.org/) to organize frames. If you have any question, please feel free to contact me at whuzhouhongyu@gmail.com