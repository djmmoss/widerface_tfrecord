# widerface_tfrecord

Creates a TFRecord of the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) datasets for use in the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

## Prerequisites

```bash
tensorflow==1.15.2
numpy==1.16.0
```

## Steps

```bash
mkdir data
cd data

git clone https://github.com/djmmoss/widerface_tfrecord.git
pip install gdown

# Download and Unpack Training Set
gdown https://drive.google.com/uc?id=0B6eKvaijfFUDQUUwd21EckhUbWs
unzip WIDER_train.zip

# Download and Unpack Validation Set
gdown https://drive.google.com/uc?id=0B6eKvaijfFUDd3dIRmpvSk8tLUk
unzip WIDER_val.zip

# Downlad and Upack Annotations
wget http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip
unzip wider_face_split.zip

# Create TFRecord
python widerface_tfrecord/wider2tfrecord.py
```
## Citation

Please cite the original dataset authors:

```bash
@inproceedings{yang2016wider,
Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
Title = {WIDER FACE: A Face Detection Benchmark},
Year = {2016}}
```

The majority of this code is an updated and simplified version of:
```bash
https://github.com/yeephycho/widerface-to-tfrecord
https://github.com/aodiwei/Tensorflow-object-detection-API-for-face-detcetion
```

