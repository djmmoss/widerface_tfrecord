#!/usr/bin/env python3
# coding:utf-8
import hashlib
import os

import cv2
import tensorflow as tf
import numpy as np

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def read_examples_list(path):
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/height': tf.FixedLenFeature((), tf.int64, 1),
            'image/width': tf.FixedLenFeature((), tf.int64, 1),
            'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/source_id': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
            'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
            'image/object/class/text': tf.VarLenFeature(tf.string),
            'image/object/class/label': tf.VarLenFeature(tf.int64)
        }
    )
    image = tf.image.decode_jpeg(features['image/encoded'])
    label = tf.cast(features['image/object/class/label'], tf.int32)
    xmin = features['image/object/bbox/xmin']
    xmax = features['image/object/bbox/xmax']
    ymin = features['image/object/bbox/ymin']
    ymax = features['image/object/bbox/ymax']
    return image, label, xmin, ymin, xmax, ymax


def parse_sample(filename, image_dir, f):
    """

    :param f:
    :return:
    """
    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)
    poses = []
    truncated = []

    # filename = f.readline().rstrip()
    print(filename)
    filepath = os.path.join(image_dir, filename)
    print(filepath)
    image_raw = cv2.imread(filepath)
    height, width, channel = image_raw.shape
    print("height is %d, width is %d, channel is %d" % (height, width, channel))


    with open(filepath, 'rb') as ff:
        encoded_image_data = ff.read()

    key = hashlib.sha256(encoded_image_data).hexdigest()
    face_num = np.max([int(f.readline().rstrip()), 1])
    valid_face_num = 0

    for i in range(face_num):
        annot = f.readline().rstrip().split()
        # WIDER FACE DATASET CONTAINS SOME ANNOTATIONS WHAT EXCEEDS THE IMAGE BOUNDARY
        if float(annot[2]) > 25.0:
            if float(annot[3]) > 30.0:
                xmins.append(max(0.005, (float(annot[0]) / width)))
                ymins.append(max(0.005, (float(annot[1]) / height)))
                xmaxs.append(min(0.995, ((float(annot[0]) + float(annot[2])) / width)))
                ymaxs.append(min(0.995, ((float(annot[1]) + float(annot[3])) / height)))
                classes_text.append('face'.encode('utf8'))
                classes.append(1)
                poses.append("front".encode('utf8'))
                truncated.append(int(0))
                print(xmins[-1], ymins[-1], xmaxs[-1], ymaxs[-1], classes_text[-1], classes[-1])
                valid_face_num += 1

    print("Face Number is %d" % face_num)
    print("Valid face number is %d" % valid_face_num)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(int(height)),
        'image/width': dataset_util.int64_feature(int(width)),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(int(0)),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))

    return valid_face_num, tf_example


def wider2tfrecord(path, image_dir, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    with open(path) as f:
        # WIDER FACE DATASET ANNOTATED 12880 IMAGES
        valid_image_num = 0
        invalid_image_num = 0
        # each picture start with filename, use for loop to get filename, other arg use readline fun to read
        for filename in f:
            filename = filename.strip()
            valid_face_number, tf_example = parse_sample(filename, image_dir, f)
            if valid_face_number != 0:
                writer.write(tf_example.SerializeToString())
                valid_image_num += 1
            else:
                invalid_image_num += 1
                print("Pass!")
    writer.close()

    print("Valid image number is %d" % valid_image_num)
    print("Invalid image number is %d" % invalid_image_num)


if __name__ == '__main__':
    # Parse Training Set
    annot = 'wider_face_split/wider_face_train_bbx_gt.txt'
    out = 'dataset/wider_face_train.record'
    wider2tfrecord(annot, "WIDER_train/images", out)

    # Parse Validation Set
    annot = 'wider_face_split/wider_face_val_bbx_gt.txt'
    out = 'dataset/wider_face_val.record'
    wider2tfrecord(annot, "WIDER_val/images", out)
