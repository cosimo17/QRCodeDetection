import cv2
import numpy as np
import os
import tensorflow as tf
from functools import partial
from utils import util


def preprocess_img(imgname):
    """
    Load img via cv2
    """
    img = cv2.imread(imgname.numpy().decode()).astype(np.float32)
    img /= 255.0
    return img


def parse_label(txtname):
    """
    Load bbox from txtfile
    """
    labels = []
    with open(txtname.numpy().decode(), 'r') as f:
        lines = f.readlines()
    for l in lines:
        l = l.split(',')[:4]
        labels.append(l)
    labels = np.array(labels).astype(np.float32)
    return labels


def tf_preprocess_img(filename):
    img = None
    [img, ] = tf.py_function(preprocess_img, [filename], [tf.float32])
    return img


def tf_preprocess_label(filename):
    label = None
    [label, ] = tf.py_function(parse_label, [filename], [tf.float32])
    return label


def yolo_label(bbox, grids, anchor_ratios, class_number):
    return util.bbox2yololabel(bboxs=bbox,
                               grids=grids,
                               anchor_ratios=anchor_ratios,
                               class_number=class_number)


def tf_create_yolo_label(bbox, grids, anchor_ratios, class_number):
    [label, ] = tf.py_function(yolo_label, [bbox, grids, anchor_ratios, class_number], [tf.float32])
    return label


def tf_dataset(root_dir, grids, anchor_ratios, class_number):
    """
    Build training dataset pipeline
    """
    list_ds = tf.data.Dataset.list_files(root_dir + '/*.jpg')
    imgs_ds = list_ds.map(tf_preprocess_img)
    list_ds = tf.data.Dataset.list_files(root_dir + '/*.txt')
    label_ds = list_ds.map(tf_preprocess_label)
    label_ds = label_ds.map(
        partial(tf_create_yolo_label, grids=grids, anchor_ratios=anchor_ratios, class_number=class_number))
    training_dataset = tf.data.Dataset.zip((imgs_ds, label_ds))
    return training_dataset


def test_tf_dataset():
    root_dir = '../test'
    tf_dataset(root_dir)


if __name__ == '__main__':
    test_tf_dataset()
