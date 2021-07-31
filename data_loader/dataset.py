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


def create_dataset(root_dir, grids, anchor_ratios, class_number, batch_size):
    """
    Build dataset pipeline
    """
    list_ds = tf.data.Dataset.list_files(root_dir + '/*.jpg', shuffle=False)
    imgs_ds = list_ds.map(tf_preprocess_img, num_parallel_calls=4)
    list_ds = tf.data.Dataset.list_files(root_dir + '/*.txt', shuffle=False)
    label_ds = list_ds.map(tf_preprocess_label, num_parallel_calls=4)
    label_ds = label_ds.map(
        partial(tf_create_yolo_label, grids=grids, anchor_ratios=anchor_ratios, class_number=class_number), num_parallel_calls=4)
    dataset = tf.data.Dataset.zip((imgs_ds, label_ds))
    # slice all data. 70% for training, 30% for validation
    training_dataset = dataset.take(int(len(dataset)*0.7)).prefetch(batch_size*10).shuffle(batch_size*10).batch(batch_size)
    val_dataset = dataset.skip(int(len(dataset)*0.7)).prefetch(batch_size*10).batch(batch_size)
    return training_dataset, val_dataset
