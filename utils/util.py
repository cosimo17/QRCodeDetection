import numpy as np
from .anchor_generator import gen_anchors
import json
import tensorflow as tf
np.seterr(divide='raise')


def load_anchors(anchor_file):
    with open(anchor_file, 'r') as f:
        anchors = json.load(f)['anchors']
    return np.array(anchors)


def general_iou(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    left = np.max([xmin1, xmin2])
    right = np.min([xmax1, xmax2])
    top = np.max([ymin1, ymin2])
    bottom = np.min([ymax1, ymax2])
    iw = np.max([(right - left), 0])
    ih = np.max([(bottom - top), 0])
    si = iw * ih
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    _iou = si / (s1 + s2 - si)
    return _iou


def cxcy2xyxy(bbox):
    cx, cy, w, h = bbox
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = xmin + w
    ymax = ymin + h
    return xmin, ymin, xmax, ymax


def iou(bbox1, bbox2):
    """
    :param bbox1: np.ndarray. [gridw, gridh, anchor_per_grid, 4]
    :param bbox2: np.ndarray. [n, 4]
    box format: cx, cy, w, h.normalized to 0~1
    :return: ious. [n, gridw, gridh, anchor_per_grid, 1]
    """
    cx1, cy1, w1, h1 = np.split(bbox1, 4, axis=-1)
    xmin1 = cx1 - w1 / 2
    ymin1 = cy1 - h1 / 2
    xmax1 = xmin1 + w1
    ymax1 = ymin1 + h1

    cx2, cy2, w2, h2 = np.split(bbox2, 4, axis=-1)
    xmin2 = cx2 - w2 / 2
    ymin2 = cy2 - h2 / 2
    xmax2 = xmin2 + w2
    ymax2 = ymin2 + h2

    ious = np.zeros(shape=(len(cx2), cx1.shape[0], cx1.shape[1], cx1.shape[2], 1), dtype=np.float32)
    for i in range(len(cx2)):
        left = np.maximum(xmin1, xmin2[i])
        right = np.minimum(xmax1, xmax2[i])
        top = np.maximum(ymin1, ymin2[i])
        bottom = np.minimum(ymax1, ymax2[i])
        iw = np.maximum((right - left), 0)
        ih = np.maximum((bottom - top), 0)
        s1 = w1 * h1
        s2 = w2[i] * h2[i]
        _iou = iw * ih / (s1 + s2 - (iw * ih))
        ious[i] = _iou
    return ious


def postprocess(scores, classes, bboxes, score_threshold=0.5, selected_cls=1, iou_threshold=0.6):
    scores = scores.flatten()
    classes = np.reshape(classes, [-1, 2])
    bboxes = np.reshape(bboxes, [-1, 4])
    idx = scores >= score_threshold

    scores = scores[idx]
    classes = classes[idx]
    bboxes = bboxes[idx]

    idx = classes[..., 1] > classes[..., 0]
    scores = scores[idx]
    classes = classes[idx]
    bboxes = bboxes[idx]

    idx = tf.image.non_max_suppression(
        bboxes, scores, max_output_size=10)
    scores = scores[idx.numpy()]
    classes = classes[idx.numpy()]
    bboxes = bboxes[idx.numpy()]
    return scores, bboxes


def sigmoid(x):
    x = np.clip(x, -15.0, 15.0)
    return 1 / (1 + np.exp(-x))


def decode(anchors, output):
    """
    decode anchor relative offset to bbox coordinate
    output: np.ndarray. output from network. [n, w, h, anchor_per_grid, 7]
    """
    scores, cls_conf, bbox = output[..., 0], output[..., 1:3], output[..., 3:]
    scores = sigmoid(scores)
    cls_conf = sigmoid(cls_conf)
    bbox[..., :2] = np.tanh(bbox[..., :2])  # xw
    bbox[..., 2:] = np.tanh(bbox[..., 2:])  # wh
    tx, ty, tw, th = np.split(bbox, 4, axis=-1)
    anchor_cx, anchor_cy, anchor_w, anchor_h = np.split(anchors, 4, axis=-1)
    cx = tx * anchor_w + anchor_cx
    cy = ty * anchor_h + anchor_cy
    w = np.exp(tw) * anchor_w
    h = np.exp(th) * anchor_h
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = xmin + w
    ymax = ymin + h
    bboxes = np.concatenate([xmin, ymin, xmax, ymax], axis=-1)
    return scores, cls_conf, bboxes


def encode(anchor, bbox):
    """
    encode bbox coordinate to anchor relative offset
    reference: https://pjreddie.com/media/files/papers/YOLOv3.pdf &&
               https://github.com/tensorflow/models/blob/master/research/object_detection/box_coders/faster_rcnn_box_coder.py
    """
    anchor_cx, anchor_cy, anchor_w, anchor_h = anchor
    cx, cy, w, h = bbox
    tx = (cx - anchor_cx) / anchor_w  # -0.5 ~ 0.5
    ty = (cy - anchor_cy) / anchor_h  # -0.5 ~ 0.5
    tw = np.log(w / anchor_w)
    th = np.log(h / anchor_h)
    return np.array([tx, ty, tw, th])


def bbox2yololabel(bboxs, grids, anchor_ratios, class_number=2):
    """
    bboxs: np.ndarray. [n,4] [[cx, cy, w,h]*]. bboxs are normalized to 0~1
    """
    channel = len(anchor_ratios) * (1 + class_number + 4)
    labels = np.zeros(shape=(grids[0], grids[1], len(anchor_ratios), channel // len(anchor_ratios)), dtype=np.float32)
    labels[..., 1] = np.array([1.0])
    anchors = gen_anchors(grids, anchor_ratios)
    ious = iou(anchors, bboxs)
    for i in range(ious.shape[0]):
        index = np.unravel_index(ious[i].argmax(), ious[i].shape)[:-1]
        # assign label to the anchor whose iou is the max one
        encoded_bbox = encode(anchors[index], bboxs[i])
        scores = np.array([1.0])
        cls_confidence = np.array([0.0, 1.0])
        label = np.concatenate([scores, cls_confidence, encoded_bbox], axis=0)  # [score, cls_socre, bbox]
        labels[index] = label
    return labels.astype(np.float32)


def test_iou():
    anchors = [[0.5, 0.5, 1, 1], [1, 1, 2, 2]]
    boxs = [[0.5, 0.5, 1, 1], [1, 1, 2, 2], [1.5, 1.5, 1, 1]]
    anchors = np.array(anchors)
    anchors = np.expand_dims(anchors, axis=[0, 1])
    anchors = np.tile(anchors, [3, 3, 1, 1])
    boxs = np.array(boxs)
    _ious = iou(anchors, boxs)
    assert np.allclose(_ious[0, :, :, 0, :], [1])
    assert np.allclose(_ious[0, :, :, 1, :], [0.25])
    assert np.allclose(_ious[1, :, :, 0, :], [0.25])
    assert np.allclose(_ious[1, :, :, 1, :], [1])
    assert np.allclose(_ious[2, :, :, 0, :], [0])
    assert np.allclose(_ious[2, :, :, 1, :], [0.25])


if __name__ == '__main__':
    test_iou()
