import os
import cv2
import argparse
from utils.util import *
from models.yolov3 import yolov3
import multiprocessing as mp
import numpy as np
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, required=True, help='path to training dataset')
    parser.add_argument('--shape', type=str, default='(256,256)', help='input shape of network')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--score_threshold', type=float, default=0.5)
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--anchors', '-a', type=str, default='anchors.json',
                        help='anchors generated from kmean algorithm')
    parser.add_argument('--weights', '-w', type=str, default='yolo_qrcode.h5', help='pretrained weight')
    args = parser.parse_args()
    args.shape = eval(args.shape)
    return args


def _load_img(name):
    img = cv2.imread(name)
    img = img.astype(np.float32) / 255.0
    return img


def _load_label(name):
    labelname = name.replace('.jpg', '.txt')
    with open(labelname, 'r') as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        line = line.split(',')[:4]  # cx,cy,w,h
        line = [float(v) for v in line]
        line = cxcy2xyxy(line)
        labels.append(line)
    labels = np.array(labels).astype(np.float32)
    return labels


def loader(root_dir, batch_size, cpu):
    imgnames = os.listdir(root_dir)
    imgnames = [name for name in imgnames if name.endswith('.jpg')]
    imgnames.sort()
    imgnames = imgnames[int(len(imgnames) * 0.7):]  # last 30% is validation dataset
    imgnames = [os.path.join(root_dir, name) for name in imgnames]
    indexes = np.arange(len(imgnames))
    indexes = indexes[:batch_size * (len(indexes) // batch_size)]  # drop last
    indexes = np.reshape(indexes, (-1, batch_size))
    pool = mp.Pool(cpu)
    for i in range(indexes.shape[0]):
        index = indexes[i]
        _imgnames = [imgnames[idx] for idx in index]
        imgs = pool.map(_load_img, _imgnames)
        labels = pool.map(_load_label, _imgnames)
        imgs = np.array(imgs).astype(np.float32)
        yield imgs, labels


def _metrics(pred_bboxes, true_boxes, iou_threshold=0.5):
    """
    pred_bboxes: np.ndarray. [n,4]. format: normalized | xmin,ymin,xmax,ymax.
    true_boxes: list. [m,4]. format: normalized | xmin,ymin,xmax,ymax.
    """
    TP = 0  # true positive
    TN = 0  # true negative
    FP = 0  # false positive
    FN = 0  # false negative
    IOU = 0
    used = [False for _ in range(len(pred_bboxes))]  # mask indicate the pred box has matched with gt box or not
    for i in range(len(true_boxes)):
        detected = False
        for j in range(len(pred_bboxes)):
            _iou = general_iou(true_boxes[i], pred_bboxes[j])
            if _iou > iou_threshold and not used[j]:
                TP += 1
                used[j] = True
                IOU += _iou
                detected = True
                break
        if not detected:
            FN += 1
    FP += (len(used) - sum(used))  # unmatched pred box. False positive pred.
    if TP > 0:
        mean_iou = IOU / TP
    else:
        mean_iou = 0
    return TP, FP, FN, mean_iou


def run():
    args = get_args()
    anchors = load_anchors(args.anchors)
    detecter = yolov3(input_shape=args.shape, anchor_number=len(anchors), weight=args.weights)
    anchors = gen_anchors([s // 32 for s in args.shape], anchors)
    TP, FP, FN = 0, 0, 0
    IOU = 0
    count = 0
    for imgs, gt_labels in loader(args.data_dir, args.batch_size, 3):
        print("Evaluating {}/{} sample".format(count, 12000))
        # Forward
        outputs = detecter.predict(imgs)  # [n,h,w,c]
        for i in range(len(outputs)):
            scores, classes, bboxes = decode(anchors, np.expand_dims(outputs[i], axis=0))
            pred_scores, pred_bboxes = postprocess(scores, classes, bboxes)
            tp, fp, fn, iou = _metrics(pred_bboxes, gt_labels[i])
            TP += tp
            FP += fp
            FN += fn
            IOU += iou
        count += len(imgs)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    mean_iou = IOU / count
    print("\n")
    print("--------------Evaluate Result-----------------")
    print("Model: {}".format(args.weights))
    print("score_threshold: {}".format(args.score_threshold))
    print("iou_threshold: {}".format(args.iou_threshold))
    print("Precision: {:.3f}  Recall: {:.3f}  MeanIOU: {:.3f}".format(precision, recall, mean_iou))


if __name__ == '__main__':
    run()

