from models.yolov3 import yolov3
import numpy as np
import cv2
from utils.anchor_generator import gen_anchors
import utils.util as util
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help='test image')
    parser.add_argument('--weight', '-w', type=str, help='h5 weight file')
    parser.add_argument('--shape', '-s', type=str, default='(256,256)',
                        help='input shape. It should be equal with training shape')
    parser.add_argument('--anchors', '-a', type=str, default='anchors.json',
                        help='anchors generated from kmean algorithm')
    parser.add_argument('--output', '-o', type=str, default='', help='output image')
    args = parser.parse_args()
    args.shape = eval(args.shape)
    return args


def load_test_img(name):
    img = cv2.imread(name)
    src_img = img.copy()
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return src_img, img


def draw_roi(img, scores, bboxes, name='qrcode'):
    h, w = img.shape[:2]
    label_w = 46
    label_h = 18
    bbox_color = (240, 146, 31)
    label_roi_color = np.array([192, 219, 103])
    label_text_color = (255, 255, 255)
    for score, bbox in zip(scores, bboxes):
        xmin, ymin, xmax, ymax = bbox
        xmin = int(xmin * w)
        ymin = int(ymin * h)
        xmax = int(xmax * w)
        ymax = int(ymax * h)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), bbox_color, 2)
        img[ymin - label_h:ymin, xmin:xmin + label_w, :] = label_roi_color
        cv2.putText(img, str(name), (xmin, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_text_color, 1)
    return img


def main():
    args = get_args()
    model = yolov3(args.shape, anchor_number=6, weight=args.weight)
    test_img = args.input
    src_img, img = load_test_img(test_img)
    pred = model.predict(img)[0]
    anchors = util.load_anchors('./anchors.json')
    anchors = gen_anchors([8, 8], anchors)
    scores, classes, bboxes = util.decode(anchors, pred)
    scores, bboxes = util.postprocess(scores, classes, bboxes)
    src_img = draw_roi(src_img, scores, bboxes)
    cv2.imshow('qrcode_detection', src_img)
    cv2.waitKey(0)
    if args.output != '':
        cv2.imwrite(args.output, src_img)


if __name__ == '__main__':
    main()
