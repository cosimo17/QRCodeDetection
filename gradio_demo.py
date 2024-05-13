import gradio as gr
from models.yolov3 import yolov3
import numpy as np
import cv2
from utils.anchor_generator import gen_anchors
import utils.util as util
from functools import partial

shape = (256, 256)
anchors = util.load_anchors('./anchors.json')
model = yolov3((256, 256), anchor_number=len(anchors), weight='yolo_qrcode.h5')
anchors = gen_anchors([s // 32 for s in (256, 256)], anchors)


def preprocess(img):
    src_img = img.copy()
    img = cv2.resize(img, shape)
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


def detection_info(bboxes, scores):
    infos = []
    temp = 'QRCode{}\n 置信度:{:.3f}\n xmin:{}, ymin:{}, xmax:{}, ymax:{}\n\n'
    for i in range(len(scores)):
        bbox = bboxes[i]
        score = scores[i]
        xmin, ymin, xmax, ymax = bbox
        xmin = int(xmin * 256)
        ymin = int(ymin * 256)
        xmax = int(xmax * 256)
        ymax = int(ymax * 256)
        infos.append(temp.format(i + 1, score, xmin, ymin, xmax, ymax))
    return ''.join(infos)


def detect(image):
    src_img, img = preprocess(image)
    pred = model.predict(img)[0]
    scores, classes, bboxes = util.decode(anchors, pred)
    scores, bboxes = util.postprocess(scores, classes, bboxes)
    src_img = draw_roi(src_img, scores, bboxes)
    return src_img, str(detection_info(bboxes, scores))


input_image = gr.Image()
output_image = gr.Image()
output_text = gr.Textbox()

demo = gr.Interface(
    fn=detect,
    inputs=input_image,
    outputs=[output_image, output_text],
)

demo.launch()
