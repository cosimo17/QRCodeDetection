import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import tensorflow.keras as keras
import tensorflow as tf
from data_loader import dataset
from models.yolov3 import yolov3
from models.loss import yolo_loss
from utils.util import load_anchors

tf.get_logger().setLevel('WARNING')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, help='path to training dataset')
    parser.add_argument('--shape', type=str, default='(256,256)', help='input shape of network')
    parser.add_argument('--epoch', '-e', type=int, default=40)
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--anchors', '-a', type=str, default='anchors.json',
                        help='anchors generated from kmean algorithm')
    parser.add_argument('--weights', '-w', type=str, default='', help='pretrained weight')
    parser.add_argument('--output', '-o', type=str, default='yolo_qrcode.h5', help='output weight')
    parser.add_argument('--val_interval', '-i', type=int, default=1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    args = parser.parse_args()
    args.shape = eval(args.shape)
    return args


def scheduler(epoch, lr):
    if epoch % 5 == 1:
        return 0.001
    else:
        return lr * 0.7


def score_acc(yt, yp):
    pred_scores = tf.math.sigmoid(yp[..., 0])
    true_scores = yt[..., 0]
    pred_scores = tf.where(pred_scores > 0.5, 1, 0)
    acc = tf.reduce_mean(
        tf.cast(tf.math.equal(tf.cast(pred_scores, tf.float32), tf.cast(true_scores, tf.float32)), tf.float32))
    return acc


def cls_acc(yt, yp):
    pred_cls = yp[..., 1:3]
    pred_cls = tf.math.sigmoid(pred_cls)
    pred_cls = tf.math.argmax(pred_cls, axis=-1)

    true_cls = yt[..., 1:3]
    true_cls = tf.math.argmax(true_cls, axis=-1)
    acc = tf.metrics.categorical_accuracy(true_cls, pred_cls)
    return acc


def train():
    args = get_args()
    anchors = load_anchors(args.anchors)
    model = yolov3(input_shape=args.shape, anchor_number=len(anchors), weight=args.weights)
    model.compile(optimizer=keras.optimizers.Adam(args.learning_rate), loss=yolo_loss,
                  metrics=[score_acc, cls_acc])
    training_ds, val_ds = dataset.create_dataset(args.data_dir, [s // 32 for s in args.shape], anchor_ratios=anchors,
                                                 class_number=2, batch_size=args.batch_size)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    model.fit(x=training_ds, epochs=args.epoch, validation_data=val_ds, callbacks=[lr_callback, tensorboard_callback],
              validation_freq=5)
    model.save(args.output)


if __name__ == '__main__':
    train()
