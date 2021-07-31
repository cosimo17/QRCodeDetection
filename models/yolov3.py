import tensorflow.keras as keras
from tensorflow.keras.models import Model
import tensorflow as tf

def bn_act(x, act='relu'):
    acts = {'relu': keras.layers.ReLU,
            'leaky_relu': keras.layers.LeakyReLU,
            'swish': keras.activations.swish}
    x = keras.layers.BatchNormalization()(x)
    x = acts[act]()(x)
    return x

def head_layer(x, class_number=2, anchor_number=3):
    """
    Head layer for prediction.
    Reference: https://pjreddie.com/media/files/papers/YOLOv3.pdf
    """
    kernel = anchor_number * (1 + class_number + 4)
    output = keras.layers.Conv2D(kernel, (3,3), padding='SAME')(x)
    output = tf.reshape(output, [-1, output.shape[1], output.shape[2], anchor_number, output.shape[3]//anchor_number])
    return output

def downsize(x):
    """
    maxpool to downsize feature map
    """
    return keras.layers.MaxPool2D(padding='same')(x)

def yolov3(input_shape=(256,256), class_number=2, anchor_number=5, weight=''):
    """
    yolov3 like network. Not real yolov3.
    size: 256 -> 128 -> 64 -> 32 -> 16 -> 8
    kernel: 32 -> 64 -> 128 -> 256 -> 256
    """
    input_layer = keras.layers.Input(shape=input_shape + (3,))
    for _ in range(2):
        x = keras.layers.Conv2D(32, (3,3), padding='SAME')(input_layer)
        x = bn_act(x)
    # downsize
    x = downsize(x) # 128 x 128

    for _ in range(3):
        x = keras.layers.Conv2D(64, (3,3), padding='SAME')(x)
        x = bn_act(x)
    # downsize
    x = downsize(x) # 64 x 64

    for _ in range(4):
        x = keras.layers.Conv2D(128, (3,3), padding='SAME')(x)
        x = bn_act(x)
    # downsize
    x = downsize(x) # 32 x 32
    for _ in range(4):
        x = keras.layers.Conv2D(128, (3,3), padding='SAME')(x)
        x = bn_act(x)
    # downsize
    x = downsize(x) # 16 x 16
    # low level feature
    f1 = x

    for _ in range(4):
        x = keras.layers.Conv2D(128, (3,3), padding='SAME')(x)
        x = bn_act(x)

    # feature fusion
    x = keras.layers.Concatenate()([f1, x])

    # downsize via stride2 conv
    x = keras.layers.Conv2D(256, (3,3), strides=(2,2), padding='SAME')(x) # 8 x 8

    # 1x1 kernel to summary all channel
    x = keras.layers.Conv2D(256, (1,1), strides=(1,1))(x)

    output = head_layer(x, class_number=class_number, anchor_number=anchor_number)
    model = Model(inputs=input_layer, outputs=output)
    if weight != '':
        print('Load pretrained weight: {}'.format(weight))
        model.load_weights(weight)
    return model
