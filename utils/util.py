import numpy as np


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
        iw = np.abs(right - left)
        ih = np.abs(bottom - top)
        s1 = w1 * h1
        s2 = w2[i] * h2[i]
        _iou = iw * ih / (s1 + s2 - (iw * ih))
        ious[i] = _iou
    return ious


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
