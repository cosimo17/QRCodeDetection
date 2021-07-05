import numpy as np


def restack_anchors(xs, ys, ws, hs):
    xs = np.split(xs, xs.shape[2], axis=2)
    ys = np.split(ys, ys.shape[2], axis=2)
    ws = np.split(ws, ws.shape[2], axis=2)
    hs = np.split(hs, hs.shape[2], axis=2)
    anchors = np.concatenate([xs[0], ys[0], ws[0], hs[0]], axis=3)
    for i in range(1, len(xs)):
        _anchor = np.concatenate([xs[i], ys[i], ws[i], hs[i]], axis=3)
        anchors = np.concatenate([anchors, _anchor], axis=2)
    return anchors


def gen_anchors(grids, anchor_ratios):
    """
    :param grids: list. [grid_width, grid_height]. Grids means the last output from network
    :param anchor_ratios: np.ndarray. [[anchor_w, anchor_h]*]
    :return: priori anchors
    """
    grid_w, grid_h = grids
    ys, xs = np.meshgrid(list(range(grid_w)), list(range(grid_h)))
    xs = xs * (1 / grid_w) + (1 / grid_w) * 0.5
    ys = ys * (1 / grid_h) + (1 / grid_h) * 0.5
    anchor_per_grid = len(anchor_ratios)
    ws = np.ones(shape=(grid_w, grid_h, anchor_per_grid, 1), dtype=np.float32)
    ws *= np.expand_dims(anchor_ratios[..., 0], 1)
    hs = np.ones(shape=(grid_w, grid_h, anchor_per_grid, 1), dtype=np.float32)
    hs *= np.expand_dims(anchor_ratios[..., 1], 1)
    xs = np.expand_dims(xs, axis=[2, 3])
    xs = np.tile(xs, [1, 1, anchor_per_grid, 1])
    ys = np.expand_dims(ys, axis=[2, 3])
    ys = np.tile(ys, [1, 1, anchor_per_grid, 1])
    anchors = restack_anchors(xs, ys, ws, hs)
    return anchors.astype(np.float32)


def test_generated_anchors():
    """
    Test case of this anchor generator
    """
    grid = [4, 4]
    anchor_rations = np.array([[1, 1], [2, 2]])
    generated_anchors = gen_anchors(grid, anchor_rations)

    anchors = np.zeros(shape=(4, 4, 2, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            anchors[i, j, 0, :] = np.array([i * 0.25 + 0.125, j * 0.25 + 0.125, 1, 1])
            anchors[i, j, 1, :] = np.array([i * 0.25 + 0.125, j * 0.25 + 0.125, 2, 2])
    assert np.allclose(generated_anchors, anchors)

    grid = [5, 5]
    anchor_rations = np.array([[1, 1], [2, 2], [0.6, 0.8]])
    generated_anchors = gen_anchors(grid, anchor_rations)

    anchors = np.zeros(shape=(5, 5, 3, 4), dtype=np.float32)
    for i in range(5):
        for j in range(5):
            anchors[i, j, 0, :] = np.array([i * 0.2 + 0.1, j * 0.2 + 0.1, 1, 1])
            anchors[i, j, 1, :] = np.array([i * 0.2 + 0.1, j * 0.2 + 0.1, 2, 2])
            anchors[i, j, 2, :] = np.array(
                [i * 0.2 + 0.1, j * 0.2 + 0.1, 0.6, 0.8])
    assert np.allclose(generated_anchors, anchors)


if __name__ == '__main__':
    test_generated_anchors()
