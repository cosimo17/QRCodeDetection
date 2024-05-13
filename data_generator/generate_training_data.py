import os
import numpy as np
import cv2
import argparse
import imgaug.augmenters as iaa
import tqdm

MIN_W = 32
MIN_H = 32

aug_seq = iaa.Sequential([
    iaa.Crop(px=(0, 10)),
    iaa.GaussianBlur(sigma=(0.0, 4)),
    iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.5 * 255), per_channel=0.5)),
    iaa.Affine(
        scale={"x": (0.3, 0.5), "y": (0.3, 0.5)},
        rotate=(-25, 25),
        shear=(-15, 15)
    ),
    iaa.PerspectiveTransform(scale=(0.01, 0.1))
])


def augment_process_fg(image):
    image_aug = aug_seq.augment_image(image)
    return image_aug


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fg_dir', '-fg', type=str, help='path to foreground qrcode images')
    parser.add_argument('--bg_dir', '-bg', type=str, help='path to background images')
    parser.add_argument('--output', '-o', type=str, help='path to save the generated images')
    parser.add_argument('--number', '-n', type=int, help='how many images you want to generate')
    parser.add_argument('--size', '-s', type=str, default='(32,120)', help='size range of the qrcode image')
    parser.add_argument('--alpha', '-a', type=str, default='(10,30)', help='value range of the alpha parameter')
    parser.add_argument('--object_number', '-on', type=str, default='(1,5)',
                        help='the number of qrcode image in one background image')
    parser.add_argument('--shape', type=int, default=256, help='training data shape')
    parser.add_argument('--debug', type=bool, default=False, help='debug mode')
    args = parser.parse_args()
    args.size = eval(args.size) # string to tuple
    args.alpha = eval(args.alpha)
    args.object_number = eval(args.object_number)
    return args


class ImageLists(object):
    def __init__(self, root_dir, shape=None):
        imgnames = os.listdir(root_dir)
        imgnames = [os.path.join(root_dir, imgname) for imgname in imgnames]
        self.imgnames = imgnames
        self.shape = shape

    def __getitem__(self, item):
        item = item % len(self.imgnames)
        imgname = self.imgnames[item]
        img = cv2.imread(imgname)
        if self.shape is not None:
            img = cv2.resize(img, self.shape)
        return img

    def __len__(self):
        return len(self.imgnames)


def random_position(bg_size, fg_size):
    bg_w, bg_h = bg_size
    w, h = fg_size
    xmin = np.random.randint(0, bg_w - w)
    ymin = np.random.randint(0, bg_h - h)
    xmax = xmin + w
    ymax = ymin + h
    bbox = [xmin, ymin, xmax, ymax]
    return bbox


def random_resize(img, size_range):
    h, w = img.shape[:2]
    min_size, max_size = size_range
    new_w = np.random.randint(min_size, max_size)
    new_h = int(new_w * h / w)
    img = cv2.resize(img, (new_w, new_h))
    return img


def try_random_resize(fg_img, size_range):
    _fg_img = fg_img.copy()
    while True:
        fg_img = _fg_img.copy()
        # augment
        fg_img = augment_process_fg(fg_img)
        mask = np.argwhere(fg_img > 0)
        box = (np.min(mask[..., 0]),
               np.min(mask[..., 1]),
               np.max(mask[..., 0]),
               np.max(mask[..., 1]))
        if box[2] - box[0] < MIN_W or box[3] - box[1] < MIN_H:
            continue
        fg_img = fg_img[box[0]:box[2], box[1]:box[3], ...]
        fg_img = random_resize(fg_img, size_range)

        break
    return fg_img


def try_random_position(fg_img, bg_size, exist_bbox):
    fg_size = [fg_img.shape[1], fg_img.shape[0]]
    count = 0
    max_count = 20
    while True:
        if count > max_count:
            return None
        bbox = random_position(bg_size, fg_size)
        intersection = False
        for bx in exist_bbox:
            if is_overlap(bx, bbox):
                intersection = True
                break
        if not intersection:
            break
        count += 1
    return bbox


def is_overlap(box1, box2):
    if box1[0] > box2[2] or box2[0] > box1[2]:
        return False
    if box1[1] > box2[3] or box2[1] > box1[3]:
        return False
    return True


def paste(fg_img, bg_img, bbox, alpha):
    # crop qrcode image
    fg_img = fg_img[0:bbox[3] - bbox[1], 0:bbox[2] - bbox[0], ...]
    mask = np.nonzero(fg_img > np.random.randint(15, 50))
    bg_crop = bg_img[bbox[1]:bbox[3], bbox[0]:bbox[2], ...]
    # alpha fusion with random parameter
    alpha = np.random.randint(alpha[0], alpha[1]) / 100.0
    bg_crop[mask] = (bg_crop[mask] * alpha + fg_img[mask] * (1 - alpha)).astype(np.uint8)
    bg_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = bg_crop
    return bg_img


def normalize_coordinate(bbox, shape):
    """Convert absolute coordinates to relative coordinates"""
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    xmin /= shape[1]
    ymin /= shape[0]
    w /= shape[1]
    h /= shape[0]
    return xmin, ymin, w, h


def save_result(output_dir, img, count, labels):
    imgname = '{:06d}.jpg'.format(count)
    imgname = os.path.join(output_dir, imgname)
    cv2.imwrite(imgname, img)
    labels_name = imgname.replace('.jpg', '.txt')
    with open(labels_name, 'w') as f:
        for i in range(len(labels) - 1):
            f.write(labels[i] + '\n')
        f.write(labels[-1])


def visualize(img, bbox):
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    cv2.imshow('img', img)
    cv2.waitKey(0)


def generate_training_data(args):
    """
    Generate fake training data.
    1. select one background image
    2. select one or more qrcode images
    3. do some image augment for qrcode image (add noise, blur, affine ...)
    4. resize qrcode image to a random size
    5. paste qrcode image to a random location of background image(alpha Fusion)
    """
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    bg_imgs = ImageLists(args.bg_dir, [args.shape] * 2)
    fg_imgs = ImageLists(args.fg_dir)
    count = 0
    with tqdm.tqdm(total=args.number) as pbar:
        pbar.set_description('Generating {}/{} sample'.format(count, args.number))
        while True:
            if count >= args.number:
                break
            # get ackground image
            bg_img = bg_imgs[count]
            exist_bbox = []
            labels = []
            for i in range(np.random.randint(args.object_number[0], args.object_number[1])):
                # get qrcode image
                fg_img = fg_imgs[count]
                fg_img = try_random_resize(fg_img, args.size)
                bbox = try_random_position(fg_img, [bg_img.shape[1], bg_img.shape[0]], exist_bbox)
                if bbox is None:
                    continue
                synth_img = paste(fg_img, bg_img, bbox, args.alpha)
                exist_bbox.append(bbox)
                l, t, w, h = normalize_coordinate(bbox, bg_img.shape)
                cx = l + w / 2
                cy = t + h / 2
                if args.debug:
                    visualize(synth_img, bbox)
                # cx, cy, w, h, conf, cls
                one_label = '{},{},{},{},{},{}'.format(cx, cy, w, h, 1.0, 0)
                labels.append(one_label)
                bg_img = synth_img
            count += 1
            save_result(args.output, bg_img, count, labels)
            pbar.update(1)


def main():
    args = get_args()
    generate_training_data(args)


if __name__ == '__main__':
    main()
