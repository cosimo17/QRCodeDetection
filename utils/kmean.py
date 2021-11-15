from sklearn.cluster import KMeans
import numpy as np
import argparse
import json
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='path of dataset')
    parser.add_argument('--n_clusters', '-n', type=int, default=6)
    parser.add_argument('--output', '-o', type=str, default='anchors.json')
    args = parser.parse_args()
    return args

def load_data(root_dir):
    dataset = []
    filenames = [name for name in os.listdir(root_dir) if name.endswith('.txt')]
    filenames = [os.path.join(root_dir, name) for name in filenames]
    for txt in filenames:
        with open(txt, 'r') as f:
            lines = f.readlines()
        for l in lines:
            l = l.split(',')
            w,h = l[2:4]
            w = float(w)
            h = float(h)
            dataset.append([w,h])
    return dataset

def mean_iou(shape1, shape2):
    w1, h1 = shape1[...,0], shape1[...,1]
    w2, h2 = shape2
    s1 = w1 * h1
    s2 = w2 * h2
    iou = np.minimum(s1, s2) / np.maximum(s1, s2)
    return iou

def main():
    args = get_args()
    dataset = load_data(args.root_dir)
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0, max_iter=500).fit(dataset)
    anchors = kmeans.cluster_centers_
    print(anchors)
    anchors = anchors.tolist()
    anchors = {
        'anchors': anchors
    }
    print("Save kmean anchors to {}".format(args.output))
    with open(args.output, 'w') as f:
        json.dump(anchors, f, indent=2)

if __name__ == '__main__':
    main()
