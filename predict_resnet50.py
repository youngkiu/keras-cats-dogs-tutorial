"""
This script goes along my blog post:
'Keras Cats Dogs Tutorial' (https://jkjung-avt.github.io/keras-tutorial/)
"""


import argparse
import glob
import os
import sys

import numpy as np
from keras import backend as K
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image


def parse_args():
    """Parse input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    return args


def get_files(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*'))
    elif path.find('*') > 0:
        files = glob.glob(path)
    else:
        files = [path]

    files = [f for f in files if f.endswith('JPG') or f.endswith('jpg')]

    if not len(files):
        sys.exit('No images found by the given path!')

    return files


if __name__ == '__main__':
    args = parse_args()
    files = get_files(args.path)
    cls_list = ['cats', 'dogs']

    # load the trained model
    net = load_model('model-resnet50-final.h5')

    # loop through all files and make predictions
    for i, f in enumerate(files):
        img = image.load_img(f, target_size=(224, 224))
        if img is None:
            continue
        x = image.img_to_array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        pred = net.predict(x)[0]
        pred_prob = pred[0]
        pred_cls = cls_list[int(round(pred_prob))]
        if os.path.basename(f)[:3] != pred_cls[:3]:
            print(i, f)
            print('    {:.3f}  {}'.format(pred_prob, pred_cls))
