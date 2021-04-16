import os
import numpy as np
import cv2

from skimage import transform
from imageio import imread, imwrite

import matplotlib.pyplot as plt

from tqdm import tqdm

def preprocess_celeba_dataset(dataset_dir):

    src_dir = os.path.join(dataset_dir, 'images')
    tar_dir = os.path.join(dataset_dir, 'images-preprocessed')

    if not os.path.isdir(tar_dir):
        os.makedirs(tar_dir)

    lmarks_file = os.path.join(dataset_dir, 'list_landmarks_align_celeba.txt')

    with open(lmarks_file, mode='r') as f:
        lines = f.readlines()[2:]
        lines = [list(filter(lambda x: x!='', line.replace('\n', '').split(" "))) for line in lines]
        lmarks = [np.array([int(coord) for coord in line[1:]]).reshape((5, 2)) for line in lines]

    mean_lmarks = np.mean(np.stack(lmarks), axis=0)

    for img_file in tqdm(os.listdir(src_dir)):
        load_path = os.path.join(src_dir, img_file)

        img = imread(load_path)
        warped = preprocess_with_similarity_transform(img, lmarks=mean_lmarks)

        imwrite(os.path.join(tar_dir, img_file), warped)


def preprocess_with_similarity_transform(img, lmarks, target_size=(112, 112)):

    # target lmarks in 112x112 image (taken from arcface repository)
    target_lmarks = np.array([
          [38.2946, 51.6963],
          [73.5318, 51.5014],
          [56.0252, 71.7366],
          [41.5493, 92.3655],
          [70.7299, 92.2041]
      ], dtype=np.float32 )

    # define the affine similarity transformation
    tform = transform.SimilarityTransform()
    tform.estimate(lmarks, target_lmarks)
    M = tform.params[0:2,:]

    # apply transformation to img
    img = cv2.warpAffine(img, M, target_size, borderValue = 0.0)

    return img


preprocess_celeba_dataset("E:\GitHub\pt-arcface-magface\datasets\celeba")
