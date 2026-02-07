import numpy as np
import os
import glob
import tqdm
import cv2
import torch
import scipy.io as sio


def start_points(size, split_size, overlap):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


def crop_image_mask(image_dir, path, l):
    p = sio.loadmat(path)["Vol"]
    p[p < 0] = 0
    p = p / np.max(p)
    X_point = start_points(p.shape[0], 256, 0.0)
    Z_points = start_points(p.shape[2], 8, 0.0)
    for i in X_point:
        for j in X_point:
            for k in Z_points:
                rh = 1 + 0.1 * (2 * np.random.rand(1) - 1)
                new_image = torch.tensor(p[i:i + 256, j:j + 256, k:k + 8]).numpy()
                new_image = new_image.astype(np.float32) * rh
                if new_image.shape == (256, 256, 8):
                    if p.shape[0] == p.shape[1] == 256:
                        center = (128, 128)
                        ang = 180 * np.random.rand(1)[0].astype(np.float32)
                        rotation_matrix = cv2.getRotationMatrix2D(center, ang, 1)
                        for n in range(8):
                            new_image[:, :, n] = cv2.warpAffine(new_image[:, :, n], rotation_matrix, (256, 256))
                    if np.max(new_image) > 0.25:
                        np.save("{}.npy".format(image_dir+"%s"%l), new_image)
                        l += 1


if __name__ == '__main__':
    train_dir = 'training_set/'
    os.makedirs(train_dir,exist_ok=True)
    files = glob.glob(os.path.join("datasetsV/", '*.mat'))
    crop_dir = os.path.join(train_dir)
    l = 0
    os.makedirs(crop_dir, exist_ok=True)
    for path in tqdm.tqdm(files, desc='Cropping Potential Vol'):
        print(path)
        crop_image_mask(crop_dir, path, l)
