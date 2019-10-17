import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from torch.nn import functional as F

from utils import timer, create_circular_mask, make_gaussian


@tf.function
def cnn2d_tf(image: np.ndarray, filters: np.ndarray):
  features_tf = tf.nn.conv2d(image, filters, strides=[1, 1, 1, 1],
                             padding='SAME')
  return features_tf


def convert_to_torch(image, filters):
  image_torch = torch.tensor(image.transpose([0, 3, 2, 1]))
  filters_torch = torch.tensor(filters.transpose([3, 2, 1, 0]))

  return image_torch, filters_torch


def cnn2d_torch(image: np.ndarray, filters: np.ndarray):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  image_torch, filters_torch = convert_to_torch(image, filters)
  image_torch, filters_torch = image_torch.to(device), filters_torch.to(device)
  df, _, cin, cmul = filters.shape
  features_torch = F.conv2d(image_torch, filters_torch, padding=df // 2)
  features_torch_ = features_torch.cpu().numpy().transpose([0, 3, 2, 1])

  return features_torch_


def main():
  print('TensorFlow version %s' % str(tf.__version__))
  print('PyTorch version %s' % str(torch.__version__))

  # allowing soft growth of GPU in tensorflow
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  # create an image of circular masks
  g1 = create_circular_mask(21, 21, radius=4)
  g2 = create_circular_mask(21, 21, radius=5)
  g3 = create_circular_mask(21, 21, [3, 3], radius=3)
  image = np.stack([g1, g2, g3], axis=-1)[None]
  # create kernel to convolve image with
  gauss_kernel = np.stack([make_gaussian(5, 1)] * 3, axis=-1)[..., None]

  # plot image and kernel
  fig, ax = plt.subplots(nrows=1, ncols=2)
  ax[0].imshow(image[0, ...])
  ax[1].imshow(gauss_kernel[..., 0])
  ax[0].set_title('Image shape %s' % str(image.shape))
  ax[1].set_title('Kernel shape %s' % str(gauss_kernel.shape))
  plt.show()

  convolved_tf = cnn2d_tf(image, gauss_kernel)
  convolved_torch = cnn2d_torch(image, gauss_kernel)

  # plot tf and torch convolutions
  fig, ax = plt.subplots(nrows=1, ncols=2)
  ax[0].imshow(convolved_tf[0, ..., 0])
  ax[1].imshow(convolved_torch[0, ..., 0])
  ax[0].set_title('TF2 Conv shape %s' % str(convolved_tf.shape))
  ax[1].set_title('PyTorch Conv shape %s' % str(convolved_torch.shape))
  plt.show()

  iters = 20000
  start_time = time.time()
  for i in range(iters):
    cnn2d_tf(image, gauss_kernel)
  print("Time usage conv2d TF2: %s" % str(timer(start_time, time.time())))

  start_time = time.time()
  for i in range(iters):
    cnn2d_torch(image, gauss_kernel)
  print("Time usage conv2d PyTorch: %s" % str(timer(start_time, time.time())))


if __name__ == "__main__":
  main()