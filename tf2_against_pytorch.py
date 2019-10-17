import numpy as np
import tensorflow as tf
import torch
from torch.nn import functional as F


def createCircularMask(h, w, center=None, radius=None):
  if center is None:  # use the middle of the image
    center = [int(w / 2), int(h / 2)]
  if radius is None:  # use the smallest distance between the center and image walls
    radius = min(center[0], center[1], w - center[0], h - center[1])

  Y, X = np.ogrid[:h, :w]
  dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

  mask = dist_from_center <= radius
  return mask * 1.0


def makeGaussian(size, sigma=3, center=None):
  """ Make a square gaussian kernel.

  size is the length of a side of the square
  fwhm is full-width-half-maximum, which
  can be thought of as an effective radius.
  """

  x = np.arange(0, size, 1, float)
  y = x[:, np.newaxis]

  if center is None:
    x0 = y0 = size // 2
  else:
    x0 = center[0]
    y0 = center[1]

  return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2))
  # return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def check_shape_image(image):
  if len(image.shape) == 2:
    return image[np.newaxis, ..., np.newaxis]
  elif len(image.shape) == 3 and image.shape[-1] != image.shape[-2]:
    return image[np.newaxis, ...]
  elif len(image.shape) == 3 and image.shape[0] != image.shape[1]:
    return image[..., np.newaxis]
  return image


def check_shape_kernel(kernel, x):
  if len(kernel.shape) == 2:
    kernel = np.stack([kernel] * x.shape[-1], axis=-1)
    return kernel[..., np.newaxis]
  elif len(kernel.shape) == 3:
    return kernel[..., np.newaxis]
  return kernel

def cnn2d_depthwise_tf(image: np.ndarray,
    filters: np.ndarray):
  features_tf = tf.nn.depthwise_conv2d(image[None], filters,
                                       strides=[1, 1, 1, 1], padding='SAME')

  return features_tf[0]

@tf.function
def cnn2d_tf(image: np.ndarray,
    filters: np.ndarray):
  features_tf = tf.nn.conv2d(image[None], filters, strides=[1, 1, 1, 1],
                             padding='SAME')

  return features_tf[0]

@tf.function
def cnn2d_depthwise_tf_transpose(image: np.ndarray,
    filters: np.ndarray):
  image = tf.transpose(image, perm=[2, 1, 0])[None]
  features_tf = tf.nn.depthwise_conv2d(image, filters,
                                       strides=[1, 1, 1, 1], padding='SAME',
                                       data_format="NCHW")

  return tf.transpose(features_tf[0], perm=[2, 1, 0])

def convert_to_torch(image, filters):
  image_torch = torch.tensor(image.transpose([2, 1, 0])[None])
  filters_torch = torch.tensor(filters.transpose([3, 2, 1, 0]))

  return image_torch, filters_torch


def cnn2d_depthwise_torch(image: np.ndarray,
    filters: np.ndarray):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  image_torch, filters_torch = convert_to_torch(image, filters)
  image_torch, filters_torch = image_torch.to(device), filters_torch.to(device)

  df, _, cin, cmul = filters.shape
  filters_torch = filters_torch.transpose(0, 1).contiguous()
  filters_torch = filters_torch.view(cin * cmul, 1, df, df)

  features_torch = F.conv2d(image_torch, filters_torch, padding=df // 2,
                            groups=cin)
  features_torch_ = features_torch.cpu().numpy()[0].transpose([2, 1, 0])

  return features_torch_


def cnn2d_torch(image: np.ndarray,
    filters: np.ndarray):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  image_torch, filters_torch = convert_to_torch(image, filters)
  image_torch, filters_torch = image_torch.to(device), filters_torch.to(device)
  df, _, cin, cmul = filters.shape
  features_torch = F.conv2d(image_torch, filters_torch, padding=df // 2)
  features_torch_ = features_torch.cpu().numpy()[0].transpose([2, 1, 0])

  return features_torch_


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  import datetime
  import time

  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


  def plot_image(image):
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(image[..., 0])
    ax[1].imshow(image[..., 1])
    ax[2].imshow(image[..., 2])
    plt.show()


  # g1 = makeGaussian(21, 1)
  # g2 = makeGaussian(21, 2)
  # g3 = makeGaussian(21, 1, (3, 3))
  g1 = createCircularMask(21, 21, radius=4)
  g2 = createCircularMask(21, 21, radius=5)
  g3 = createCircularMask(21, 21, [3, 3], radius=3)
  gauss_image = np.stack([g1, g2, g3], axis=-1)
  plot_image(gauss_image)
  gauss_kernel = np.stack([makeGaussian(5, 1)] * 3, axis=-1)
  plot_image(gauss_kernel)

  tf_convolved = cnn2d_depthwise_tf(gauss_image,
                                    check_shape_kernel(gauss_kernel,
                                                       gauss_image))
  plot_image(tf_convolved)

  torch_convolved = cnn2d_depthwise_torch(gauss_image,
                                          check_shape_kernel(gauss_kernel,
                                                             gauss_image))
  plot_image(torch_convolved)

  print('difference between pytorch and tf ',
        np.mean(tf_convolved - torch_convolved))

  iters = 100000

  start_time = time.time()
  for i in range(iters):
    tf_convolved = cnn2d_tf(gauss_image,
                            check_shape_kernel(gauss_kernel,
                                               gauss_image))
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time usage conv2d TF2: " + time_usage, flush=True)

  start_time = time.time()
  for i in range(iters):
    torch_convolved = cnn2d_torch(gauss_image,
                                  check_shape_kernel(gauss_kernel,
                                                     gauss_image))
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time usage conv2d Torch: " + time_usage, flush=True)


  start_time = time.time()
  for i in range(iters):
    tf_convolved = cnn2d_depthwise_tf(gauss_image,
                            check_shape_kernel(gauss_kernel,
                                               gauss_image))
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time usage depth_wise_conv2d TF2: " + time_usage, flush=True)

  start_time = time.time()
  for i in range(iters):
    torch_convolved = cnn2d_depthwise_torch(gauss_image,
                                  check_shape_kernel(gauss_kernel,
                                                     gauss_image))
  time_usage = str(datetime.timedelta(
      seconds=int(round(time.time() - start_time))))
  print("Time usage depth_wise_conv2d Torch: " + time_usage, flush=True)

  # With @tf.function
  # Time usage conv2d TF2: 0:00:21
  # Time usage conv2d Torch: 0:00:14
  # Time usage depth_wise_conv2d TF2: 0:00:17
  # Time usage depth_wise_conv2d Torch: 0:00:10

  # No @ tf function
  # Time usage conv2d TF2: 0:00:24
  # Time usage conv2d Torch: 0:00:14
  # Time usage depth_wise_conv2d TF2: 0:00:32
  # Time usage depth_wise_conv2d Torch: 0:00:10