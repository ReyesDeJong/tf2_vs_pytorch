import time

import numpy as np
import tensorflow as tf
import torch
from torch.nn import functional as F

from utils import timer, create_circular_mask, make_gaussian, plot_images


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


# TODO: include diference in conv, and a mean of n experiments
def main(perform_exp_n_times: int, iters_per_experiment: int, show_plots: bool):
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
  plot_images([image[0, ...], gauss_kernel[..., 0]],
              ['Image shape %s' % str(image.shape),
               'Kernel shape %s' % str(gauss_kernel.shape)], 'Image to process',
              show_plots)

  convolved_tf = cnn2d_tf(image, gauss_kernel)
  convolved_torch = cnn2d_torch(image, gauss_kernel)

  # plot tf and torch convolutions
  plot_images([convolved_tf[0, ..., 0], convolved_torch[0, ..., 0]],
              ['TF2 Conv shape %s' % str(convolved_tf.shape),
               'PyTorch Conv shape %s' % str(convolved_torch.shape)],
              'Difference between convolutions %.2f' % np.mean(
                  convolved_tf - convolved_torch), show_plots)

  delta_time_list = []
  for i in range(perform_exp_n_times):
    start_time = time.time()
    for i in range(iters_per_experiment):
      cnn2d_tf(image, gauss_kernel)
    delta_time_list.append(time.time() - start_time)
  print("Time usage conv2d TF2: %s +/- %s" % (
    str(timer(np.mean(delta_time_list))),
    str(timer(np.std(delta_time_list), True))))

  delta_time_list = []
  for i in range(perform_exp_n_times):
    start_time = time.time()
    for i in range(iters_per_experiment):
      cnn2d_torch(image, gauss_kernel)
    delta_time_list.append(time.time() - start_time)
  print("Time usage conv2d PyTorch: %s +/- %s" % (
    str(timer(np.mean(delta_time_list))),
    str(timer(np.std(delta_time_list), True))))

  if __name__ == "__main__":
    main(10, 10000, True)
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--perform_exp_n_times",
    #                     help="How many time to perform n_iterations for every function",
    #                     default=10,
    #                     type=int)
    # parser.add_argument("--iters_per_experiment",
    #                     help="How many times to run each function",
    #                     default=10000,
    #                     type=int)
    # parser.add_argument("--show_plots",
    #                     help="Whether to show image that are being convolved and their results",
    #                     default=0, type=int)
    # args = parser.parse_args()
    #
    # main(args.perform_exp_n_times, args.iters_per_experiment,
    #      bool(args.show_plots))
