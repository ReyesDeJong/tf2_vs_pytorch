import matplotlib.pyplot as plt
import numpy as np


def timer(start, end):
  hours, rem = divmod(end - start, 3600)
  minutes, seconds = divmod(rem, 60)
  return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def plot_image(image):
  fig, ax = plt.subplots(nrows=1, ncols=3)
  ax[0].imshow(image[..., 0])
  ax[1].imshow(image[..., 1])
  ax[2].imshow(image[..., 2])
  plt.show()


def create_circular_mask(h, w, center=None, radius=None):
  if center is None:  # use the middle of the image
    center = [int(w / 2), int(h / 2)]
  if radius is None:  # use the smallest distance between the center and image walls
    radius = min(center[0], center[1], w - center[0], h - center[1])

  Y, X = np.ogrid[:h, :w]
  dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

  mask = dist_from_center <= radius
  return mask * 1.0


def make_gaussian(size, sigma=3, center=None):
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
