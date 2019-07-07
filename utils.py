import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
import numpy as np
import time
from PIL import Image


def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize_images(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def imshow(image, title):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  plt.title(title)

