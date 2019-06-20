#!/usr/bin/python3

import skimage.io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import sys

image = skimage.io.imread(sys.argv[1], as_grey=True)
plt.imshow(image, cmap='gray')
print('Image Dimensions:', image.shape)

kernel_h = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
kernel_v = np.array([[1, 0, 1],[2, 0, -2],[-1, 0, -1]])

input_placeholder = tf.placeholder(dtype=tf.float32, shape=(1, image.shape[0], image.shape[1], 1))
with tf.name_scope('convolution'):
  conv_w_h = tf.constant(kernel_h, dtype=tf.float32, shape=(3,3,1,1))
  conv_w_v = tf.constant(kernel_v, dtype=tf.float32, shape=(3,3,1,1))
  output_h = tf.nn.conv2d(input=input_placeholder, filter=conv_w_h, strides=[1,1,1,1], padding='SAME')
  output_v = tf.nn.conv2d(input=input_placeholder, filter=conv_w_v, strides=[1,1,1,1], padding='SAME')

with tf.Session() as sess:
    result_h = sess.run(output_h, feed_dict={input_placeholder: image[np.newaxis, :, :, np.newaxis]})
    result_v = sess.run(output_v, feed_dict={input_placeholder: image[np.newaxis, :, :, np.newaxis]})

# plt.imshow(result_h[0, :, :, 0])
# plt.show()
# plt.imshow(result_v[0, :, :, 0])
# plt.show()

result_length = ((result_h)**2+(result_v)**2)**.5

# plt.imshow(result_length[0, :, :, 0], cmap='hot')
# plt.show()
# plt.imshow(result_length[0, :, :, 0])
# plt.show()
# plt.imshow(result_length[0, :, :, 0], cmap='hot')
# plt.show()

result_angle = np.arctan(result_v / (result_h + 1e-8))

# plt.imshow(result_angle[0, :, :, 0], cmap='hot')
# plt.show()


result_lenght_norm = (result_length[0,:,:,0] + (np.min(result_length)*-1) ) / (np.min(result_length)*-1 + np.max(result_length))
result_angle_norm = result_angle[0,:,:,0]

result_red = np.absolute(result_lenght_norm * np.cos(result_angle_norm+4.2))
result_green = np.absolute(result_lenght_norm * np.cos(result_angle_norm+2.1))
result_blue = np.absolute(result_lenght_norm * np.cos(result_angle_norm))

result_rgb = np.zeros((image.shape[0],image.shape[1],3))
result_rgb[...,0] = (result_red + (np.min(result_red)*-1) ) / (np.min(result_red)*-1 + np.max(result_red)) 
result_rgb[...,1] = (result_green + (np.min(result_green)*-1) ) / (np.min(result_green)*-1 + np.max(result_green)) 
result_rgb[...,2] = (result_blue + (np.min(result_blue)*-1) ) / (np.min(result_blue)*-1 + np.max(result_blue))

plt.imshow(result_rgb)
plt.show()
