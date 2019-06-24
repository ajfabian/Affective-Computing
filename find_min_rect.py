#!/usr/bin/python3

import skimage.io
import os

Shapes = set()
for f in os.listdir('Images'):
  img = skimage.io.imread(os.path.join('Images', f))
  Shapes.add((img.shape[0], img.shape[1]))

print(Shapes)
