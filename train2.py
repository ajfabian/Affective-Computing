#!/usr/bin/python3

'''
Training using landmarks (this is not Deep Learning)
0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
'''

import os
import re
import numpy as np
from random import *

def normalize(list):
  xmin, xmax = float("inf"), float("-inf")
  ymin, ymax = float("inf"), float("-inf")
  for i in list:
    xmin = min(xmin, i[0])
    xmax = max(xmax, i[0])
    ymin = min(ymin, i[1])
    ymax = max(ymax, i[1])
  
  dx = xmax - xmin
  dy = ymax - ymin

  ret = list.copy()
  for i in range(len(ret)):
    ret[i][0] -= xmin
    ret[i][1] -= ymin
  
  for i in range(len(ret)):
    ret[i][0] /= dx
    ret[i][1] /= dy

  return ret

lmarks, emot = [], []
L = os.listdir('Emotion')
shuffle(L)
for emotion in L:
  fh = open(os.path.join('Emotion', emotion), 'r')
  emot.append(float(fh.readline().strip()) - 1.0) 
  fh.close()

  landmarks = re.sub(r'emotion', 'landmarks', emotion)
  fh = open(os.path.join('Landmarks', landmarks), 'r')
  tmp = []
  for ln in fh.readlines():
    tmp.append([float(_) for _ in ln.strip().split()])
  lmarks.append(normalize(tmp))
  # lmarks.append(tmp)
  # lmarks.append(np.array([tmp]))

  # print('File: ', landmarks)
  # print(tmp)

  fh.close()
  
# print(lmarks[0].shape)

# exit(0)


# print(emot, 'Total: ', len(emot_train))
# print(lmarks, 'Total: ', len(lmarks))

import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(68, 2)),
  tf.keras.layers.Dense(10000, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.333333333333),
  tf.keras.layers.Dense(7, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

lmarks = np.array(lmarks)
emot = np.array(emot)

model.fit(lmarks, emot, epochs=1000)

print('Train:')
model.evaluate(lmarks, emot)

print('Resumen:')
Y = model.predict(lmarks)
ids = {0:'neutral', 1:'anger\t', 2:'contempt', 3:'disgust', 4:'fear\t', 
        5:'happy\t', 6:'sadness', 7:'surprise'}
ok = [0, 0, 0, 0, 0, 0, 0, 0]
tot = [0, 0, 0, 0, 0, 0, 0, 0]
for i in range(len(Y)):
  real = emot[i]
  val = np.argmax(Y[i])
  if real == val:
      ok[int(real)+1] += 1
  tot[int(real)+1] += 1

for i in range(1, 8):
  print(ids[i],'\t', int(100.0 * ok[i] / tot[i]), '% (', ok[i] , '/', tot[i], ')')


'''
>>> Tested on the 'Training Set'
Train:
327/327 [==============================] - 0s 482us/sample - loss: 0.1933 - acc: 0.9450
Resumen:
anger            80 % ( 36 / 45 )
contempt         94 % ( 17 / 18 )
disgust          88 % ( 52 / 59 )
fear             100 % ( 25 / 25 )
happy            100 % ( 69 / 69 )
sadness          100 % ( 28 / 28 )
surprise         98 % ( 82 / 83 )
'''