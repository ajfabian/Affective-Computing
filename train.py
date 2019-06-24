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

lmarks_test, emot_test = np.array(lmarks[:100]), np.array(emot[:100])
lmarks_train, emot_train = np.array(lmarks[100:]), np.array(emot[100:])

def count_x(list, x):
  res = 0
  for e in list:
    if x == e: res += 1
  return res

def check_emos(list):
  for i in range(7):
    if count_x(list, i) == 0:
      return False
  return True

print(check_emos(emot))

if not check_emos(emot_test) or not check_emos(emot_train):
  print("Bad Set")
  exit(1)

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

model.fit(lmarks_train, emot_train, epochs=1000)
print('Train:')
model.evaluate(lmarks_train, emot_train)
print('Test:')
model.evaluate(lmarks_test, emot_test)

print('Resumen:')
Y = model.predict(lmarks_test)
ids = {0:'neutral', 1:'anger\t', 2:'contempt', 3:'disgust', 4:'fear\t', 
        5:'happy\t', 6:'sadness', 7:'surprise'}
ok = [0, 0, 0, 0, 0, 0, 0, 0]
tot = [0, 0, 0, 0, 0, 0, 0, 0]
for i in range(len(Y)):
  real = emot_test[i]
  val = np.argmax(Y[i])
  if real == val:
      ok[int(real)+1] += 1
  tot[int(real)+1] += 1

for i in range(1, 8):
  print(ids[i],'\t', int(100.0 * ok[i] / tot[i]), '% (', ok[i] , '/', tot[i], ')')

'''
>>> Tested on unknown set
Train:
227/227 [==============================] - 0s 603us/sample - loss: 0.1829 - acc: 0.9427
Test:
100/100 [==============================] - 0s 213us/sample - loss: 0.4888 - acc: 0.8300
Resumen:
anger            52 % ( 9 / 17 )
contempt         60 % ( 3 / 5 )
disgust          89 % ( 17 / 19 )
fear             87 % ( 7 / 8 )
happy            100 % ( 20 / 20 )
sadness          70 % ( 7 / 10 )
surprise         95 % ( 20 / 21 )
'''
