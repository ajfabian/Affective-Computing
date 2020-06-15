# this is a generic object-detection script

from object_detection.utils import label_map_util
from imutils.video import VideoStream, FPS
import tensorflow as tf, numpy as np, argparse, imutils, cv2, time, os

# images
imagesList = [os.sep.join(['wider', _]) for _ in os.listdir('wider')]
modelPath = os.sep.join(['fdetect11785', 'frozen_inference_graph.pb'])

# initialize the model
with tf.device('/device:XLA_GPU:2'):
  model = tf.Graph()

# create a context manager that makes this model the default one for execution
with model.as_default():
  # initialize the graph definition
  graphDef = tf.GraphDef()
  
  # load the graph from disk
  with tf.gfile.GFile(modelPath, 'rb') as f:
    serializedGraph = f.read()
    graphDef.ParseFromString(serializedGraph)
    tf.import_graph_def(graphDef, name="")

# load the class labels from disk
labelMap = label_map_util.load_labelmap("classes.pbtxt")
categories = label_map_util.convert_label_map_to_categories(
    labelMap, max_num_classes = 1, use_display_name = True
  )
categoryIdx = label_map_util.create_category_index(categories)

# initialize FPS counter
print("[INFO] starting FPS counter...")


times = []

# create a session to perform inference
with model.as_default():
  with tf.Session(graph = model) as sess:
    # grab a reference to the input image tensor and the boxes tensor
    imageTensor = model.get_tensor_by_name("image_tensor:0")
    boxesTensor = model.get_tensor_by_name("detection_boxes:0")
    
    # for each bounding box we would like to know the score (i.e., probability) and class label
    scoresTensor = model.get_tensor_by_name("detection_scores:0")
    classesTensor = model.get_tensor_by_name("detection_classes:0")
    numDetections = model.get_tensor_by_name("num_detections:0")
    
    L = [cv2.imread(imagesList[i]) for i in range(1000)]
    for image in L:
      image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
      image = np.expand_dims(image, axis = 0)
      
      st = time.time()
      # perform inference and compute bounding boxes, probabilities and class labels
      (boxes, scores, labels, N) = sess.run([boxesTensor, scoresTensor, classesTensor, numDetections], feed_dict = {imageTensor : image})
      elapsed = time.time() - st

      times.append(elapsed)

# print FPS
print('fps:', len(times) / sum(times))
print('Elapsed:', str(sum(times)) + 's')
print('AVG time:', str(sum(times)/len(times)) + 's')




