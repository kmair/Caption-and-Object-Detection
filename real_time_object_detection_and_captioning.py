# USAGE
# python real_time_object_detection_and_captioning.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

import os
from PIL import Image
from pickle import dump, load
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import img_to_array

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Get trained Inception V3 model for Caption generation
path = os.path.join( os.getcwd(), os.path.join('Caption_analysis', 'flickr8k') )
txtpath = os.path.join(path, "Flickr_TextData")
imgpath = os.path.join(path, "Images")

with open(os.path.join(path, 'wordtoix.pkl'), 'rb') as f:
    wordtoix = load(f)

ixtoword = dict(map(reversed, wordtoix.items()))

# Constants found when analyzing the train file
max_length = 34
vocab_size = len(ixtoword) + 1 # one for appended 0's
embedding_dim = 200

# Model for object detection
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Model for image captioning
# DL model
def captionModel():
    inputs1 = Input(shape=(2048,))

    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))

    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)

    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

model = captionModel()

model.load_weights( os.path.join(os.path.join(path, 'Models'), 'best_imagenet.h5') )

# Load the inception v3 model
inception = InceptionV3(weights='imagenet')

inception_model = Model(inception.input, inception.layers[-2].output)

def preprocess(image):
    # inputs:
    # :image: image file to be converted to an array
	# outputs: 
	# :image array: of shape (1, 3, 299, 299)

	x = img_to_array(cv2.resize(image, (299, 299)), data_format="channels_last")    # x: (299, 299, 3)
	
	x = np.expand_dims(x, axis=0) # Dimensions of x: (1, 3, 299, 299)

    # preprocess the images using preprocess_input() from inception module
	x = preprocess_input(x)
	return x

def encode(image):
    # inputs:
	# :image: the image file
	
	# outputs: 
	# :fea_vec: of shape (1, 2048)
	image_arr = preprocess(image) # preprocess the image
	fea_vec = inception_model.predict(image_arr) # Get the encoding vector for the image
	return fea_vec

def captionGenerator(photo):
    # inputs:
	# :photo: the frame whose caption will be generated
	
	# outputs: 
	# :final_seq: The string of text generated
	in_text = 'startseq'
	for i in range(max_length):
		sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
		sequence = pad_sequences([sequence], maxlen=max_length)
		yhat = model.predict([photo,sequence], verbose=0)
		yhat = np.argmax(yhat)
		word = ixtoword[yhat]
		in_text += ' ' + word
		if word == 'endseq':
			break   
	
	final = in_text.split()
	final_seq = final[1:-1]
	final_seq = ' '.join(final_seq)
	return final_seq

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

t_ref = time.time()-1
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	## Captions updated after 1 second

	encoded_img = encode(frame)

	captionX = 5
	captiony = 15

	if ( time.time() - t_ref) >= 1:
		t_ref = time.time()
		caption = captionGenerator(encoded_img)
	
	caption_words = caption.split()
	N = 10		# max words in a line
	if len(caption_words) > N:
		l1 = caption_words[:N]
		cv2.putText(frame, ' '.join(l1), (captionX, captiony),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, np.ones(3)*255, 1)
		
		l2 = caption_words[N:]
		cv2.putText(frame, ' '.join(l2), (captionX, captiony+10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.35, np.ones(3)*255, 1)

	else:
		cv2.putText(frame, caption, (captionX, captiony),
				cv2.FONT_HERSHEY_SIMPLEX, 0.35, np.ones(3)*255, 1)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extracting the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	
	# showing the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# doing a bit of cleanup
cv2.destroyAllWindows()
vs.stop()