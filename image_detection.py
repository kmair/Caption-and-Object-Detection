# USAGE
# python image_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --image images/polo.jpg 

# import the necessary packages
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
ap.add_argument("-i", "--image", required=True,
	help="path to image")
args = vars(ap.parse_args())

############ CAPTIONING ############
path = os.path.join( os.getcwd(), os.path.join('images', os.path.join('flickr8k','Flickr_Data')) )
txtpath = os.path.join(path, "Flickr_TextData")
imgpath = os.path.join(path, "Images")

with open(os.path.join(path, 'wordtoix.pkl'), 'rb') as f:
    wordtoix = load(f)

ixtoword = dict(map(reversed, wordtoix.items()))

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

img = cv2.imread(args["image"])

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
     
    # ip: img, scalefactor=1, spatial size for output,
    # image = cv2.dnn.blobFromImage(cv2.resize(image, (299, 299)), 1, (299, 299), 127.5)
    # x = np.array(image)

    x = img_to_array(cv2.resize(image, (299, 299)), data_format="channels_last")    # output is (299,299,3)
    
    x = np.expand_dims(x, axis=0)
    # print('Img_arr', x.shape)

    # # Dimensions of blob: (1, 3, 299, 299)
    # # Model is not channels first. So will convert it to (1, 299, 299, 3)
    # x = np.swapaxes(image, 1, 2)
    # # # print(x.shape)
    # x = np.swapaxes(x, 2, 3)
    # # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x

def encode(image):
    image_arr = preprocess(image) # preprocess the image
    fea_vec = inception_model.predict(image_arr) # Get the encoding vector for the image
    # fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec

def captionGenerator(photo):
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
    final = final[1:-1]
    final = ' '.join(final)
    return final

# grab the frame dimensions and convert it to a blob
(h, w) = img.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(img, (299, 299)),
    0.007843, (299, 299), 127.5)

# pass the blob through the network and obtain the detections and
# predictions
net.setInput(blob)
detections = net.forward()

print(detections.shape)
# loop over the onjects detected
for i in np.arange(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with
    # the prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > args["confidence"]:
        # extract the index of the class label from the
        # `detections`, then compute the (x, y)-coordinates of
        # the bounding box for the object
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the prediction on the frame
        label = "{}: {:.2f}%".format(CLASSES[idx],
            confidence * 100)
        cv2.rectangle(img, (startX, startY), (endX, endY),
            COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(img, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# Captions
start = time.time()
# encoded_img shape is (2048,) and 
encoded_img = encode(img)#.reshape((1,2048))    
end = time.time()

captionX = 15
captiony = 15

caption = captionGenerator(encoded_img)
cv2.putText(img, caption, (captionX, captiony),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, np.zeros(3), 2)
# COLORS[0]
print("caption:",caption)

# show the output frame
cv2.imshow("Output", img)
# key = cv2.waitKey(1) & 0xFF
cv2.waitKey(0)

#/#/
print("Time taken in seconds =", end-start)