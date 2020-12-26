# Caption-and-Object-Detection
Does Object detection and captioning in tandem

This project combines:

- Object detection: The objects are classified on a video stream and highlighted in the video being played. Check source code [here](https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/?__s=49gzk51snmgloxi5tiun).
- Video captioning: Evaluating video frames, the model identifies the best caption based on a model from Flickr8K dataset. The code is inspired from [this post](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/).

## Data:

- Flickr data for generating image captions: [Flickr 8K data](https://www.kaggle.com/shadabhussain/flickr8k)

- GloVe data for vector representation of words: [Glove data as txt](https://www.kaggle.com/incorpes/glove6b200d)
## Running:

After cloning the repository, you can run 2 implementations of object detection and caption generation:

### 1. Image detection 

**Required commands**

i. -p or --prototxt: Path to trained Caffe prototxt file 

ii. -m or --model: Path to Caffe pre-trained model

iii. -i or --image: Path to the image file

**Optional commands**

iv. -c or --confidence: Minimum probability to accept detection of images (default=0.2)

Example: python image_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --image <image file location>

### 2. Video detection 

**Required commands**

i. -p or --prototxt: Path to trained Caffe prototxt file 

ii. -m or --model: Path to Caffe pre-trained model

**Optional commands**

iii. -c or --confidence: Minimum probability to accept detection of images (default=0.2)

Example: python real_time_object_detection_and_captioning.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

Note: Allow the laptop's camera permissions to capture.

## Workflow:

The pretrained Caffe model weights were used for object detection. 

For the captioning system, a model had to be built from scratch. The `Caption_analysis` folder contains the relevant notebook and models used for modeling this and employed in the script files.

## Future Work:

Plans to extend the latter to a saved mp4 file.

## Acknowledgements:

A lot of help from Adrian Rosenbrock and Jason Brownlee's articles described above.
