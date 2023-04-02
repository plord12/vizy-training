#!/usr/bin/env python3

# train model - see https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb

import os
import tarfile
import re
import subprocess
import tensorflow as tf
import os
import cv2
import numpy as np
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter

import matplotlib
import matplotlib.pyplot as plt

np.random.seed(42)
random.seed(42)

rootdir = os.getcwd()

#
# clean up
#
os.system('rm -rf training models/mymodel custom_model_lite')

train_record_fname = rootdir+'/train.tfrecord'
val_record_fname = rootdir+'/val.tfrecord'
label_map_pbtxt_fname = rootdir+'/labelmap.pbtxt'

#
# 4. Set Up Training Configuration
#

# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
# https://github.com/tensorflow/models/tree/master/community


# 
# Change the chosen_model variable to deploy different models available in the TF2 object detection zoo
chosen_model = 'ssd-mobilenet-v2-fpnlite-320' # 85.80
#chosen_model = 'ssd-mobilenet-v2' # 68.77
#chosen_model = 'efficientdet-d0' # 18.15
#chosen_model = 'centernet-mobilenet-v2' # fails
#chosen_model = 'ssd-resnet50-v1' # fails

MODELS_CONFIG = {
    'ssd-mobilenet-v2': {
        'model_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
        'base_pipeline_file': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
    },
    'efficientdet-d0': {
        'model_name': 'efficientdet_d0_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d0_coco17_tpu-32.tar.gz',
    },
    'ssd-mobilenet-v2-fpnlite-320': {
        'model_name': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8',
        'base_pipeline_file': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz',
    },
    'ssd-resnet50-v1': {
        'model_name': 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8',
        'base_pipeline_file': 'ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz',
    },
    # The centernet model isn't working as of 9/10/22
    'centernet-mobilenet-v2': {
        'model_name': 'centernet_mobilenetv2fpn_512x512_coco17_od',
        'base_pipeline_file': 'pipeline.config',
        'pretrained_checkpoint': 'centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz',
    }
}

model_name = MODELS_CONFIG[chosen_model]['model_name']
pretrained_checkpoint = MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
base_pipeline_file = MODELS_CONFIG[chosen_model]['base_pipeline_file']

# Create "mymodel" folder for holding pre-trained weights and configuration files
os.system('mkdir -p models/mymodel')
os.chdir('models/mymodel')

# Download pre-trained model weights
if chosen_model == 'centernet-mobilenet-v2':
  download_tar = 'http://download.tensorflow.org/models/object_detection/tf2/20210210/' + pretrained_checkpoint
else:
  download_tar = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/' + pretrained_checkpoint

print(download_tar+"\n")
os.system('curl -s -L -O '+download_tar)
tar = tarfile.open(pretrained_checkpoint)
tar.extractall()
tar.close()

# Download training configuration file for model
download_config = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/' + base_pipeline_file

os.system('curl -s -L -O '+download_config)

os.chdir(rootdir)

# Set training parameters for the model
num_steps = 100000

if chosen_model == 'efficientdet-d0':
  # CPU only
  batch_size = 4
  os.system('pip uninstall tensorflow-metal')
else:
  # GPU or CPU
  batch_size = 16
  os.system('pip install tensorflow-metal==0.6.0')

# Set file locations and get number of classes for config file
pipeline_fname = rootdir+'/models/mymodel/' + base_pipeline_file
fine_tune_checkpoint = rootdir+'/models/mymodel/' + model_name + '/checkpoint/ckpt-0'

def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())
num_classes = get_num_classes(label_map_pbtxt_fname)
print('Total classes:', num_classes)

# Create custom configuration file by writing the dataset, model checkpoint, and training parameters into the base pipeline file

os.chdir('models/mymodel')
print('writing custom configuration file')

with open(pipeline_fname) as f:
    s = f.read()
with open('pipeline_file.config', 'w') as f:
    
    # Set fine_tune_checkpoint path
    s = re.sub('fine_tune_checkpoint: ".*?"',
               'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)
    
    # Set tfrecord files for train and test datasets
    s = re.sub(
        '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
    s = re.sub(
        '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(val_record_fname), s)

    # Set label_map_path
    s = re.sub(
        'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

    # Set batch_size
    s = re.sub('batch_size: [0-9]+',
               'batch_size: {}'.format(batch_size), s)

    # Set training steps, num_steps
    s = re.sub('num_steps: [0-9]+',
               'num_steps: {}'.format(num_steps), s)
    
    # Set number of classes num_classes
    s = re.sub('num_classes: [0-9]+',
               'num_classes: {}'.format(num_classes), s)

    # Change fine-tune checkpoint type from "classification" to "detection"
    s = re.sub(
        'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)
    
    # If using ssd-mobilenet-v2, reduce learning rate (because it's too high in the default config file)
    if chosen_model == 'ssd-mobilenet-v2':
      s = re.sub('learning_rate_base: .8',
                 'learning_rate_base: .08', s)
      
      s = re.sub('warmup_learning_rate: 0.13333',
                 'warmup_learning_rate: .026666', s)
    
    # If using efficientdet-d0, use fixed_shape_resizer instead of keep_aspect_ratio_resizer (because it isn't supported by TFLite)
    if chosen_model == 'efficientdet-d0':
      s = re.sub('keep_aspect_ratio_resizer', 'fixed_shape_resizer', s)
      s = re.sub('pad_to_max_dimension: true', '', s)
      s = re.sub('min_dimension', 'height', s)
      s = re.sub('max_dimension', 'width', s)

    f.write(s)

# Set the path to the custom config file and the directory to store training checkpoints in
pipeline_file = rootdir+'/models/mymodel/pipeline_file.config'
model_dir = rootdir+'/training/'

#
# 5. Train Custom TFLite Detection Model
#

# start tensorboard
tensorboard=subprocess.Popen(["tensorboard","--logdir",rootdir+"/training/train"])

# Run training!
res = os.system('python '+rootdir+'/models/research/object_detection/model_main_tf2.py --pipeline_config_path='+pipeline_file+' --model_dir='+model_dir+' --alsologtostderr --num_train_steps='+str(num_steps)+' --sample_1_of_n_eval_examples=1' )
if res != 0 :
  print ('res='+str(res))
  tensorboard.terminate()
  exit

tensorboard.terminate()

#
# 6. Convert Model to TensorFlow Lite
#

# Make a directory to store the trained TFLite model
os.system('mkdir -p custom_model_lite')
output_directory = rootdir+'/custom_model_lite'

# Path to training directory (the conversion script automatically chooses the highest checkpoint file)
last_model_path = rootdir+'/training'

os.system('python '+rootdir+'/models/research/object_detection/export_tflite_graph_tf2.py --trained_checkpoint_dir '+last_model_path+' --output_directory '+output_directory+' --pipeline_config_path '+pipeline_file)

# Convert exported graph file into TFLite model file
converter = tf.lite.TFLiteConverter.from_saved_model(rootdir+'/custom_model_lite/saved_model')
tflite_model = converter.convert()

with open(rootdir+'/custom_model_lite/detect.tflite', 'wb') as f:
  f.write(tflite_model)

#
# 7. Test TensorFlow Lite Model and Calculate mAP
#

# 7.1 Inference test images

### Define function for inferencing with TFLite model and displaying results

def tflite_detect_images(modelpath, imgpath, lblpath, min_conf=0.5, num_test_images=10, savepath='/content/results', txt_only=False):

  # Grab filenames of all images in test folder
  images = glob.glob(imgpath + '/*.jpg') + glob.glob(imgpath + '/*.JPG') + glob.glob(imgpath + '/*.png') + glob.glob(imgpath + '/*.bmp')

  # Load the label map into memory
  with open(lblpath, 'r') as f:
      labels = [line.strip() for line in f.readlines()]

  # Load the Tensorflow Lite model into memory
  interpreter = Interpreter(model_path=modelpath)
  interpreter.allocate_tensors()

  # Get model details
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]

  float_input = (input_details[0]['dtype'] == np.float32)

  input_mean = 127.5
  input_std = 127.5

  # Randomly select test images
  images_to_test = random.sample(images, num_test_images)

  # Loop over every image and perform detection
  for image_path in images_to_test:
      print(image_path)
      # Load image and resize to expected shape [1xHxWx3]
      image = cv2.imread(image_path)
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      imH, imW, _ = image.shape 
      image_resized = cv2.resize(image_rgb, (width, height))
      input_data = np.expand_dims(image_resized, axis=0)

      # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
      if float_input:
          input_data = (np.float32(input_data) - input_mean) / input_std

      # Perform the actual detection by running the model with the image as input
      interpreter.set_tensor(input_details[0]['index'],input_data)
      interpreter.invoke()

      # Retrieve detection results
      boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
      classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
      scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

      detections = []

      # Loop over all detections and draw detection box if confidence is above minimum threshold
      for i in range(len(scores)):
          if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

              # Get bounding box coordinates and draw box
              # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
              ymin = int(max(1,(boxes[i][0] * imH)))
              xmin = int(max(1,(boxes[i][1] * imW)))
              ymax = int(min(imH,(boxes[i][2] * imH)))
              xmax = int(min(imW,(boxes[i][3] * imW)))
              
              cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

              # Draw label
              object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
              label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
              labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
              label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
              cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
              cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

              detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

      
      # All the results have been drawn on the image, now display the image
      if txt_only == False: # "text_only" controls whether we want to display the image results or just save them in .txt files
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12,16))
        plt.imshow(image)
        plt.show()
      
      # Save detection results in .txt files (for calculating mAP)
      elif txt_only == True:

        # Get filenames and paths
        image_fn = os.path.basename(image_path)      
        base_fn, ext = os.path.splitext(image_fn)
        txt_result_fn = base_fn +'.txt'
        txt_savepath = os.path.join(savepath, txt_result_fn)

        # Write results to text file
        # (Using format defined by https://github.com/Cartucho/mAP, which will make it easy to calculate mAP)
        with open(txt_savepath,'w') as f:
            for detection in detections:
                f.write('%s %.4f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))

  return

# Set up variables for running user's model
PATH_TO_IMAGES=rootdir+'/images/test'   # Path to test images folder
PATH_TO_MODEL=rootdir+'/custom_model_lite/detect.tflite'   # Path to .tflite model file
PATH_TO_LABELS=rootdir+'/labelmap.txt'   # Path to labelmap.txt file
min_conf_threshold=0.5   # Confidence threshold (try changing this to 0.01 if you don't see any detection results)
images_to_test = 10   # Number of images to run detection on

# Run inferencing function!
tflite_detect_images(PATH_TO_MODEL, PATH_TO_IMAGES, PATH_TO_LABELS, min_conf_threshold, images_to_test)

# 7.2 Calculate mAP

os.system('git clone https://github.com/Cartucho/mAP '+rootdir+'/mAP')
os.chdir(rootdir+'/mAP')
os.system('rm -rf '+rootdir+'/mAP/input/detection-results/*')
os.system('rm -rf '+rootdir+'/mAP/input/ground-truth/*')
os.system('rm -rf '+rootdir+'/mAP/input/images-optional/*')
os.system('rm -rf '+rootdir+'/mAP/outputs')
os.system('curl -s -L -O https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/calculate_map_cartucho.py')

os.system('cp '+rootdir+'/images/test/* '+rootdir+'/mAP/input/images-optional') # Copy images and xml files
os.system('mv '+rootdir+'/mAP/input/images-optional/*.xml '+rootdir+'/mAP/input/ground-truth/')  # Move xml files to the appropriate folder

os.system('python '+rootdir+'/mAP/scripts/extra/convert_gt_xml.py')

# Set up variables for running inference, this time to get detection results saved as .txt files
PATH_TO_RESULTS=rootdir+'/mAP/input/detection-results' # Folder to save detection results in
min_conf_threshold=0.1   # Confidence threshold

# Use all the images in the test folder
image_list = glob.glob(PATH_TO_IMAGES + '/*.jpg') + glob.glob(PATH_TO_IMAGES + '/*.JPG') + glob.glob(PATH_TO_IMAGES + '/*.png') + glob.glob(PATH_TO_IMAGES + '/*.bmp')
images_to_test = min(500, len(image_list)) # If there are more than 500 images in the folder, just use 500

# Tell function to just save results and not display images
txt_only = True

# Run inferencing function!
print('Starting inference on %d images...' % images_to_test)
tflite_detect_images(PATH_TO_MODEL, PATH_TO_IMAGES, PATH_TO_LABELS, min_conf_threshold, images_to_test, PATH_TO_RESULTS, txt_only)
print('Finished inferencing!')

os.system('python scripts/extra/intersect-gt-and-dr.py')
os.system('python calculate_map_cartucho.py --labels='+PATH_TO_LABELS)

os.chdir(rootdir)

#
# 8. Deploy TensorFlow Lite Model
#

# 8.1. Download TFLite model

# Move labelmap and pipeline config files into TFLite model folder and zip it up
os.system('cp '+PATH_TO_LABELS+' '+rootdir+'/custom_model_lite')
os.system('cp '+label_map_pbtxt_fname+' '+rootdir+'/custom_model_lite')
os.system('cp '+pipeline_fname+' '+rootdir+'/custom_model_lite')

#
# 9. (Optional) Post-Training Quantization
#

# 9.1. Quantize model

# Get list of all images in train directory
image_path = rootdir+'/images/train'

jpg_file_list = glob.glob(image_path + '/*.jpg')
JPG_file_list = glob.glob(image_path + '/*.JPG')
png_file_list = glob.glob(image_path + '/*.png')
bmp_file_list = glob.glob(image_path + '/*.bmp')

quant_image_list = jpg_file_list + JPG_file_list + png_file_list + bmp_file_list

# A generator that provides a representative dataset
# Code modified from https://colab.research.google.com/github/google-coral/tutorials/blob/master/retrain_classification_ptq_tf2.ipynb

# First, get input details for model so we know how to preprocess images
interpreter = Interpreter(model_path=PATH_TO_MODEL) # PATH_TO_MODEL is defined in Step 7 above
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

def representative_data_gen():
  dataset_list = quant_image_list
  quant_num = 300
  for i in range(quant_num):
    pick_me = random.choice(dataset_list)
    image = tf.io.read_file(pick_me)

    if pick_me.endswith('.jpg') or pick_me.endswith('.JPG'):
      image = tf.io.decode_jpeg(image, channels=3)
    elif pick_me.endswith('.png'):
      image = tf.io.decode_png(image, channels=3)
    elif pick_me.endswith('.bmp'):
      image = tf.io.decode_bmp(image, channels=3)

    image = tf.image.resize(image, [width, height])  # TO DO: Replace 300s with an automatic way of reading network input size
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]


# Initialize converter module
converter = tf.lite.TFLiteConverter.from_saved_model(rootdir+'/custom_model_lite/saved_model')

# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input tensors to uint8 and output tensors to float32
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32
tflite_model = converter.convert()

with open(rootdir+'/custom_model_lite/detect_quant.tflite', 'wb') as f:
  f.write(tflite_model)

# 9.2. Test quantized model

# Set up parameters for inferencing function (using detect_quant.tflite instead of detect.tflite)
PATH_TO_MODEL=rootdir+'/custom_model_lite/detect_quant.tflite'   #Path to .tflite model file
min_conf_threshold=0.5   #Confidence threshold (try changing this to 0.01 if you don't see any detection results)
images_to_test = 10   #Number of images to run detection on

# Run inferencing function!
tflite_detect_images(PATH_TO_MODEL, PATH_TO_IMAGES, PATH_TO_LABELS, min_conf_threshold, images_to_test)

# 9.3 Calculate quantized model mAP

# Need to remove existing detection results first
os.system('rm '+rootdir+'/mAP/input/detection-results/*')
os.system('rm -rf '+rootdir+'/mAP/outputs')

# Set up variables for running inference, this time to get detection results saved as .txt files
min_conf_threshold=0.1   # Confidence threshold

# Use all the images in the test folder
image_list = glob.glob(PATH_TO_IMAGES + '/*.jpg') + glob.glob(PATH_TO_IMAGES + '/*.JPG') + glob.glob(PATH_TO_IMAGES + '/*.png') + glob.glob(PATH_TO_IMAGES + '/*.bmp')
images_to_test = min(500, len(image_list)) # If there are more than 500 images in the folder, just use 500

# Tell function to just save results and not display images
txt_only = True

# Run inferencing function!
print('Starting inference on %d images...' % images_to_test)
tflite_detect_images(PATH_TO_MODEL, PATH_TO_IMAGES, PATH_TO_LABELS, min_conf_threshold, images_to_test, PATH_TO_RESULTS, txt_only)
print('Finished inferencing!')

os.chdir(rootdir+'/mAP')
os.system('python scripts/extra/intersect-gt-and-dr.py')
os.system('python calculate_map_cartucho.py --labels='+PATH_TO_LABELS)
os.chdir(rootdir)

os.chdir(rootdir+'/custom_model_lite')
os.system('rm -f '+rootdir+'/'+chosen_model+'.zip')
os.system('zip -r '+rootdir+'/'+chosen_model+'.zip .')
os.chdir(rootdir)