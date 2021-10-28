#!/usr/bin/env python3
#
# training
#

import os
import urllib.request
import tarfile
import os.path
import shutil
import re
from pathlib import Path

#
# download and extract tgz
#
def download_extract_tgz(url, testfile):
    if not os.path.exists(testfile):
        print ("Downloading "+url)
        ftpstream = urllib.request.urlopen(url)
        thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
        thetarfile.extractall()

#
# setup model build tree
#
def setup_model(basedir, model):
    destdir = os.path.join(basedir, model)
    sourcedir = model
    if not os.path.exists(destdir):
        print ("Setting up "+destdir)
        os.makedirs(destdir)
        classes = 0
        with open('classes.txt','r') as file_in:
            for line in file_in:
                classes=classes+1
        with open(os.path.join(sourcedir, 'pipeline.config'),'r') as file_in:
            with open(os.path.join(destdir, 'pipeline.config'), 'w') as file_out:
                intrain = False
                ineval = False
                for line in file_in:
                    if re.match("train_input_reader", line):
                        intrain = True
                        ineval = False
                    if re.match("eval_input_reader", line):
                        intrain = False
                        ineval = True
                    # FIX THIS - probabally depends on available memory.  512 is too big
                    line = re.sub('batch_size: [0-9]*', 'batch_size: 128', line)
                    line = re.sub('num_classes: [0-9]*', 'num_classes: '+str(classes), line)
                    line = re.sub('fine_tune_checkpoint: "[^"]*"', 'fine_tune_checkpoint: "'+os.path.join(sourcedir, 'checkpoint', 'ckpt-0')+'"', line)
                    line = re.sub('fine_tune_checkpoint_type: "[^"]*"', 'fine_tune_checkpoint_type: "detection"', line)
                    line = re.sub('label_map_path: "[^"]*"', 'label_map_path: "label_map.pbtxt"', line)
                    if intrain:
                        line = re.sub('input_path: "[^"]*"', 'input_path: "training.tfrecord"', line)
                    if ineval:
                        line = re.sub('input_path: "[^"]*"', 'input_path: "validation.tfrecord"', line)

                    print (line, end = "", file=file_out)


#
# model name
#
model = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
modeldir = 'models'
#
# get original model
#
download_extract_tgz('http://download.tensorflow.org/models/object_detection/tf2/20200711/'+model+'.tar.gz', model)

#
# setup model build directory
#
setup_model(modeldir, model)

#
# try training
#
os.system('python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir='+os.path.join(modeldir, model)+' --pipeline_config_path='+os.path.join(modeldir, model, 'pipeline.config'))

# FIX THIS - add monitoring
# FIX THIS - add valuation
# FIX THIS - add export
