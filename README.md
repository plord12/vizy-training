# Vizy cam training

An attempt to train [vizycam](https://vizycam.com/) with UK birds + animals

## Data sets :

* https://github.com/visipedia/inat_comp/tree/master/2021

Included are :

10 most common UK birds :

* House Sparrow :heavy_check_mark:
* Blue Tit :heavy_check_mark:
* European Starling :heavy_check_mark:
* Eurasian Blackbird :heavy_check_mark:
* Wood Pigeon :heavy_check_mark:
* European Robin :heavy_check_mark:
* Great Tit :heavy_check_mark:
* Eurasian Goldfinch :heavy_check_mark:
* Eurasian Magpie :heavy_check_mark:
* Long-tailed Tit :heavy_check_mark:

Extra birds ( seen in my garden ) :

* Red Kite :heavy_check_mark:
* Grey Heron :heavy_check_mark:
* Blackcap :heavy_check_mark:
* Redwing :heavy_check_mark:

Common garden animals :

* Common Hedgehog :heavy_check_mark:
* Red Fox :heavy_check_mark:
* Eastern Gray Squirrel :heavy_check_mark:
* Domestic Cat :heavy_check_mark:
* Person :heavy_check_mark:

Weeds :

* Common Dandelion :heavy_check_mark:

( :heavy_check_mark: indicates bounding boxes created )

## Tensor flow on mac :

* https://github.com/apple/tensorflow_macos

## Models :

* https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

## Other links :

* https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html
* https://www.tensorflow.org/tutorials/images/classification
* http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz
* https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb
* https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api

## To run on mac intel :

    ./install-tensorflow.bash
    . Tensorflow/tf2_api_env/bin/activate
    ./data-prep.py
    ./train.py

## To run on mac M1 :

    ./install-tensorflow-macos-m1.sh
    conda activate tensorflow
    ./data-prep.py
    ./train2.py

## Directories used :

* Tensorflow - installed tensorflow binaries
* train - training dataset download directory
* val - training dataset download directory
* boundingbox - source of bounding box data
* images/training - training images ready to be used
* images/validation - validation images ready to be used
* models - tensorflow models and where training happens

