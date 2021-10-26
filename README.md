# Vizi cam training

An attempt to train vizicam with UK birds + animals

Data sets :

* https://github.com/visipedia/inat_comp/tree/master/2021

Tensor flow on mac :

* https://github.com/apple/tensorflow_macos

Training :

* https://www.tensorflow.org/tutorials/images/classification
* https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
* http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz
* https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb
* https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api
* https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

Steps will probabally be :

    ./install-tensorflow.bash
    . Tensorflow/tf2_api_env/bin/activate
    ./data-prep.py
    ....

Directories used :

    Tensorflow - installed tensorflow binaries
    train - training dataset download directory
    val - training dataset download directory
    boundingbox - source of bounding box data
    images/training - training images ready to be used
    images/validation - validation images ready to be used
