#!/bin/bash
#
# setup python + tensorflow on M1 mac
#

WHEEL_CACHE=$HOME/wheel

# Tensorflow 2.10.0 ... fails with model-maker deps
TENSORFLOW_VER=2.10.0
TENSORFLOW_IO_VER=0.27.0
TENSORFLOW_TEXT_VER=2.10.0

# Tensorflow 2.8.0
TENSORFLOW_VER=2.8.0
# should be 2.8.0 but this doesn't build
TENSORFLOW_TEXT_VER=2.10.0 

# default version - override as needed
export USE_BAZEL_VERSION=5.1.1

mkdir -p ${WHEEL_CACHE}

brew install --cask miniforge
brew install graphviz
#brew install protobuf
brew install cmake 
#brew install opencv 
brew install ffmpeg@4
brew install bazelisk

brew install llvm

PATH=/opt/homebrew/Cellar/qt@5/5.15.7/bin:$PATH
export PATH

#
# clean up
#
conda env remove -n tensorflow 
rm -rf models io text opencv opencv_contrib opencv-python protobuf addons

set -e

#
# see https://developer.apple.com/metal/tensorflow-plugin/
#
conda create --yes --name tensorflow python=3.8
conda init bash
. ~/.bash_profile
conda activate tensorflow 
pip install --upgrade pip
conda install --yes -c apple tensorflow-deps
pip install tensorflow-macos==${TENSORFLOW_VER}
pip install tensorflow-metal==0.6.0

#
# from training course
#
pip install notebook
pip install scikit-learn
pip install pydot
pip install matplotlib
pip install pandas
pip install seaborn
pip install numpy
pip install tensorflow-hub
pip install sacrebleu==2.2.0

#
# display versions
#
type python
python --version
type jupyter
jupyter --version
type pip
pip --version
python -c 'import tensorflow as tf; print(tf.__version__)'

#
# from https://www.tensorflow.org/hub/tutorials/tf2_object_detection
#

#
# some deps not provided in arm format
# 

# tensorflow_io for tensorflow 2.10.0
if [ ! -f ${WHEEL_CACHE}/io/tensorflow_io-${TENSORFLOW_IO_VER}-cp38-cp38-macosx_11_0_arm64.whl ]
then
  git clone https://github.com/tensorflow/io.git
  cd io
  git checkout v${TENSORFLOW_IO_VER} 
  python setup.py --project tensorflow-io-gcs-filesystem -q bdist_wheel
  python setup.py -q bdist_wheel
  mkdir -p ${WHEEL_CACHE}/io
  cp dist/*.whl ${WHEEL_CACHE}/io
  cd ..
  rm -rf io
fi
pip install ${WHEEL_CACHE}/io/tensorflow_io-${TENSORFLOW_IO_VER}-cp38-cp38-macosx_11_0_arm64.whl ${WHEEL_CACHE}/io/tensorflow_io_gcs_filesystem-${TENSORFLOW_IO_VER}-cp38-cp38-macosx_11_0_arm64.whl

# tensorflow_text for tensorflow 2.10.0
if [ ! -f ${WHEEL_CACHE}/text/tensorflow_text-${TENSORFLOW_TEXT_VER}-cp38-cp38-macosx_11_0_arm64.whl ]
then
  git clone https://github.com/tensorflow/text.git
  cd text
  git checkout v${TENSORFLOW_TEXT_VER}
  # needed for mac sed
  # curl -L https://raw.githubusercontent.com/tensorflow/text/master/oss_scripts/prepare_tf_dep.sh > oss_scripts/prepare_tf_dep.sh
  ./oss_scripts/run_build.sh
  mkdir -p ${WHEEL_CACHE}/text
  cp *.whl ${WHEEL_CACHE}/text
  cd ..
  rm -rf text
fi
pip install ${WHEEL_CACHE}/text/tensorflow_text-${TENSORFLOW_TEXT_VER}-cp38-cp38-macosx_11_0_arm64.whl

# addons for tensorflow 2.10.0
if [ ! -f ${WHEEL_CACHE}/addons/tensorflow_addons-0.18.0-cp38-cp38-macosx_12_0_arm64.whl ]
then
  git clone https://github.com/tensorflow/addons.git
  cd addons
  git checkout v0.18.0
  python3 ./configure.py
  bazel build build_pip_pkg
  bazel-bin/build_pip_pkg artifacts
  mkdir -p ${WHEEL_CACHE}/addons
  cp artifacts/*.whl ${WHEEL_CACHE}/addons/
  cd ..
  rm -rf addons
fi
pip install ${WHEEL_CACHE}/addons/tensorflow_addons-0.18.0-cp38-cp38-macosx_12_0_arm64.whl

pip install --no-deps tf-models-official==2.10.1
pip install tf-slim

pip install pycocotools
pip install lvis
pip install pyyaml
pip install gin-config
pip install labelImg

git clone https://github.com/tensorflow/models
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install --no-deps  .
cd ../..

#
# test install
#
python models/research/object_detection/builders/model_builder_tf2_test.py

#
# conversion scripts
#
#curl -s -L -O https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/create_csv.py
curl -s -L -O https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/create_tfrecord.py

#jupyter-notebook