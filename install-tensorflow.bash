#!/bin/bash

#set -e
set -x

export PATH="/opt/homebrew/opt/qt5/bin:$PATH"
export LDFLAGS="-L/opt/homebrew/opt/qt@5/lib"
export CPPFLAGS="-I/opt/homebrew/opt/qt@5/include"

#if [ -d Tensorflow ]
#then
#	rm -rf Tensorflow
#fi

mkdir Tensorflow
cd Tensorflow
tensordir=$(pwd)

#python3 -m venv tf2_api_env
#. tf2_api_env/bin/activate
#python -m pip install -U pip

#pip install tensorflow-macos
#pip install tensorflow-metal

git clone https://github.com/tensorflow/models.git

mkdir protoc
cd protoc
curl -OJL https://github.com/protocolbuffers/protobuf/releases/download/v3.19.0/protoc-3.19.0-osx-x86_64.zip
unzip protoc-3.19.0-osx-x86_64.zip

cd ${tensordir}/models/research
${tensordir}/protoc/bin/protoc object_detection/protos/*.proto --python_out=.

cd ${tensordir}
pip install cython
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools ${tensordir}/models/research/
cd ${tensordir}
	
cd models/research/
cp object_detection/packages/tf2/setup.py .
python -m pip install .	
cd ${tensordir}/..

pip install labelImg

mkdir ${tensordir}/scripts
curl https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/da4babe668a8afb093cc7776d7e630f3/generate_tfrecord.py -o ${tensordir}/scripts/generate_tfrecord.py

#
# validate environment
#
. Tensorflow/tf2_api_env/bin/activate
python Tensorflow/models/research/object_detection/builders/model_builder_tf2_test.py
