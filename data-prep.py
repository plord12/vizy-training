#!/usr/bin/env python3
#
# data prep with some UK data
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
    print ('Downloading ' + url)
    if not os.path.exists(testfile):
        print ("Downloading "+testfile)
        ftpstream = urllib.request.urlopen(url)
        thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
        thetarfile.extractall()

#
# copy interesting image directories
#
def copy_images(source, destination):
    print ('Copying images from ' + Path(source).as_posix() + ' to ' + Path(destination).as_posix() )
    #
    # 10 most common UK birds
    #
    dirs = {}
    dirs['House Sparrow'] = '04132_Animalia_Chordata_Aves_Passeriformes_Passeridae_Passer_diffusus'
    dirs['Blue Tit'] = '04031_Animalia_Chordata_Aves_Passeriformes_Paridae_Cyanistes_caeruleus'
    #dirs['European Starling'] = '04202_Animalia_Chordata_Aves_Passeriformes_Sturnidae_Sturnus_vulgaris'
    #dirs['Eurasian Blackbird'] = '04284_Animalia_Chordata_Aves_Passeriformes_Turdidae_Turdus_merula'
    #dirs['Wood Pigeon'] = '03516_Animalia_Chordata_Aves_Columbiformes_Columbidae_Columba_palumbus'
    #dirs['European Robin'] = '03978_Animalia_Chordata_Aves_Passeriformes_Muscicapidae_Erithacus_rubecula'
    #dirs['Great Tit'] = '04034_Animalia_Chordata_Aves_Passeriformes_Paridae_Parus_major'
    #dirs['Eurasian Goldfinch'] = '03809_Animalia_Chordata_Aves_Passeriformes_Fringillidae_Carduelis_carduelis'
    #dirs['Eurasian Magpie'] = '03775_Animalia_Chordata_Aves_Passeriformes_Corvidae_Pica_pica'
    #dirs['Long-tailed Tit'] = '03688_Animalia_Chordata_Aves_Passeriformes_Aegithalidae_Aegithalos_caudatus'

    #
    # extra birds
    #
    #dirs['Red Kite'] = '03172_Animalia_Chordata_Aves_Accipitriformes_Accipitridae_Milvus_milvus'
    #dirs['Grey Heron'] = '04356_Animalia_Chordata_Aves_Pelecaniformes_Ardeidae_Ardea_cinerea'

    #
    # common garden animals
    #
    #dirs['Common Hedgehog'] = '04745_Animalia_Chordata_Mammalia_Eulipotyphla_Erinaceidae_Erinaceus_europaeus'
    #dirs['Red Fox'] = '04677_Animalia_Chordata_Mammalia_Carnivora_Canidae_Vulpes_vulpes'
    #dirs['Eastern Gray Squirrel'] = '04844_Animalia_Chordata_Mammalia_Rodentia_Sciuridae_Sciurus_carolinensis'
    dirs['Domestic Cat'] = '04679_Animalia_Chordata_Mammalia_Carnivora_Felidae_Felis_catus'

    #
    # people
    #

    os.makedirs(destination, exist_ok=True)
    for name, directory in dirs.items():
        shutil.copytree(os.path.join(source, directory), destination, dirs_exist_ok=True)

    return dirs

#
# label images if missing
#
def label_img(data_dir):
    #
    # copy to tmp dir first so we only edit necessary files
    #
    tmp_data_dir = Path('images/tmp')
    with open('classes.txt', "w") as file_out:
        for name in classes.keys():
            print (name, file=file_out)
    for thisdirpath, subdirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jpg"):
                xml=os.path.splitext(file)[0]+'.xml'
                print(str(file)+' '+str(xml))
                if not os.path.isfile(os.path.join(thisdirpath, xml)):
                    os.makedirs(tmp_data_dir, exist_ok=True)    
                    shutil.copyfile(os.path.join(thisdirpath, file), os.path.join(tmp_data_dir, file))

    if os.path.exists(tmp_data_dir):
        os.system('labelImg '+str(tmp_data_dir)+' classes.txt')
        for thisdirpath, subdirs, files in os.walk(tmp_data_dir):
            for file in files:
                if file.endswith(".xml"):
                    source = os.path.join(thisdirpath, file)
                    dest = os.path.join(data_dir, file)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(source,'r') as file_in:
                        with open(dest, "w") as file_out:
                            for line in file_in:
                                newline = re.sub('<path>'+os.path.join(os.getcwd(), tmp_data_dir), '<path>'+str(data_dir), line)
                                print (newline, end = "", file=file_out)
        shutil.rmtree(tmp_data_dir)

#
# main parameters
#
training_data_dir = Path('images/training')
validation_data_dir = Path('images/validation')
boundingbox_data_dir = Path('boundingbox')

#
# download data
#
download_extract_tgz('https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz', 'train')
download_extract_tgz('https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz', 'val')
#download_extract_tgz('https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.tar.gz', 'public_test')

#
# extract images we are interested in
#
classes = copy_images('train', training_data_dir);
copy_images('val', validation_data_dir);

#
# copy any bounding boxes in
#
print ('Copying bounding box xml data')
shutil.copytree(boundingbox_data_dir, 'images', dirs_exist_ok=True)

#
# run labelImg
#
label_img(training_data_dir)
label_img(validation_data_dir)

#
# backup xml (for any further data prep runs)
#
for thisdirpath, subdirs, files in os.walk('images'):
    for file in files:
        if file.endswith(".xml"):
            source = os.path.join(thisdirpath, file)
            dest = os.path.join(boundingbox_data_dir, os.path.join(*(source.split(os.path.sep)[1:])))
            print('Backing up '+source+' to '+dest)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(source, dest)

#
# create label map
#
count=1
with open('label_map.pbtxt', "w") as file_out:
    for name in classes.keys():
        print ('item {', file=file_out)
        print ('    id: '+str(count), file=file_out)
        print ('    name: \''+name+'\'', file=file_out)
        print ('}', file=file_out)
        count=count+1

#
# create TensorFlow records from xml
#
os.system('python Tensorflow/scripts/generate_tfrecord.py -x images/training/ -l label_map.pbtxt -o training.tfrecord')
os.system('python Tensorflow/scripts/generate_tfrecord.py -x images/validation/ -l label_map.pbtxt -o validation.tfrecord')


