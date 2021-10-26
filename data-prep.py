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
    #dirs['Blue Tit'] = '04031_Animalia_Chordata_Aves_Passeriformes_Paridae_Cyanistes_caeruleus'
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

    for name, directory in dirs.items():
        os.makedirs(os.path.join(destination, name), exist_ok=True)
        shutil.copytree(os.path.join(source, directory), os.path.join(destination, name), dirs_exist_ok=True)


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
copy_images('train', training_data_dir);
copy_images('val', validation_data_dir);

#
# copy any bounding boxes in
#
print ('Copying bounding box xml data')
shutil.copytree(boundingbox_data_dir, 'images', dirs_exist_ok=True)

#
# run labelImg if any missing xml
#
needlabel = {}
for dirpath, subdirs, files in os.walk(training_data_dir):
    for file in files:
        if file.endswith(".jpg"):
            xml = os.path.join(dirpath, file).rsplit('.', 1)[0] + '.xml'
            if not os.path.isfile(xml):
                needlabel[dirpath] = 'true'

for dirpath in needlabel.keys():
    print (dirpath)
    os.system('labelImg "'+dirpath+'"')

    #
    # fix and copy xml
    #
    for thisdirpath, subdirs, files in os.walk(dirpath):
        for file in files:
            if file.endswith(".xml"):
                source = os.path.join(thisdirpath, file)
                dest = os.path.join(boundingbox_data_dir, os.path.join(*(source.split(os.path.sep)[1:])))
                print('Backing up '+source+' to '+dest)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(source,'r') as file_in:
                    with open(dest, "w") as file_out:
                        for line in file_in:
                            newline = re.sub('<name>.*</name>', '<name>'+os.path.basename(thisdirpath)+'</name>', line)
                            newline = re.sub('<path>'+os.getcwd()+os.path.sep, '<path>', newline)
                            print (newline, end = "", file=file_out)


#
# create label map
#

#
# create TensorFlow records from xml
#
