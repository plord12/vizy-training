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
import glob
from pathlib import Path
import random

random.seed(42)

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
# download and extract tar
#
def download_extract_tar(url, testfile):
    print ('Downloading ' + url)
    if not os.path.exists(testfile):
        print ("Downloading "+testfile)
        ftpstream = urllib.request.urlopen(url)
        thetarfile = tarfile.open(fileobj=ftpstream, mode="r|")
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
    dirs['House_Sparrow'] = '04132_Animalia_Chordata_Aves_Passeriformes_Passeridae_Passer_diffusus'
    dirs['Blue_Tit'] = '04031_Animalia_Chordata_Aves_Passeriformes_Paridae_Cyanistes_caeruleus'
    dirs['European_Starling'] = '04202_Animalia_Chordata_Aves_Passeriformes_Sturnidae_Sturnus_vulgaris'
    dirs['Eurasian_Blackbird'] = '04284_Animalia_Chordata_Aves_Passeriformes_Turdidae_Turdus_merula'
    dirs['Wood_Pigeon'] = '03516_Animalia_Chordata_Aves_Columbiformes_Columbidae_Columba_palumbus'
    dirs['European_Robin'] = '03978_Animalia_Chordata_Aves_Passeriformes_Muscicapidae_Erithacus_rubecula'
    dirs['Great_Tit'] = '04034_Animalia_Chordata_Aves_Passeriformes_Paridae_Parus_major'
    dirs['Eurasian_Goldfinch'] = '03809_Animalia_Chordata_Aves_Passeriformes_Fringillidae_Carduelis_carduelis'
    dirs['Eurasian_Magpie'] = '03775_Animalia_Chordata_Aves_Passeriformes_Corvidae_Pica_pica'
    dirs['Long-tailed_Tit'] = '03688_Animalia_Chordata_Aves_Passeriformes_Aegithalidae_Aegithalos_caudatus'

    # Greenfinch
    # Bullfinch
    # Wren
    # Carrion Crow
    # Jackdaw
    # Jay
    # Dunnock
    # Chaffinch
    # Goldcrest
    
    #
    # extra birds
    #
    dirs['Red_Kite'] = '03172_Animalia_Chordata_Aves_Accipitriformes_Accipitridae_Milvus_milvus'
    dirs['Grey_Heron'] = '04356_Animalia_Chordata_Aves_Pelecaniformes_Ardeidae_Ardea_cinerea'
    dirs['Blackcap'] = '04204_Animalia_Chordata_Aves_Passeriformes_Sylviidae_Sylvia_atricapilla'
    dirs['Redwing'] = '04281_Animalia_Chordata_Aves_Passeriformes_Turdidae_Turdus_ignobilis'
    dirs['Eurasian_Green_Woodpecker'] = '04462_Animalia_Chordata_Aves_Piciformes_Picidae_Picus_viridis'
    dirs['Eurasian_Collared_Dove'] = '03536_Animalia_Chordata_Aves_Columbiformes_Columbidae_Streptopelia_decaocto'

    #
    # common garden animals
    #
    dirs['Common_Hedgehog'] = '04745_Animalia_Chordata_Mammalia_Eulipotyphla_Erinaceidae_Erinaceus_europaeus'
    dirs['Red_Fox'] = '04677_Animalia_Chordata_Mammalia_Carnivora_Canidae_Vulpes_vulpes'
    dirs['Eastern_Gray_Squirrel'] = '04844_Animalia_Chordata_Mammalia_Rodentia_Sciuridae_Sciurus_carolinensis'
    dirs['Domestic_Cat'] = '04679_Animalia_Chordata_Mammalia_Carnivora_Felidae_Felis_catus'
    dirs['Brown_Rat'] = '04814_Animalia_Chordata_Mammalia_Rodentia_Muridae_Rattus_norvegicus'

    #
    # people
    #
    # from voc ... although dramaticly deacreases mAP
    #
    dirs['Person'] = 'VOCdevkit'

    #
    # weeds
    # 
    dirs['Common_Dandelion'] = '07039_Plantae_Tracheophyta_Magnoliopsida_Asterales_Asteraceae_Taraxacum_officinale'

    os.makedirs(destination, exist_ok=True)
    for name, directory in dirs.items():
        for filename in glob.glob(os.path.join(source, directory, '*.*')):
            shutil.copy(filename, destination)

    return dirs

#
# label images if missing
#
def label_img(data_dir):
    #
    # copy to tmp dir first so we only edit necessary files
    #
    tmp_data_dir = Path('images/tmp')
    with open('labelmap.txt', "w") as file_out:
        for name in classes.keys():
            print (name, file=file_out)
    for thisdirpath, subdirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jpg"):
                xml=os.path.splitext(file)[0]+'.xml'
                if not os.path.isfile(os.path.join(thisdirpath, xml)):
                    os.makedirs(tmp_data_dir, exist_ok=True)    
                    shutil.copyfile(os.path.join(thisdirpath, file), os.path.join(tmp_data_dir, file))

    if os.path.exists(tmp_data_dir):
        os.system('labelImg '+str(tmp_data_dir)+' labelmap.txt')
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
all_data_dir = Path('images/all')
boundingbox_data_dir = Path('boundingbox')

#
# download data
#
download_extract_tgz('https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz', 'train')
download_extract_tgz('https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz', 'val')
#download_extract_tgz('https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.tar.gz', 'public_test')
download_extract_tgz('https://plord.co.uk/garden.tgz', 'garden')

#
# extract images we are interested in
#
classes = copy_images('train', all_data_dir);
copy_images('val', all_data_dir);
for filename in glob.glob('garden/*'):
  shutil.copy(filename, all_data_dir)

#
# copy any bounding boxes in
#
print ('Copying bounding box xml data')
for filename in glob.glob(os.path.join(boundingbox_data_dir, '*.xml')):
  shutil.copy(filename, all_data_dir)

#
# use _ in name to fool later stats
#
for thisdirpath, subdirs, files in os.walk('images'):
    for file in files:
        if file.endswith(".xml"):
            source = os.path.join(thisdirpath, file)
            dest = os.path.join(thisdirpath, file+"new")
            with open(source,'r') as file_in:
                with open(dest, "w") as file_out:
                    for line in file_in:
                        newline = line
                        if '<name>' in newline:
                            newline = re.sub(' ', '_', newline)
                        print (newline, end = "", file=file_out)
            shutil.move(dest, source)

#
# download voc ( for people )
#
download_extract_tar('http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar', 'VOCdevkit')

#
# copy any bounding boxes in
#
print ('Finding only person')
pattern = re.compile(r'<name>([^<]*)</name>')
i=0
for filename in glob.glob('VOCdevkit/VOC2012/Annotations/*.xml'):
    with open(filename) as xmlfile:
        names = dict()
        contents = xmlfile.read()
        for (name) in re.findall(pattern, contents):
            names[name] = 1
        if len(names) == 1 and names.get('person') == 1 and i < 250:
            image='VOCdevkit/VOC2012/JPEGImages/'+(Path(filename).stem)+'.jpg'
            text_file = open(os.path.join(all_data_dir,Path(filename).name), "w")
            n = text_file.write(contents.replace('<name>person</name>', '<name>Person</name>'))
            text_file.close()
            shutil.copy(image, all_data_dir)
            i=i+1

#
# run labelImg
#
label_img(all_data_dir)

#
# backup xml (for any further data prep runs)
#
for thisdirpath, subdirs, files in os.walk('images'):
    for file in files:
        if file.endswith(".xml"):
            source = os.path.join(thisdirpath, file)
            dest = os.path.join(boundingbox_data_dir, file)
            if not os.path.exists(dest):
                print('Backing up '+source+' to '+dest)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(source, dest)

os.system('mkdir -p images/train images/validation images/test images/difficult')

#
# move difficult ones out of the way
#
for thisdirpath, subdirs, files in os.walk('images'):
    for file in files:
        if file.endswith(".xml"):
            xml = os.path.join(thisdirpath, file)
            jpg = os.path.join(thisdirpath,os.path.splitext(file)[0]+'.jpg')
            difficult=False
            with open(xml,'r') as file_in:
                for line in file_in:
                    newline = line
                    if '<difficult>1</difficult>' in newline:
                        difficult=True
            if difficult:
                shutil.move(xml, os.path.join("images/difficult", file))
                shutil.move(jpg, os.path.join("images/difficult", os.path.splitext(file)[0]+'.jpg'))


# Define paths to image folders
image_path = 'images/all'
train_path = 'images/train'
val_path = 'images/validation'
test_path = 'images/test'

# Get list of all images
jpg_file_list = glob.glob(image_path + '/*.jpg')
JPG_file_list = glob.glob(image_path + '/*.JPG')
png_file_list = glob.glob(image_path + '/*.png')
bmp_file_list = glob.glob(image_path + '/*.bmp')

file_list = jpg_file_list + JPG_file_list + png_file_list + bmp_file_list
file_num = len(file_list)
print('Total images: %d' % file_num)

# Determine number of files to move to each folder
train_percent = 0.8  # 80% of the files go to train
val_percent = 0.1 # 10% go to validation
test_percent = 0.1 # 10% go to test
train_num = int(file_num*train_percent)
val_num = int(file_num*val_percent)
test_num = file_num - train_num - val_num
print('Files moving to train: %d' % train_num)
print('Files moving to validation: %d' % val_num)
print('Files moving to test: %d' % test_num)

# Select 80% of files randomly and move them to train folder
for i in range(train_num):
    move_me = random.choice(file_list)
    fn = move_me.split('/')[-1]
    base_fn = fn[:(len(fn)-4)] # Gets rid of .jpg, .png, or .bmp at end of the string
    xml_fn = base_fn + '.xml'
    os.rename(move_me, train_path+'/'+fn)
    os.rename(os.path.join(image_path,xml_fn),os.path.join(train_path,xml_fn))
    file_list.remove(move_me)

# Select 10% of remaining files and move them to validation folder
for i in range(val_num):
    move_me = random.choice(file_list)
    fn = move_me.split('/')[-1]
    base_fn = fn[:(len(fn)-4)] # Gets rid of .jpg, .png, or .bmp at end of the string
    xml_fn = base_fn + '.xml'
    os.rename(move_me, val_path+'/'+fn)
    os.rename(os.path.join(image_path,xml_fn),os.path.join(val_path,xml_fn))
    file_list.remove(move_me)

# Move remaining files to test folder
for i in range(test_num):
    file = file_list[i]
    fn = file.split('/')[-1]
    base_fn = fn[:(len(fn)-4)] # Gets rid of .jpg, .png, or .bmp at end of the string
    xml_fn = base_fn + '.xml'
    os.rename(file, test_path+'/'+fn)
    os.rename(os.path.join(image_path,xml_fn),os.path.join(test_path,xml_fn))


#
# create label map
#
#count=1
#with open('label_map.pbtxt', "w") as file_out:
#    for name in classes.keys():
#        print ('item {', file=file_out)
#        print ('    id: '+str(count), file=file_out)
#        print ('    name: \''+name+'\'', file=file_out)
#        print ('}', file=file_out)
#        count=count+1

# Create CSV data files and TFRecord files
os.system('python3 create_csv.py')
os.system('python3 create_tfrecord.py --csv_input=images/train_labels.csv --labelmap=labelmap.txt --image_dir=images/train --output_path=train.tfrecord')
os.system('python3 create_tfrecord.py --csv_input=images/validation_labels.csv --labelmap=labelmap.txt --image_dir=images/validation --output_path=val.tfrecord')
