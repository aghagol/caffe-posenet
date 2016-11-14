caffe_root = '/home/mo/github/caffe-posenet/'  # Change to your directory to caffe-posenet
import sys, os
sys.path.insert(0, caffe_root + 'python')

import numpy as np
import lmdb
import caffe
import random
import cv2

directory = '/home/mo/Desktop/ROS_data/pioneer/2016-04-27/'
dataset = 'train'

poses = []
images = []

images_left = [i for i in os.listdir('%s/%s/left'%(directory,dataset)) if i.endswith('png')]
images_left.sort()
images_dict = dict(zip(images_left,range(len(images_left))))

with open(directory+('dataset_%s.txt'%dataset)) as f:
    for idx, line in enumerate(f):
        fname, p0,p1,p2,p3,p4,p5,p6 = line.split()
        p0 = float(p0)
        p1 = float(p1)
        p2 = float(p2)
        p3 = float(p3)
        p4 = float(p4)
        p5 = float(p5)
        p6 = float(p6)
        poses.append((p0,p1,p2,p3,p4,p5,p6))
        images.append(images_dict[fname+'.png'])

r = list(range(len(images)))
random.shuffle(r)

print 'Creating PoseNet Dataset.'
env = lmdb.open(directory+dataset+'_lmdb_siamese_224', map_size=int(1e12))

count = 0

for i in r:

    if (count+1) % 100 == 0:
        print 'Saving image: ', count+1

    image_left = '%s/%s/image_0/%06d.png'%(directory,dataset,images[i])
    X_left = cv2.imread(image_left)
    # X_left = cv2.resize(X_left, (344,256))
    X_left = cv2.resize(X_left, (224,224))
    X_left = np.transpose(X_left,(2,0,1))
    image_rite = '%s/%s/image_1/%06d.png'%(directory,dataset,images[i])
    X_rite = cv2.imread(image_rite)
    # X_rite = cv2.resize(X_rite, (344,256))
    X_rite = cv2.resize(X_rite, (224,224))
    X_rite = np.transpose(X_rite,(2,0,1))
    X = np.concatenate((X_left,X_rite),axis=0)
    im_dat = caffe.io.array_to_datum(np.array(X).astype(np.uint8))

    im_dat.float_data.extend(poses[i])
    str_id = '{:0>10d}'.format(count)
    with env.begin(write=True) as txn:
        txn.put(str_id, im_dat.SerializeToString())
    count = count+1

env.close()

