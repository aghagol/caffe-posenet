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
# dataset = 'dataset_test.txt'

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
        images.append('%s/%s/LRB/%05d.png'%(directory,dataset,images_dict[fname+'.png']))

r = list(range(len(images)))
# random.shuffle(r)

print 'Creating PoseNet Dataset.'
env = lmdb.open(directory+dataset+'_lmdb224', map_size=int(1e12))

count = 0

for i in r:
    if (count+1) % 100 == 0:
        print 'Saving image: ', count+1
    X = cv2.imread(images[i])
    X = cv2.resize(X, (224,224))    # to reproduce PoseNet results, please resize the images so that the shortest side is 256 pixels
    X = np.transpose(X,(2,0,1))
    im_dat = caffe.io.array_to_datum(np.array(X).astype(np.uint8))
    im_dat.float_data.extend(poses[i])
    str_id = '{:0>10d}'.format(count)
    with env.begin(write=True) as txn:
        txn.put(str_id, im_dat.SerializeToString())
    count = count+1

env.close()

