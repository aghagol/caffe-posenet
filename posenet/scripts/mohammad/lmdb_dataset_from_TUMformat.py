caffe_root = '../../../' #this should be caffe that came with posenet
import sys, os
sys.path.insert(0, caffe_root + 'python')

import numpy as np
import lmdb
import caffe
import random
import cv2

datapath = sys.argv[1]

runs = ['train','test']

for run in runs:

	print 'creating LMDB for %s' %(datapath+run)

	poses = []
	images = []

	with open(datapath+'orbslam_pose_%s_corrected.txt'%run) as f:
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
	        images.append('%s/%s/left/%s.png'%(datapath,run,fname))

	r = list(range(len(images)))
	random.shuffle(r)

	env = lmdb.open(datapath+'%s_lmdb'%run, map_size=int(1e12))

	for count, i in enumerate(r):
	    X = cv2.imread(images[i])
	    # to reproduce PoseNet results, please resize the images so that the shortest side is 256 pixels
	    X = cv2.resize(X, (344,256))
	    X = np.transpose(X,(2,0,1))
	    im_dat = caffe.io.array_to_datum(np.array(X).astype(np.uint8))
	    im_dat.float_data.extend(poses[i])
	    str_id = '{:0>10d}'.format(count)
	    with env.begin(write=True) as txn:
	        txn.put(str_id, im_dat.SerializeToString())

	env.close()

