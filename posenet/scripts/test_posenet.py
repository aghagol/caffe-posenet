import numpy as np
# import matplotlib.pyplot as plt
# import os.path
# import json
# import scipy
import time
import argparse
import math
# import pylab
# from sklearn.preprocessing import normalize
# from mpl_toolkits.mplot3d import Axes3D
import os
os.environ['GLOG_minloglevel'] = '2'

# Make sure that caffe is on the python path:
caffe_root = '/home/mo/github/caffe-posenet/'  # Change to your directory to caffe-posenet
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# data_name = 'vggpose_fc6'
data_name = 'kingscollege'
model_path = '/home/mo/github/caffe-posenet/posenet/models/pioneer/'
net_model = model_path + 'train224_%s.prototxt' % data_name
net_weits = model_path + 'weights_%s.caffemodel' % data_name

# Import arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, required=True)
# parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

results = np.zeros((args.iter,2))

caffe.set_mode_gpu()
# caffe.set_mode_cpu()

# net = caffe.Net(args.model,
#                 args.weights,
#                 caffe.TEST)

net = caffe.Net(net_model, net_weits, caffe.TEST)

t1 = time.time()

for i in range(0, args.iter):

	net.forward()

# print 'time per image = %f' %((time.time()-t1)/args.iter)

	pose_q= net.blobs['label_wpqr'].data
	pose_x= net.blobs['label_xyz'].data
	predicted_q = net.blobs['cls3_fc_wpqr'].data 
	predicted_x = net.blobs['cls3_fc_xyz'].data 
	# predicted_q = net.blobs['fc_pose_wpqr'].data 
	# predicted_x = net.blobs['fc_pose_xyz'].data 

	pose_q = np.squeeze(pose_q)
	pose_x = np.squeeze(pose_x)
	predicted_q = np.squeeze(predicted_q)
	predicted_x = np.squeeze(predicted_x)

	#Compute Individual Sample Error
	q1 = pose_q / np.linalg.norm(pose_q)
	q2 = predicted_q / np.linalg.norm(predicted_q)
	d = abs(np.sum(np.multiply(q1,q2)))
	theta = 2 * np.arccos(d) * 180/math.pi
	error_x = np.linalg.norm(pose_x-predicted_x)

	results[i,:] = [error_x,theta]

	print 'Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta

median_result = np.median(results,axis=0)
print 'Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.'

np.savetxt('results.txt', results, delimiter=' ',fmt='%08.5f')

print 'Success!'

