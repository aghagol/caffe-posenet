import sys, os, pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
# os.environ['GLOG_minloglevel'] = '2'
caffe_root = '/home/mo/github/caffe-posenet'
sys.path.append(os.path.join(caffe_root,'python'))
import caffe
caffe.set_mode_gpu()

'''//////////////////////////// parameters //////////////////////////////// '''

net_name 		= 'left'
net_model_def 	= 'train_siamese.prototxt'
# net_model_def 	= 'train_siamese_left.prototxt'
# net_model_def 	= 'train_left.prototxt'
net_weights 	= 'left.caffemodel'

''' /////////////////////////////////////////////////////////////////////// '''

print 'loading ' + net_name
net = caffe.Net(net_model_def, net_weights, caffe.TEST)
print 'done'

# net.save('initial_siamese.caffemodel')


