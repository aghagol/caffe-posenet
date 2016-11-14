import matplotlib.pyplot as plt
import numpy as np
import os, sys

print 'Fixing timestamps to match with filenames'

datapath = sys.argv[1]

runs = ['train','test']

for run in runs:

	# load orbslam pose output and reference (true) timestamps
	orbslam_ts = np.loadtxt(datapath+'orbslam_pose_%s.txt'%run)[:,0]
	correct_ts = np.loadtxt(datapath+'%s.left.times.txt'%run)

	# search for the closest timestamp in correct_ts
	nn = np.zeros_like(orbslam_ts) #nearest neighbor
	it = iter(correct_ts)
	t1 = next(it, None)
	t0 = t1
	for i, t in enumerate(orbslam_ts):
		while t1<=t:
			t0 = t1
			t1 = next(it, None)
		nn[i] = t0

	# write poses with corrected timestamps into a new txt file
	with open(datapath+'orbslam_pose_%s_corrected.txt'%run,'w') as fw:
		with open(datapath+'orbslam_pose_%s.txt'%run) as fr:
			for i,s in enumerate(fr):
				fw.write('%f '%nn[i])
				fw.write(' '.join(s.split()[1:]))
				fw.write('\n')

