import os, sys
import numpy as np
import time

datapath = sys.argv[1]

runs = ['train','test']
cams = ['right','left']

for run in runs:
	for cam in cams:
		times = [i[:-4] for i in os.listdir(os.path.join(datapath,run,cam)) if i.endswith('.png')]
		with open(datapath+'%s.%s.times.txt'%(run,cam),'w') as fw:
			for t in times:
				fw.write('%s\n'%(t))







