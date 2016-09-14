import numpy as np
import os
import matplotlib.pyplot as plt

X = np.loadtxt(open('results.txt'))

h = np.histogram(X[:,0], bins=np.arange(0,np.max(X[:,0]),.1))
plt.plot(.5*h[1][:-1]+.5*h[1][1:],h[0])
plt.grid()
plt.title('location error histogram (m)')
plt.savefig('hist_loc.png',ppi=300)
plt.clf()

h = np.histogram(X[:,1], bins=np.arange(0,np.max(X[:,1]),.1))
plt.plot(.5*h[1][:-1]+.5*h[1][1:],h[0])
plt.grid()
plt.title('angular error histogram (degrees)')
plt.savefig('hist_angle.png',ppi=300)