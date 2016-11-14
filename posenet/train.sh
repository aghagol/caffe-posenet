#!/bin/bash

DATAPATH="/home/mo/Desktop/posgit2/"
../build/tools/caffe train --solver=models/solver_posenet.prototxt --weights="$DATAPATH/weights_kingscollege.caffemodel"
