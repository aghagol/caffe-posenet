#!/bin/bash

DATAPATH="/home/mo/Desktop/posgit2/"
python scripts/mohammad/timestamps_from_filenames.py $DATAPATH
python scripts/mohammad/fix_ros_orbslam_timestamps.py $DATAPATH
python scripts/mohammad/lmdb_dataset_from_TUMformat.py $DATAPATH
../build/tools/compute_image_mean "$DATAPATH/train_lmdb" "$DATAPATH/imagemean.binaryproto"
