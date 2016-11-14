First, adjust the "DATAPATH" variable in `prepare.sh` and `train.sh`. I have created a sample dataset in `P:/Projects/DeepLearning/Dataset/posgit2`.

Then, run `prepare.sh` to create the LMDB database and the mean file.

To train, run `train.sh`. Before this, you should correct the paths in `models/solver_posenet.prototxt` and `models/train_posenet.prototxt`.

Good luck :)
