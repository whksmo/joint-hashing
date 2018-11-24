~/repository/caffe/build/tools/caffe.bin train \
-solver  ~/repository/caffe/examples/cifar10/solver_cifar10_modify.prototxt \
-weights ~/repository/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
-gpu all 2>&1 | tee ~/repository/caffe/examples/cifar10/perceptual_log.txt
