./build/tools/caffe.bin train \
-solver  ./examples/cifar10/solver_cifar10_feature.prototxt \
-weights ./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
-gpu 0 2>&1 | tee ./examples/cifar10/SSDH_feature_log.txt
