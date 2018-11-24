./build/tools/caffe.bin train \
-solver  ./examples/cifar10/solver_cifar10.prototxt \
-weights ./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
-gpu all 2>&1 | tee ./examples/cifar10/SSDH_log.txt
