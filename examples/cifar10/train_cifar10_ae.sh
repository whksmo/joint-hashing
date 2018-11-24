./build/tools/caffe.bin train \
-solver  ./examples/cifar10/solver_cifar10_ae.prototxt \
-weights ./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
-gpu all 2>&1 | tee ./examples/cifar10/SSDH_AE_log.txt
