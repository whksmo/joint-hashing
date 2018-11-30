./build/tools/caffe.bin train \
-solver  ./examples/cifar100/solver_cifar100.prototxt \
-weights ./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
-gpu all 2>&1 | tee ./examples/cifar100/log.txt
