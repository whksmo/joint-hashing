./build/tools/caffe.bin train \
-solver  ./examples/bone-finetune/solver_bone.prototxt \
-weights ./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
-gpu 2 2>&1 | tee ./examples/bone-finetune/log_SSDH.txt
