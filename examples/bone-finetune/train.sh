./build/tools/caffe.bin train -solver  ./examples/bone-finetune/solver.prototxt -weights ./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu 1 2>&1 | tee log.txt
