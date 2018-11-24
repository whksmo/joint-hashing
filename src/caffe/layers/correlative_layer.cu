#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/correlative_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CorrelativeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
 
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                        K_, K_, M_, (Dtype)(1. / M_),
                        bottom_data, bottom_data, (Dtype)0., top_data);
}

template <typename Dtype>
void CorrelativeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool> &propagate_down, 
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // CHECK_EQ(top[0]->count(),48*48)
    //  <<"inequal count!";
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
        M_, K_, K_,
        (Dtype)(2. / M_), bottom[0]->gpu_data(), top_diff,
        (Dtype)0., bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(CorrelativeLayer);

}  // namespace caffe
