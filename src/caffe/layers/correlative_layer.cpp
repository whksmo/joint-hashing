#include <vector>

#include "caffe/layers/correlative_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{


template <typename Dtype>
void CorrelativeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int axis = 1;
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  M_ = bottom[0]->count(0, axis);
  // CHECK_EQ(K_,48)
    // <<K_;
  // Check if we need to set up the weights
}

template <typename Dtype>
void CorrelativeLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top)
{
  const int axis = 1;

  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[0] = K_;
  top_shape[axis] = K_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void CorrelativeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top)
{
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  // H: [M_ * K_]
  // top = H^T * H : [K_ * K_]
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                        K_, K_, M_, (Dtype)(1. / M_),
                        bottom_data, bottom_data, (Dtype)0., top_data);
}

template <typename Dtype>
void CorrelativeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                            const vector<bool> &propagate_down,
                                            const vector<Blob<Dtype> *> &bottom)
{

    const Dtype *top_diff = top[0]->cpu_diff();
    // CHECK_EQ(top[0]->count(),48*48)
      // <<"inequal count!";
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                          M_, K_, K_,
                          (Dtype)(2. / M_), bottom[0]->cpu_data(),top_diff,
                          (Dtype)0., bottom[0]->mutable_cpu_diff());

}

#ifdef CPU_ONLY
STUB_GPU(CorrelativeLayer);
#endif

INSTANTIATE_CLASS(CorrelativeLayer);
REGISTER_LAYER_CLASS(Correlative);

} // namespace caffe
