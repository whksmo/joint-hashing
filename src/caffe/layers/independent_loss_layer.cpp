#include <vector>

#include "caffe/layers/independent_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void IndependentLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  diff_.ReshapeLike(*bottom[0]);

  I_.ReshapeLike(*bottom[0]);
  Dtype* I_data = I_.mutable_cpu_data();
  int N_ = I_.count(0,1);
  int M_ = I_.count(1);
//   CHECK_EQ(N_,48)
//     << "N should equal to 48";
  CHECK_EQ(N_,M_)
    <<"N:"<< N_ << "and M:" << M_;
  for (int i = 0;i<N_;i++){
    for (int j = 0; j<M_; j++){
      if(i == j)
        I_data[i*M_+j] = 1;
      else
        I_data[i*M_+j] = 0;
    }
  }
}

template <typename Dtype>
void IndependentLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int N_ = I_.count(0,1);
  int M_ = I_.count(1);
  int count = bottom[0]->count();
  CHECK_EQ(I_.count(),count)
    << "the count should be equal"; 
  CHECK_EQ(I_.count(),M_*N_)
    << "the count should be equal to N_*M_"; 
  // caffe_sub(
  //     count,
  //     bottom[0]->cpu_data(),
  //     I_.cpu_data(),
  //     diff_.mutable_cpu_data());
  // Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  // Dtype loss = dot / bottom[0]->count() / Dtype(2);

  Dtype loss = 0;
  for (int i = 0;i<N_;i++){
    for (int j = 0; j<M_; j++){
      Dtype sub;
      if(i == j)
        sub = bottom[0]->cpu_data()[i*M_+j]-Dtype(1);
      else
        sub = bottom[0]->cpu_data()[i*M_+j];
      diff_.mutable_cpu_data()[i*M_+j] = sub;
      Dtype dot = sub * sub;
      loss += dot;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss/(Dtype(2)*count);
}

template <typename Dtype>
void IndependentLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    CHECK_EQ(propagate_down[0],1);
    if (propagate_down[0]) {
      const Dtype sign = 1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[0]->count();
      caffe_cpu_axpby(
          bottom[0]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[0]->mutable_cpu_diff());  // b
    }
}

#ifdef CPU_ONLY
STUB_GPU(IndependentLossLayer);
#endif

INSTANTIATE_CLASS(IndependentLossLayer);
REGISTER_LAYER_CLASS(IndependentLoss);

}  // namespace caffe
