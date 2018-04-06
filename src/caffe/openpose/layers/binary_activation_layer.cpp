#include <algorithm>
#include <vector>

#include "caffe/openpose/layers/binary_activation_layer.hpp"

namespace caffe {

template <typename Dtype>
void BinaryActivationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i] < 0 ? Dtype(-1) : Dtype(1);
  }
}

template <typename Dtype>
void BinaryActivationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = std::max(Dtype(-1), std::min(Dtype(1), top_diff[i]));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(BinaryActivationLayer);
#endif

INSTANTIATE_CLASS(BinaryActivationLayer);
REGISTER_LAYER_CLASS(BinaryActivation);
}  // namespace caffe
