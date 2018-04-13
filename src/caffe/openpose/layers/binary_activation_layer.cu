#include <algorithm>
#include <vector>

#include "caffe/openpose/layers/binary_activation_layer.hpp"
#include "caffe/openpose/gpu.hu"

namespace caffe {

// template <typename Dtype>
// __global__ void BinaryActivationForward(const int n, const Dtype* in, Dtype* out) {
//   CUDA_KERNEL_LOOP(index, n) {
//     out[index] = in[index] < 0 ? -1 : 1;
//   }
// }

template <typename Dtype>
void BinaryActivationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // // Option a - Weight = +-1
  // const Dtype* bottom_data = bottom[0]->gpu_data();
  // Dtype* top_data = top[0]->mutable_gpu_data();
  // const int count = bottom[0]->count();
  // // NOLINT_NEXT_LINE(whitespace/operators)
  // BinaryActivationForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
  //     count, bottom_data, top_data);
  // CUDA_POST_KERNEL_CHECK;
  // // << " count: " << count << " bottom_data: "
  // //     << (unsigned long)bottom_data
  // //     << " top_data: " << (unsigned long)top_data
  // //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  // //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
  // Option b - Weight = +-n
  const auto count = bottom[0]->count();
  const auto weightArea = bottom[0]->shape()[2] * bottom[0]->shape()[3];
  const auto countReduced = count/weightArea;
  normalizeWeightsGpu<<<CAFFE_GET_BLOCKS(countReduced), CAFFE_CUDA_NUM_THREADS>>>(
    top[0]->mutable_gpu_data(), bottom[0]->gpu_data(), countReduced, weightArea);
  CUDA_POST_KERNEL_CHECK;
}

// Binary replaced
// template <typename Dtype>
// __global__ void BinaryActivationBackward(const int n, const Dtype* in_diff,
//     const Dtype* in_data, Dtype* out_diff) {
//   CUDA_KERNEL_LOOP(index, n) {
//     out_diff[index] = max(Dtype(-1), min(Dtype(1), in_diff[index]));
//   }
// }

// template <typename Dtype>
// __global__ void CopyBackward(const int n, const Dtype* in_diff, Dtype* out_diff) {
//   CUDA_KERNEL_LOOP(index, n) {
//     out_diff[index] = in_diff[index];
//   }
// }
// Binary replaced ended

template <typename Dtype>
void BinaryActivationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // // Option a - Weight = +-1
    // // NOLINT_NEXT_LINE(whitespace/operators)
    // BinaryActivationBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    //     count, top_diff, bottom_data, bottom_diff);
    // Option b - Weight = +-n
    const auto weightArea = top[0]->shape()[2] * top[0]->shape()[3];
    const auto countReduced = count/weightArea;
    backwardNormalizeWeightsGpu<<<CAFFE_GET_BLOCKS(countReduced), CAFFE_CUDA_NUM_THREADS>>>(
      bottom_diff, top_diff, bottom_data, countReduced, weightArea);
    // // Option c - No backprop
    // const Dtype* bottom_data = bottom[0]->gpu_data();
    // CopyBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    //     count, top_diff, bottom_diff);
    // Security checks
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(BinaryActivationLayer);


}  // namespace caffe
