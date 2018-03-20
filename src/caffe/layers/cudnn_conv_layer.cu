#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {

__global__ void sync_conv_groups() { }

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Binary added
  if (this->layer_param_.convolution_param().binary())
  {
    // TRAIN commented
    // // TRAIN
    // if (this->phase_ == TRAIN)
    //   normalizeWeights();
    // TEST (only first time)
    // else if (!weight_initialized_)
    if (!weight_initialized_)
    {
      weight_initialized_ = true;
      if (this->phase_ == TEST)
      {
        CHECK_GE(this->blobs_.size(), 1);
        CHECK_GT(this->blobs_[0]->shape().size(), 2u);
        weight_binary_.reset(new Blob<Dtype>());
        weight_binary_->Reshape(this->blobs_[0]->shape());
        // Data to weightReal
        normalizeWeights();
      }
    }
  }
  // Binary added end

  // const Dtype* weight = this->blobs_[0]->gpu_data(); // Binary commented
  // Binary added
  // const Dtype* weight = (this->layer_param_.convolution_param().binary() && this->phase_ == TRAIN
  //   ? weight_binary_->gpu_data() : this->blobs_[0]->gpu_data());
  // const Dtype* weight = (this->layer_param_.convolution_param().binary()
  //   ? weight_binary_->gpu_data() : this->blobs_[0]->gpu_data());
  const Dtype* weight = (this->layer_param_.convolution_param().binary() && this->phase_ == TEST
    ? weight_binary_->gpu_data() : this->blobs_[0]->gpu_data());
  // Binary added ended
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + this->weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    // weight = this->blobs_[0]->gpu_data(); // Binary commented
    // Binary added
    // My binary way
    weight = this->blobs_[0]->gpu_data();
    // Plain truncating
    // weight = (this->layer_param_.convolution_param().binary()
    //           ? weight_binary_->gpu_data() : this->blobs_[0]->gpu_data());
    // Binary added ended
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_filter_algo_[i], workspace[1*this->group_ + g],
              workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + this->weight_offset_ * g));
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          // weight = this->blobs_[0]->gpu_data(); // Binary commented
          // Binary added
          weight = (this->layer_param_.convolution_param().binary()
                    ? weight_binary_->gpu_data() : this->blobs_[0]->gpu_data());
          // Binary added ended
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight + this->weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[2*this->group_ + g],
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
  // } // Binary added
  // Binary added
  // // If binary (XNOR-style)
  // if (this->layer_param_.convolution_param().binary()) // Binary added
  // {
  //   if (this->param_propagate_down_[0]) {
  //     // Channel area = volume from axis 2 to final (num, channel, h, w)
  //     const auto* weight_real = this->blobs_[0]->cpu_data();
  //     auto* weight_real_diff = this->blobs_[0]->mutable_cpu_diff();
  //     const auto channelArea = weight_binary_->count(1);
  //     const auto imageArea = weight_binary_->count(2);
  //     const auto oneOverN = 1/Dtype(imageArea);
  //     for (auto num = 0 ; num < weight_binary_->shape()[0] ; num++)
  //     {
  //       const auto offsetNum = num*channelArea;
  //       for (auto channel = 0 ; channel < weight_binary_->shape()[1] ; channel++)
  //       {
  //         const auto offset = offsetNum + channel * imageArea;
  //         // L1 norm
  //         auto l1Norm = Dtype(0);
  //         for (auto i = 0 ; i < imageArea ; i++)
  //           l1Norm += (weight_real[offset+i] < 0 ? -weight_real[offset+i] : weight_real[offset+i]);
  //         // Update weight_real_diff
  //         for (auto i = 0 ; i < imageArea ; i++)
  //           weight_real_diff[offset+i] = weight_real_diff[offset+i] * oneOverN
  //                                      * (1 + l1Norm * std::max(Dtype(-1), std::min(Dtype(1), weight_real[offset+i])));
  //       }
  //     }
  //   }
  // }
  // My binary way (guiding weights to 1)
  if (this->layer_param_.convolution_param().binary()) // Binary added
  {
    if (this->param_propagate_down_[0]) {
      const auto lambda = 0.01f;
      const auto* const weight_real = this->blobs_[0]->cpu_data();
      auto* weight_real_diff = this->blobs_[0]->mutable_cpu_diff();
// std::cout << this->blobs_[0]->count() << ": " << this->blobs_[0]->shape()[0] << " " << this->blobs_[0]->shape()[1] << " " << this->blobs_[0]->shape()[2] << " " << this->blobs_[0]->shape()[3] << std::endl;
      for (auto index = 0 ; index < this->blobs_[0]->count() ; index++)
        weight_real_diff[index] += 2*lambda*(   weight_real[index] - (weight_real[index] < 0 ? -1 : 1)   );
    }
  }
  // Binary added end
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
