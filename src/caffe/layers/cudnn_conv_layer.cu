#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"
  // Binary added
#include "caffe/openpose/gpu.hu"
  // Binary added end

namespace caffe {

__global__ void sync_conv_groups() { }

// // Binary weights = +-1
// template <typename Dtype>
// __global__ void normalizeWeightsGpuBinary(Dtype* weightBinaryData, Dtype* weightRealData, const int count)
// {
//   const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (globalIdx < count)
//   {
//     weightRealData[globalIdx] = max(-Dtype(1), min(Dtype(1), weightRealData[globalIdx]));
//     weightBinaryData[globalIdx] = (weightRealData[globalIdx] < 0 ? -Dtype(1) : Dtype(1));
//   }
// }

// Float data into binary data
template <typename Dtype>
__global__ void getBinaryData(Dtype* binaryData, const Dtype* realData, const int count)
{
  const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalIdx < count)
    binaryData[globalIdx] = (realData[globalIdx] < 0 ? Dtype(-1) : Dtype(1));
}

// NxCxHxW --> Nx1xHxW
// output(n,1,h,w) = sum(abs(input(n,:,h,w)))
template <typename Dtype>
__global__ void addOverChannels(Dtype* outputData, const Dtype* inputData, const int bottomChannels,
                                const int bottomWHArea)
{
  const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalIdx < bottomWHArea)
  {
    auto& output = outputData[globalIdx];
    output = 0;
    for (auto i = 0 ; i < bottomChannels ; i++)
    {
      const auto value = inputData[globalIdx+i*bottomWHArea];
      output += (value < 0 ? -value : value);
    }
  }
}

// Float data into binary data
template <typename Dtype>
__global__ void multiplyOverChannels(Dtype* outputData, const Dtype* multiplierData, const int topChannels,
                                     const int topWHArea)
{
  const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalIdx < topWHArea)
  {
    for (auto i = 0 ; i < topChannels ; i++)
    {
      outputData[globalIdx+i*topChannels] *= multiplierData[globalIdx];
    }
  }
}

// // Binary weights = +-n - XNOR-style
// template <typename Dtype>
// __global__ void normalizeL1Norm(Dtype* binaryData, Dtype* aData, Dtype* kData,
//                                 const Dtype* realData, const int count, const int binarizationArea)
// {
//   const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (globalIdx < count)
//   {
//     // Offset
//     const auto offset = globalIdx * binarizationArea;
//     const auto* weightRealDataOffset = &realData[offset];
//     auto* weightBinaryDataOffset = &binaryData[offset];
//     // XNOR-style
//     // L1 norm
//     const auto l1Norm = getL1Norm(weightRealDataOffset, binarizationArea);
//     // Update output
//     const auto alphaOptimal = l1Norm / binarizationArea;
//     for (auto i = 0 ; i < binarizationArea ; i++)
//       weightBinaryDataOffset[i] = (weightRealDataOffset[i] < 0 ? -1 : 1);
//   }
// }

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::binarizeWeightsAndInputGpu(Blob<Dtype>* weight_binary_,
  Blob<Dtype>* bottom_binary_, Blob<Dtype>* matrix_A_, Blob<Dtype>* matrix_K_, const Blob<Dtype>* const matrix_one_over_chw,
  const boost::shared_ptr<caffe::Blob<Dtype>>& this_blobs_0, const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top, const int num, const int binaryOption)
{
  // Binary weights
  if (binaryOption > 0)
  {
    // // Option a - Weight = +-1
    // const auto count = this_blobs_0->count();
    // normalizeWeightsGpuBinary<<<CAFFE_GET_BLOCKS(count/binarizationArea), CAFFE_CUDA_NUM_THREADS>>>(
    //   weight_binary_->mutable_gpu_data(), this_blobs_0->mutable_gpu_data(), count);
    // Option b - Weight = +-n per w,h
    if (binaryOption == 1)
    {
      const auto count = this_blobs_0->count();
      const auto binarizationArea = this_blobs_0->count(2);
      const auto countReduced = count/binarizationArea;
      normalizeWeightsGpu<<<CAFFE_GET_BLOCKS(countReduced), CAFFE_CUDA_NUM_THREADS>>>(
        weight_binary_->mutable_gpu_data(), this_blobs_0->gpu_data(), countReduced, binarizationArea);
    }
    // Option c - Weight = +-n per c,w,h
    else if (binaryOption > 1)
    {
      const auto binarizationArea = this_blobs_0->count(1);
      const auto countReduced = this_blobs_0->shape(0);
      normalizeWeightsGpu<<<CAFFE_GET_BLOCKS(countReduced), CAFFE_CUDA_NUM_THREADS>>>(
        weight_binary_->mutable_gpu_data(), this_blobs_0->gpu_data(), countReduced, binarizationArea);
    }
    // Full binary (weights + input)
    // else if (binaryOption == 3)
    if (binaryOption == 3)
    {
      // Get binary input (bottom_binary_)
      const auto count = bottom_binary_->count();
      getBinaryData<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        bottom_binary_->mutable_gpu_data(), bottom[0]->gpu_data(), count);
      // Matrix K (matrix_K_) dependent
      const auto matrixKNArea = matrix_K_->count(1);
      // Input (bottom) dependent
      const auto bottomChannels = bottom_binary_->shape(1);
      const auto bottomWHArea = bottom_binary_->count(2);
      const auto bottomNArea = bottomChannels * bottomWHArea;
      const auto matrixANArea = matrix_A_->count(1);
      // // Output (top) dependent
      const auto* const weightOneOverCHW = matrix_one_over_chw->gpu_data();
      const auto* const matrix_A_data = matrix_A_->gpu_data();
      auto* matrix_K_data = matrix_K_->mutable_gpu_data();
      for (int n = 0; n < num; ++n)
      {
        // Get A matrix (matrix_A_)
        addOverChannels<<<CAFFE_GET_BLOCKS(bottomWHArea), CAFFE_CUDA_NUM_THREADS>>>(
          matrix_A_->mutable_gpu_data() + n*matrixANArea, bottom[0]->gpu_data() + n*bottomNArea, bottomChannels,
          bottomWHArea);
        // Get K matrix (matrix_K_)
        CUDNN_CHECK(cudnnConvolutionForward(matrix_K_handle_,
          cudnn::dataType<Dtype>::one,
          matrix_A_desc_, matrix_A_data,
          matrix_one_filter_desc_, weightOneOverCHW,
          matrix_AK_conv_descs_,
          matrix_AK_fwd_algo_, workspace[0], matrix_AK_workspace_fwd_sizes_,
          cudnn::dataType<Dtype>::zero,
          matrix_K_desc_, matrix_K_data));
      }
// for (auto i = 0 ; i < matrix_A_->count(); i++)
// {
// if (i % matrix_A_->count(1) == 0)
// std::cout << "\n";
// std::cout << matrix_A_->cpu_data()[i] << " ";
// }
// std::cout << std::endl;
// for (auto i = 0 ; i < matrix_K_->count(); i++)
// {
// if (i % matrix_K_->count(1) == 0)
// std::cout << "\n";
// std::cout << matrix_K_->cpu_data()[i] << " ";
// }
// std::cout << std::endl;
    }
  }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Binary added
  if (this->layer_param_.convolution_param().binary() > 0)
  {
    CHECK_EQ(this->group_, 1) << "Binary conv net not implemented for !=1 groups.";
    CHECK_EQ(bottom.size(), 1) << "Binary conv net not implemented for !=1 bottoms.";
    // TEST/TRAIN - First frame (initialization)
    if (!weight_initialized_)
    {
      weight_initialized_ = true;
      CHECK_GE(this->blobs_.size(), 1);
      CHECK_GT(this->blobs_[0]->shape().size(), 2u);
      weight_binary_.reset(new Blob<Dtype>());
      weight_binary_->Reshape(this->blobs_[0]->shape());
      if (this->layer_param_.convolution_param().binary() > 2)
      {
        // Blob initialization
        bottom_binary_.reset(new Blob<Dtype>());
        matrix_A_.reset(new Blob<Dtype>());
        matrix_K_.reset(new Blob<Dtype>());
        matrix_one_over_chw.reset(new Blob<Dtype>());
        // Blob reshape
        bottom_binary_->Reshape(bottom[0]->shape());
        matrix_A_->Reshape(bottom[0]->shape(0), 1, bottom[0]->shape(2), bottom[0]->shape(3));
        matrix_K_->Reshape(top[0]->shape(0), 1, top[0]->shape(2), top[0]->shape(3));
        matrix_one_over_chw->Reshape(bottom[0]->shape(0), 1, this->blobs_[0]->shape(2), this->blobs_[0]->shape(3));
        // Filling matrix_one_over_chw
        auto* inputOnes = matrix_one_over_chw->mutable_cpu_data();
        const auto bottomNArea = bottom_binary_->count(1);
        for (auto i = 0 ; i < matrix_one_over_chw->count() ; i++)
          inputOnes[i] = Dtype(1)/Dtype(bottomNArea);
      }
      // Data to weightReal
      binarizeWeightsAndInputGpu(weight_binary_.get(), bottom_binary_.get(), matrix_A_.get(), matrix_K_.get(),
                                 this->matrix_one_over_chw.get(), this->blobs_[0], bottom, top, this->num_,
                                 this->layer_param_.convolution_param().binary());
    }
    // TRAIN (every frame)
    if (this->phase_ == TRAIN)
      binarizeWeightsAndInputGpu(weight_binary_.get(), bottom_binary_.get(), matrix_A_.get(), matrix_K_.get(),
                                 this->matrix_one_over_chw.get(), this->blobs_[0], bottom, top, this->num_,
                                 this->layer_param_.convolution_param().binary());
// if (this->layer_param_.convolution_param().binary() == 2)
// {
// std::cout << "\n"
// << this->blobs_[0]->shape(0) << " " << this->blobs_[0]->shape(1) << " " << this->blobs_[0]->shape(2) << " " << this->blobs_[0]->shape(3) << "\t"
// << bottom[0]->shape(0) << " " << bottom[0]->shape(1) << " " << bottom[0]->shape(2) << " " << bottom[0]->shape(3) << "\t"
// << top[0]->shape(0) << " " << top[0]->shape(1) << " " << top[0]->shape(2) << " " << top[0]->shape(3) << std::endl;
// }
  }
  // Binary added end

  // const Dtype* weight = this->blobs_[0]->gpu_data(); // Binary commented
  // Binary added
  // const Dtype* weight = (this->layer_param_.convolution_param().binary() > 0 && this->phase_ == TRAIN
  //   ? weight_binary_->gpu_data() : this->blobs_[0]->gpu_data());
  const Dtype* weight = (this->layer_param_.convolution_param().binary() > 0
    ? weight_binary_->gpu_data() : this->blobs_[0]->gpu_data());
  // const Dtype* weight = (this->layer_param_.convolution_param().binary() > 0 && this->phase_ == TEST
  //   ? weight_binary_->gpu_data() : this->blobs_[0]->gpu_data());
  // Binary added ended
  for (int i = 0; i < bottom.size(); ++i) {
    // const Dtype* bottom_data = bottom[i]->gpu_data(); // Binary commented
    // Binary added
    const Dtype* bottom_data = (this->layer_param_.convolution_param().binary() > 2
      ? bottom_binary_->gpu_data() : bottom[i]->gpu_data());
    // Binary added ended
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

      // Binary added
      if (this->layer_param_.convolution_param().binary() > 2)
      {
        const auto topChannels = top[i]->shape(1);
        const auto topWHArea = top[i]->count(2);
// Dtype* top_data = top[i]->mutable_gpu_data();
        multiplyOverChannels<<<CAFFE_GET_BLOCKS(topWHArea), CAFFE_CUDA_NUM_THREADS>>>(
          top_data + top_offset_ * g, matrix_K_->gpu_data(), topChannels, topWHArea);
        // top_descs_[i] .*= matrix_K_;
// for (auto asdf = 0 ; asdf < top[i]->count(); asdf++)
// {
// if (asdf % top[i]->count(1) == 0)
// std::cout << "\n";
// std::cout << top[i]->cpu_data()[asdf] << " ";
// }
// std::cout << std::endl;
// for (auto asdf = 0 ; asdf < bottom_binary_->count(); asdf++)
// for (auto asdf = 0 ; asdf < 20; asdf++)
//   std::cout << bottom_binary_->cpu_data()[asdf] << " ";
// std::cout << "\n";
// for (auto asdf = 0 ; asdf < 20; asdf++)
//   std::cout << bottom[i]->cpu_data()[asdf] << " ";
// std::cout << "\n\n" << std::endl;
      }
      // Binary added end

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
    // weight = this->blobs_[0]->gpu_data();
    // Plain truncating
    weight = (this->layer_param_.convolution_param().binary() > 0
              ? weight_binary_->gpu_data() : this->blobs_[0]->gpu_data());
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
          weight = (this->layer_param_.convolution_param().binary() > 0
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
  if (this->layer_param_.convolution_param().binary() > 0) // Binary added
  {
    if (this->param_propagate_down_[0])
    {
      const auto count = this->blobs_[0]->count();
      // // Option a - Weight = +-1
      // // Do nothing
      // Option b - Weight = +-n
      const auto binarizationArea = this->blobs_[0]->shape(2) * this->blobs_[0]->shape(3);
      const auto countReduced = count/binarizationArea;
      backwardNormalizeWeightsGpu<<<CAFFE_GET_BLOCKS(countReduced), CAFFE_CUDA_NUM_THREADS>>>(
        this->blobs_[0]->mutable_gpu_diff(), this->blobs_[0]->gpu_diff(), this->blobs_[0]->gpu_data(),
        countReduced, binarizationArea);
    }
  }
  // // If binary (XNOR-style) - First tried (didn't work)
  // if (this->layer_param_.convolution_param().binary() > 0) // Binary added
  // {
  //   if (this->param_propagate_down_[0]) {
  //     // Channel area = volume from axis 2 to final (num, channel, h, w)
  //     const auto* weight_real = this->blobs_[0]->cpu_data();
  //     auto* weight_real_diff = this->blobs_[0]->mutable_cpu_diff();
  //     const auto channelArea = weight_binary_->count(1);
  //     const auto imageArea = weight_binary_->count(2);
  //     const auto oneOverN = 1/Dtype(imageArea);
  //     for (auto num = 0 ; num < weight_binary_->shape(0) ; num++)
  //     {
  //       const auto offsetNum = num*channelArea;
  //       for (auto channel = 0 ; channel < weight_binary_->shape(1) ; channel++)
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
  // // My binary way (guiding weights to 1)
  // if (this->layer_param_.convolution_param().binary() > 0) // Binary added
  // {
  //   if (this->param_propagate_down_[0]) {
  //     const auto lambda = 0.01f;
  //     const auto* const weight_real = this->blobs_[0]->cpu_data();
  //     auto* weight_real_diff = this->blobs_[0]->mutable_cpu_diff();
  //     for (auto index = 0 ; index < this->blobs_[0]->count() ; index++)
  //       weight_real_diff[index] += 2*lambda*(   weight_real[index] - (weight_real[index] < 0 ? -1 : 1)   );
  //   }
  // }
  // // Binary added end
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
