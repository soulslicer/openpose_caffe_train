#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <atomic>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/openpose/layers/cpm_data_layer.hpp"

namespace caffe {

template <typename Dtype>
CPMDataLayer<Dtype>::CPMDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param),
    cpm_transform_param_(param.cpm_transform_param()){ // OpenPose: added
}

template <typename Dtype>
CPMDataLayer<Dtype>::~CPMDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void CPMDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());

  // OpenPose: commented
  // vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  // this->transformed_data_.Reshape(top_shape);
  // // Reshape top[0] and prefetch_data according to the batch_size.
  // top_shape[0] = batch_size;
  // top[0]->Reshape(top_shape);
  // for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
  //   this->prefetch_[i].data_.Reshape(top_shape);
  // }
  // OpenPose: added
  cpm_data_transformer_.reset(
     new CPMDataTransformer<Dtype>(cpm_transform_param_, this->phase_));
  cpm_data_transformer_->InitRand();

  LOG(INFO) << datum.height() << " " << datum.width() << " " << datum.channels();

  bool force_color = this->layer_param_.data_param().force_encoded_color();
  if ((force_color && DecodeDatum(&datum, true)) ||
      DecodeDatumNative(&datum)) {
    LOG(INFO) << "Decoding Datum";
  }

  // image
  const int crop_size = this->layer_param_.cpm_transform_param().crop_size();
  if (crop_size > 0) {
    // top[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
    // for (int i = 0; i < this->prefetch_.size(); ++i) {
    //   this->prefetch_[i]->data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
    // }
    // //this->transformed_data_.Reshape(1, 4, crop_size, crop_size);
    // this->transformed_data_.Reshape(1, 6, crop_size, crop_size);
  } 
  else {
    const int height = this->phase_ != TRAIN ? datum.height() :
      this->layer_param_.cpm_transform_param().crop_size_y();
    const int width = this->phase_ != TRAIN ? datum.width() :
      this->layer_param_.cpm_transform_param().crop_size_x();
    LOG(INFO) << "prefetch_.size() is " << this->prefetch_.size();
    top[0]->Reshape(batch_size, datum.channels(), height, width);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->data_.Reshape(batch_size, datum.channels(), height, width);
    }
    //this->transformed_data_.Reshape(1, 4, height, width);
    this->transformed_data_.Reshape(1, datum.channels(), height, width);
  }
  // OpenPose: added end
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // label
  if (this->output_labels_) {
    // OpenPose: commented
    // vector<int> label_shape(1, batch_size);
    // top[1]->Reshape(label_shape);
    // for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    //   this->prefetch_[i].label_.Reshape(label_shape);
    // }
    // OpenPose: added
    const int stride = this->layer_param_.cpm_transform_param().stride();
    const int height = this->phase_ != TRAIN ? datum.height() :
      this->layer_param_.cpm_transform_param().crop_size_y();
    const int width = this->phase_ != TRAIN ? datum.width() :
      this->layer_param_.cpm_transform_param().crop_size_x();

    int num_parts = this->layer_param_.cpm_transform_param().num_parts();
    top[1]->Reshape(batch_size, 2*(num_parts+1), height/stride, width/stride);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(batch_size, 2*(num_parts+1), height/stride, width/stride);
    }
    this->transformed_label_.Reshape(1, 2*(num_parts+1), height/stride, width/stride);
    // OpenPose: added end
  }
}

// This function is called on prefetch thread
std::atomic<int> sLoadBatchCounter{0}; // OpenPose: added
template<typename Dtype>
void CPMDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  // double read_time = 0; // OpenPose: commented
  double deque_time = 0; // OpenPose: added
  double decod_time = 0; // OpenPose: added
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(reader_.full().peek());
  // OpenPose: commented
  // // Use data_transformer to infer the expected blob shape from datum.
  // vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  // this->transformed_data_.Reshape(top_shape);
  // // Reshape batch according to the batch_size.
  // top_shape[0] = batch_size;
  // batch->data_.Reshape(top_shape);
  // OpenPose: added
  const int crop_size = this->layer_param_.cpm_transform_param().crop_size();
  bool force_color = this->layer_param_.data_param().force_encoded_color();
  if (batch_size == 1 && crop_size == 0) {
    if (datum.encoded()) {
      if (force_color) {
        DecodeDatum(&datum, true);
      } else {
        DecodeDatumNative(&datum);
      }
    }
    batch->data_.Reshape(1, datum.channels(),
        datum.height(), datum.width());
        this->transformed_data_.Reshape(1, datum.channels(),
        datum.height(), datum.width());
  }
  // OpenPose: end

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    // read_time += timer.MicroSeconds(); // OpenPose: commented
    deque_time += timer.MicroSeconds(); // OpenPose: added
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    // OpenPose: added
    cv::Mat cv_img;
    if (datum.encoded()) {
      if (force_color) {
        cv_img = DecodeDatumToCVMat(datum, true);
      } else {
        cv_img = DecodeDatumToCVMatNative(datum);
      }
      if (cv_img.channels() != this->transformed_data_.channels()) {
        LOG(WARNING) << "Your dataset contains encoded images with mixed "
        << "channel sizes. Consider adding a 'force_color' flag to the "
        << "model definition, or rebuild your dataset using "
        << "convert_imageset.";
      }
    }
    decod_time += timer.MicroSeconds();
    const int offset_label = batch->label_.offset(item_id);
    this->transformed_label_.set_cpu_data(top_label + offset_label);
    if (datum.encoded()) {
      this->cpm_data_transformer_->Transform(cv_img, &(this->transformed_data_));
    } else {
      this->cpm_data_transformer_->Transform_nv(datum, 
        &(this->transformed_data_),
        &(this->transformed_label_), sLoadBatchCounter);
      sLoadBatchCounter++;
    }
    // OpenPose: added end
    // OpenPose: commented
    // Copy label.
    // if (this->output_labels_) {
    //   top_label[item_id] = datum.label();
    // }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  // DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms."; // OpenPose: commented
  DLOG(INFO) << "  Dequeue time: " << deque_time / 1000 << " ms."; // OpenPose: added
  DLOG(INFO) << "   Decode time: " << decod_time / 1000 << " ms."; // OpenPose: added
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(CPMDataLayer);
REGISTER_LAYER_CLASS(CPMData);

}  // namespace caffe
