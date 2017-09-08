#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"
// OpenPose: added
#include  <stdexcept>
#include "caffe/util/io.hpp" // DecodeDatum, DecodeDatumNative
#include "caffe/openpose/getLine.hpp"
#include "caffe/openpose/layers/oPDataLayer.hpp"
// OpenPose: added end

namespace caffe {

template <typename Dtype>
OPDataLayer<Dtype>::OPDataLayer(const LayerParameter& param) :
    BasePrefetchingDataLayer<Dtype>(param),
    offset_(),
    op_transform_param_(param.op_transform_param()) // OpenPose: added
{
    db_.reset(db::GetDB(param.data_param().backend()));
    db_->Open(param.data_param().source(), db::READ);
    cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
OPDataLayer<Dtype>::~OPDataLayer()
{
    this->StopInternalThread();
}

template <typename Dtype>
void OPDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    const int batch_size = this->layer_param_.data_param().batch_size();
    // Read a data point, and use it to initialize the top blob.
    Datum datum;
    datum.ParseFromString(cursor_->value());

    // OpenPose: added
    mOPDataTransformer.reset(new OPDataTransformer<Dtype>(op_transform_param_, this->phase_));
    mOPDataTransformer->InitRand();
    // Force color
    bool forceColor = this->layer_param_.data_param().force_encoded_color();
    if ((forceColor && DecodeDatum(&datum, true)) || DecodeDatumNative(&datum))
        LOG(INFO) << "Decoding Datum";
    // Image shape
    const int width = this->phase_ != TRAIN ? datum.width() : this->layer_param_.op_transform_param().crop_size_x();
    const int height = this->phase_ != TRAIN ? datum.height() : this->layer_param_.op_transform_param().crop_size_y();
    std::vector<int> topShape{batch_size, 3, height, width};
    top[0]->Reshape(topShape);
    this->transformed_data_.Reshape(1, topShape[1], topShape[2], topShape[3]);
    // Reshape top[0] and prefetch_data according to the batch_size.
    for (int i = 0; i < this->prefetch_.size(); ++i)
        this->prefetch_[i]->data_.Reshape(topShape);
    LOG(INFO) << "Image shape: " << topShape[0] << ", " << topShape[1] << ", " << topShape[2] << ", " << topShape[3];
    // Label
    if (this->output_labels_)
    {
        const int stride = this->layer_param_.op_transform_param().stride();
        const int numberParts = this->layer_param_.op_transform_param().num_parts();
        std::vector<int> labelShape{batch_size, 2*(numberParts+1), height/stride, width/stride};
        top[1]->Reshape(labelShape);
        for (int i = 0; i < this->prefetch_.size(); ++i)
            this->prefetch_[i]->label_.Reshape(labelShape);
        this->transformed_label_.Reshape(1, labelShape[1], labelShape[2], labelShape[3]);
        LOG(INFO) << "Label shape: " << labelShape[0] << ", " << labelShape[1] << ", " << labelShape[2] << ", " << labelShape[3];
    }
    else
        throw std::runtime_error{"output_labels_ must be set to true" + getLine(__LINE__, __FUNCTION__, __FILE__)};
    // OpenPose: end

    // OpenPose: commented
    // // Use data_transformer to infer the expected blob shape from datum.
    // vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
    // this->transformed_data_.Reshape(top_shape);
    // // Reshape top[0] and prefetch_data according to the batch_size.
    // top_shape[0] = batch_size;
    // top[0]->Reshape(top_shape);
    // for (int i = 0; i < this->prefetch_.size(); ++i) {
    //   this->prefetch_[i]->data_.Reshape(top_shape);
    // }
    // LOG_IF(INFO, Caffe::root_solver())
    //     << "output data size: " << top[0]->num() << ","
    //     << top[0]->channels() << "," << top[0]->height() << ","
    //     << top[0]->width();
    // // label
    // if (this->output_labels_) {
    //   vector<int> label_shape(1, batch_size);
    //   top[1]->Reshape(label_shape);
    //   for (int i = 0; i < this->prefetch_.size(); ++i) {
    //     this->prefetch_[i]->label_.Reshape(label_shape);
    //   }
    // }
    // OpenPose: end
}

template <typename Dtype>
bool OPDataLayer<Dtype>::Skip()
{
    int size = Caffe::solver_count();
    int rank = Caffe::solver_rank();
    bool keep = (offset_ % size) == rank ||
                  // In test mode, only rank 0 runs, so avoid skipping
                  this->layer_param_.phase() == TEST;
    return !keep;
}

template<typename Dtype>
void OPDataLayer<Dtype>::Next()
{
    cursor_->Next();
    if (!cursor_->valid())
    {
        LOG_IF(INFO, Caffe::root_solver())
                << "Restarting data prefetching from start.";
        cursor_->SeekToFirst();
    }
    offset_++;
}

// This function is called on prefetch thread
template<typename Dtype>
void OPDataLayer<Dtype>::load_batch(Batch<Dtype>* batch)
{
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());
    const int batch_size = this->layer_param_.data_param().batch_size();

    // OpenPose: added
    auto* topLabel = batch->label_.mutable_cpu_data();
    // OpenPose: added ended

    Datum datum;
    for (int item_id = 0; item_id < batch_size; ++item_id) {
        timer.Start();
        while (Skip()) {
            Next();
        }
        datum.ParseFromString(cursor_->value());
        read_time += timer.MicroSeconds();

        if (item_id == 0) {
            // OpenPose: added
            // this->transformed_data_.Reshape({1, 3, height, width});
            // top_shape[0] = batch_size;
            const int width = this->phase_ != TRAIN ? datum.width() : this->layer_param_.op_transform_param().crop_size_x();
            const int height = this->phase_ != TRAIN ? datum.height() : this->layer_param_.op_transform_param().crop_size_y();
            batch->data_.Reshape({batch_size, 3, height, width});
            // OpenPose: added ended
            // OpenPose: commented
            // // Reshape according to the first datum of each batch
            // // on single input batches allows for inputs of varying dimension.
            // // Use data_transformer to infer the expected blob shape from datum.
            // vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
            // this->transformed_data_.Reshape(top_shape);
            // // Reshape batch according to the batch_size.
            // top_shape[0] = batch_size;
            // batch->data_.Reshape(top_shape);
            // OpenPose: commented ended
        }

        // Apply data transformations (mirror, scale, crop...)
        timer.Start();
        // OpenPose: added
        // Image
        const int offset = batch->data_.offset(item_id);
        auto* topData = batch->data_.mutable_cpu_data();
        this->transformed_data_.set_cpu_data(topData + offset);
        // Label
        const int offsetLabel = batch->label_.offset(item_id);
        this->transformed_label_.set_cpu_data(topLabel + offsetLabel);
        // Process iamge & label
        this->mOPDataTransformer->Transform(datum, 
                                            &(this->transformed_data_),
                                            &(this->transformed_label_));
        // OpenPose: added ended
        // OpenPose: commented
        // this->data_transformer_->Transform(datum, &(this->transformed_data_));
        // // Copy label.
        // if (this->output_labels_) {
        //   Dtype* topLabel = batch->label_.mutable_cpu_data();
        //   topLabel[item_id] = datum.label();
        // }
        // OpenPose: commented ended
        trans_time += timer.MicroSeconds();
        Next();
    }
    timer.Stop();
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(OPDataLayer);
REGISTER_LAYER_CLASS(OPData);

}  // namespace caffe
