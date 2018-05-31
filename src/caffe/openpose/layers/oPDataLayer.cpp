#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"
// OpenPose: added
#include <chrono>
#include <stdexcept>
#include "caffe/util/io.hpp" // DecodeDatum, DecodeDatumNative
#include "caffe/openpose/getLine.hpp"
#include "caffe/openpose/layers/oPDataLayer.hpp"
// OpenPose: added end

#include <iostream>
using namespace std;

namespace caffe {

template <typename Dtype>
OPDataLayer<Dtype>::OPDataLayer(const LayerParameter& param) :
    BasePrefetchingDataLayer<Dtype>(param),
    offset_(),
    offsetSecond(), // OpenPose: added
    offsetThird(),
    op_transform_param_(param.op_transform_param()) // OpenPose: added
{
    db_.reset(db::GetDB(param.data_param().backend()));
    db_->Open(param.data_param().source(), db::READ);
    cursor_.reset(db_->NewCursor());
    // OpenPose: added
    mOnes = 0;
    mTwos = 0;
    mThrees = 0;
    // Set up secondary DB
    if (!param.op_transform_param().source_secondary().empty())
    {
        secondDb = true;
        secondProbability = param.op_transform_param().prob_secondary();
        CHECK_GE(secondProbability, 0.f);
        CHECK_LE(secondProbability, 1.f);
        dbSecond.reset(db::GetDB(DataParameter_DB::DataParameter_DB_LMDB));
        dbSecond->Open(param.op_transform_param().source_secondary(), db::READ);
        cursorSecond.reset(dbSecond->NewCursor());
    }
    else
    {
        secondDb = false;
        secondProbability = 0.f;
    }
    // Set up tertiary DB
    if (!param.op_transform_param().source_tertiary().empty())
    {
        thirdDb = true;
        thirdProbability = param.op_transform_param().prob_tertiary();
        CHECK_GE(thirdProbability, 0.f);
        CHECK_LE(thirdProbability, 1.f);
        dbThird.reset(db::GetDB(DataParameter_DB::DataParameter_DB_LMDB));
        dbThird->Open(param.op_transform_param().source_tertiary(), db::READ);
        cursorThird.reset(dbThird->NewCursor());
    }
    else
    {
        thirdDb = false;
        thirdProbability = 0.f;
    }
    // Set up negatives DB
    if (!param.op_transform_param().source_background().empty())
    {
        backgroundDb = true;
        dbBackground.reset(db::GetDB(DataParameter_DB::DataParameter_DB_LMDB));
        dbBackground->Open(param.op_transform_param().source_background(), db::READ);
        cursorBackground.reset(dbBackground->NewCursor());
    }
    else
        backgroundDb = false;
    // Timer
    mDuration = 0;
    mCounter = 0;
    // OpenPose: added end
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
    // Load extra strides amounts
    extra_labels_count_ = top.size() - 2;
    const std::string extra_strides_string = op_transform_param_.extra_strides();
    for(int i=0; i<extra_strides_string.size(); i++){
        extra_strides_.push_back(extra_strides_string[i] - '0');
        std::cout << extra_strides_.back() << std::endl;
    }
    if(extra_strides_.size() != extra_labels_count_) throw std::runtime_error("Invalid extra_strides");

    const int batch_size = this->layer_param_.data_param().batch_size();
    // Read a data point, and use it to initialize the top blob.
    Datum datum;
    datum.ParseFromString(cursor_->value());

    // OpenPose: added
    mOPDataTransformer.reset(new OPDataTransformer<Dtype>(op_transform_param_, this->phase_, op_transform_param_.model()));
    if (secondDb)
        mOPDataTransformerSecondary.reset(new OPDataTransformer<Dtype>(op_transform_param_, this->phase_, op_transform_param_.model_secondary()));
    if (thirdDb)
        mOPDataTransformerTertiary.reset(new OPDataTransformer<Dtype>(op_transform_param_, this->phase_, op_transform_param_.model_tertiary()));
    // mOPDataTransformer->InitRand();
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
        const int numberChannels = this->mOPDataTransformer->getNumberChannels();
        std::vector<int> labelShape{batch_size, numberChannels, height/stride, width/stride};
        top[1]->Reshape(labelShape);
        for (int i = 0; i < this->prefetch_.size(); ++i)
            this->prefetch_[i]->label_.Reshape(labelShape);

        // Generate extra label shapes
        for(int j=0; j<extra_labels_count_; j++){
            extra_labels_shapes_.push_back({batch_size, numberChannels, height/extra_strides_[j], width/extra_strides_[j]});
        }
        for(int j=0; j<extra_labels_count_; j++){
            top[j+2]->Reshape(extra_labels_shapes_[j]);
            std::cout << extra_labels_shapes_.back()[0] << " " << extra_labels_shapes_.back()[1] << " " << extra_labels_shapes_.back()[2] << " " << extra_labels_shapes_.back()[3] << std::endl;
        }
        for (int i = 0; i < this->prefetch_.size(); ++i)
            for (int j = 0; j < Batch<float>::extra_labels_count; ++j)
                this->prefetch_[i]->extra_labels_[j].Reshape(labelShape);
        for (int i = 0; i < this->prefetch_.size(); ++i)
            for (int j = 0; j < extra_labels_count_; ++j)
                this->prefetch_[i]->extra_labels_[j].Reshape(extra_labels_shapes_[j]);
        for (int j=0; j<extra_labels_count_; j++){
            extra_transformed_labels_[j].Reshape(1,extra_labels_shapes_[j][1],extra_labels_shapes_[j][2],extra_labels_shapes_[j][3]);
        }

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

// OpenPose: added
template <typename Dtype>
bool OPDataLayer<Dtype>::SkipSecond()
{
    int size = Caffe::solver_count();
    int rank = Caffe::solver_rank();
    bool keep = (offsetSecond % size) == rank ||
                  // In test mode, only rank 0 runs, so avoid skipping
                  this->layer_param_.phase() == TEST;
    return !keep;
}

// OpenPose: added
template <typename Dtype>
bool OPDataLayer<Dtype>::SkipThird()
{
    int size = Caffe::solver_count();
    int rank = Caffe::solver_rank();
    bool keep = (offsetThird % size) == rank ||
                  // In test mode, only rank 0 runs, so avoid skipping
                  this->layer_param_.phase() == TEST;
    return !keep;
}


template<typename Dtype>
void OPDataLayer<Dtype>::NextBackground()
{
    if (backgroundDb)
    {
        cursorBackground->Next();
        if (!cursorBackground->valid())
        {
            LOG_IF(INFO, Caffe::root_solver())
                    << "Restarting negatives data prefetching from start.";
            cursorBackground->SeekToFirst();
        }
    }
}

template<typename Dtype>
void OPDataLayer<Dtype>::NextSecond()
{
    cursorSecond->Next();
    if (!cursorSecond->valid())
    {
        LOG_IF(INFO, Caffe::root_solver())
                << "Restarting second data prefetching from start.";
        cursorSecond->SeekToFirst();
    }
    offsetSecond++;
}

template<typename Dtype>
void OPDataLayer<Dtype>::NextThird()
{
    cursorThird->Next();
    if (!cursorThird->valid())
    {
        LOG_IF(INFO, Caffe::root_solver())
                << "Restarting third data prefetching from start.";
        cursorThird->SeekToFirst();
    }
    offsetThird++;
}
// OpenPose: added ended

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
    for(int i=0; i<Batch<float>::extra_labels_count; i++)
        batch->extra_labels_[i].mutable_cpu_data();
    // OpenPose: added ended

    Datum datum;
    Datum datumBackground;

    // OpenPose: added ended
    std::string debugString = "";
    for (int item_id = 0; item_id < batch_size; ++item_id) {
        // OpenPose: added
        const float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
        bool desiredDbIs1 = true, desiredDbIs2 = false, desiredDbIs3 = false;
        if(!thirdDb){
            float firstProbability = (1-(secondProbability));
            if(dice <= firstProbability){
                desiredDbIs1 = true; desiredDbIs2 = false; desiredDbIs3 = false;
            }else{
                desiredDbIs1 = false; desiredDbIs2 = true; desiredDbIs3 = false;
            }
        }else{
            float firstProbability = (1-(secondProbability+thirdProbability));
            if(dice <= firstProbability){
                desiredDbIs1 = true; desiredDbIs2 = false; desiredDbIs3 = false;
            }else if(dice <= (firstProbability + secondProbability)){
                desiredDbIs1 = false; desiredDbIs2 = true; desiredDbIs3 = false;
            }else if(dice <= (firstProbability + secondProbability + thirdProbability)){
                desiredDbIs1 = false; desiredDbIs2 = false; desiredDbIs3 = true;
            }
        }

        // Debug
        //if(desiredDbIs1) debugString += "1";
        //if(desiredDbIs2) debugString += "2";
        //if(desiredDbIs3) debugString += "3";

        timer.Start();
        // OpenPose: commended
        // while (Skip()) {
        //     Next();
        // }
        // datum.ParseFromString(cursor_->value());
        // OpenPose: commended ended
        // OpenPose: added
        // If only main DB or if 2 DBs but 1st must go
        auto oPDataTransformerPtr = this->mOPDataTransformer;
        if (desiredDbIs1)
        {
            mOnes++;
            while (Skip())
                Next();
            datum.ParseFromString(cursor_->value());
        }
        // If 2 DBs & 2nd one must go
        else if (desiredDbIs2)
        {
            oPDataTransformerPtr = this->mOPDataTransformerSecondary;
            mTwos++;
            while (SkipSecond())
                NextSecond();
            datum.ParseFromString(cursorSecond->value());
        }
        // 3rd DB
        else if (desiredDbIs3)
        {
            oPDataTransformerPtr = this->mOPDataTransformerTertiary;
            mThrees++;
            while (SkipThird())
                NextThird();
            datum.ParseFromString(cursorThird->value());
        }
        if (backgroundDb)
        {
            NextBackground();
            datumBackground.ParseFromString(cursorBackground->value());
        }
        // OpenPose: added ended
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
        // Extra Labels
        if(extra_labels_count_){
            for(int j=0; j<extra_labels_count_; j++){
                extra_transformed_labels_[j].set_cpu_data(batch->extra_labels_[j].mutable_cpu_data() + batch->extra_labels_[j].offset(item_id));
            }
        }
        // Process image & label
        const auto begin = std::chrono::high_resolution_clock::now();
        if (backgroundDb){
            if(!extra_labels_count_)
                oPDataTransformerPtr->Transform(&(this->transformed_data_),
                                                &(this->transformed_label_),
                                                datum,
                                                &datumBackground);
            else
                oPDataTransformerPtr->Transform(&(this->transformed_data_),
                                                &(this->transformed_label_),
                                                datum,
                                                &datumBackground,
                                                extra_transformed_labels_,
                                                extra_strides_,
                                                extra_labels_count_);
        }else{
            if(extra_labels_count_) throw std::runtime_error("This case is not handled");
            oPDataTransformerPtr->Transform(&(this->transformed_data_),
                                            &(this->transformed_label_),
                                            datum);
        }
        const auto end = std::chrono::high_resolution_clock::now();
        mDuration += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();

        // DB 1
        if (desiredDbIs1)
            Next();
        // DB 2
        else if (desiredDbIs2)
            NextSecond();
        // DB 3
        else if (desiredDbIs3)
            NextThird();
        trans_time += timer.MicroSeconds();
        // OpenPose: added ended
        // OpenPose: commented
        // this->data_transformer_->Transform(datum, &(this->transformed_data_));
        // // Copy label.
        // if (this->output_labels_) {
        //   Dtype* topLabel = batch->label_.mutable_cpu_data();
        //   topLabel[item_id] = datum.label();
        // }
        // trans_time += timer.MicroSeconds();
        // Next();
        // OpenPose: commented ended
    }
    if(debugString.size()) std::cout << debugString << std::endl;
    // Timer (every 20 iterations x batch size)
    mCounter++;
    const auto repeatEveryXVisualizations = 2;
    if (mCounter == 20*repeatEveryXVisualizations)
    {
        std::cout << "Time: " << mDuration/repeatEveryXVisualizations * 1e-9 << "s\t"
                  << "Ratio: " << mOnes/float(mOnes+mTwos) << std::endl;
        mDuration = 0;
        mCounter = 0;
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
