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
#include "caffe/openpose/layers/oPVideoLayer.hpp"
// OpenPose: added end

#include <iostream>
using namespace std;

namespace caffe {

template <typename Dtype>
OPVideoLayer<Dtype>::OPVideoLayer(const LayerParameter& param) :
    BasePrefetchingDataLayer<Dtype>(param),
    offset_(),
    offsetSecond(), // OpenPose: added
    op_transform_param_(param.op_transform_param()) // OpenPose: added
{
    db_.reset(db::GetDB(param.data_param().backend()));
    db_->Open(param.data_param().source(), db::READ);
    cursor_.reset(db_->NewCursor());
    // OpenPose: added
    mOnes = 0;
    mTwos = 0;
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
OPVideoLayer<Dtype>::~OPVideoLayer()
{
    this->StopInternalThread();
}

template <typename Dtype>
void OPVideoLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    frame_size = this->layer_param_.op_transform_param().frame_size();

    const int batch_size = this->layer_param_.data_param().batch_size();
    // Read a data point, and use it to initialize the top blob.
    Datum datum;
    datum.ParseFromString(cursor_->value());

    // OpenPose Module
    mOPDataTransformer.reset(new OPDataTransformer<Dtype>(op_transform_param_, this->phase_, op_transform_param_.model()));
    if (secondDb)
        mOPDataTransformerSecondary.reset(new OPDataTransformer<Dtype>(op_transform_param_, this->phase_, op_transform_param_.model_secondary()));

    // Multi Image shape (Data layer is ([frame*batch * 3 * 368 * 38])) - Set Data size
    const int width = this->phase_ != TRAIN ? datum.width() : this->layer_param_.op_transform_param().crop_size_x();
    const int height = this->phase_ != TRAIN ? datum.height() : this->layer_param_.op_transform_param().crop_size_y();
    std::vector<int> topShape{batch_size * frame_size, 3, height, width};
    top[0]->Reshape(topShape);

    // Set output and prefetch size
    this->transformed_data_.Reshape(topShape[0], topShape[1], topShape[2], topShape[3]);
    for (int i = 0; i < this->prefetch_.size(); ++i)
        this->prefetch_[i]->data_.Reshape(topShape);
    LOG(INFO) << "Video shape: " << topShape[0] << ", " << topShape[1] << ", " << topShape[2] << ", " << topShape[3];

    // Labels
    if (this->output_labels_)
    {
        const int stride = this->layer_param_.op_transform_param().stride();
        const int numberChannels = this->mOPDataTransformer->getNumberChannels();
        std::vector<int> labelShape{batch_size * frame_size, numberChannels, height/stride, width/stride};
        top[1]->Reshape(labelShape);
        for (int i = 0; i < this->prefetch_.size(); ++i)
            this->prefetch_[i]->label_.Reshape(labelShape);
        this->transformed_label_.Reshape(labelShape[0], labelShape[1], labelShape[2], labelShape[3]);
        LOG(INFO) << "Label shape: " << labelShape[0] << ", " << labelShape[1] << ", " << labelShape[2] << ", " << labelShape[3];
    }
    else
        throw std::runtime_error{"output_labels_ must be set to true" + getLine(__LINE__, __FUNCTION__, __FILE__)};

    cout << "\t\t****Data Layer successfully initialized****" << endl;
}

template <typename Dtype>
bool OPVideoLayer<Dtype>::Skip()
{
    int size = Caffe::solver_count();
    int rank = Caffe::solver_rank();
    bool keep = (offset_ % size) == rank ||
                  // In test mode, only rank 0 runs, so avoid skipping
                  this->layer_param_.phase() == TEST;
    return !keep;
}

template<typename Dtype>
void OPVideoLayer<Dtype>::Next()
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
bool OPVideoLayer<Dtype>::SkipSecond()
{
    int size = Caffe::solver_count();
    int rank = Caffe::solver_rank();
    bool keep = (offsetSecond % size) == rank ||
                  // In test mode, only rank 0 runs, so avoid skipping
                  this->layer_param_.phase() == TEST;
    return !keep;
}

template<typename Dtype>
void OPVideoLayer<Dtype>::NextBackground()
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
void OPVideoLayer<Dtype>::NextSecond()
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
// OpenPose: added ended

// This function is called on prefetch thread
template<typename Dtype>
void OPVideoLayer<Dtype>::load_batch(Batch<Dtype>* batch)
{
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());
    const int batch_size = this->layer_param_.data_param().batch_size();

    // Get Label pointer [Label shape: 20, 132, 46, 46]
    auto* topLabel = batch->label_.mutable_cpu_data();

    // Sample lmdb for video?
    Datum datum;
    Datum datumBackground;
    for (int item_id = 0; item_id < batch_size; ++item_id) {
        const float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
        const auto desiredDbIs1 = !secondDb || (dice <= (1-secondProbability));

        // Read from desired DB - DB1, DB2 or BG
        timer.Start();
        auto oPDataTransformerPtr = this->mOPDataTransformer;
        if (desiredDbIs1)
        {
            mOnes++;
            while (Skip())
                Next();
            datum.ParseFromString(cursor_->value());
        }
        else
        {
            oPDataTransformerPtr = this->mOPDataTransformerSecondary;
            mTwos++;
            while (SkipSecond())
                NextSecond();
            datum.ParseFromString(cursorSecond->value());
        }
        if (backgroundDb)
        {
            NextBackground();
            datumBackground.ParseFromString(cursorBackground->value());
        }
        read_time += timer.MicroSeconds();

        // First item
        if (item_id == 0) {
            const int width = this->phase_ != TRAIN ? datum.width() : this->layer_param_.op_transform_param().crop_size_x();
            const int height = this->phase_ != TRAIN ? datum.height() : this->layer_param_.op_transform_param().crop_size_y();
            batch->data_.Reshape({batch_size * frame_size, 3, height, width});
        }

        // Read in data
        timer.Start();
        VSeq vs;
        const int offset = batch->data_.offset(item_id);
        auto* topData = batch->data_.mutable_cpu_data();
        this->transformed_data_.set_cpu_data(topData);
        // Label
        const int offsetLabel = batch->label_.offset(item_id);
        this->transformed_label_.set_cpu_data(topLabel);
        // Process image & label
        const auto begin = std::chrono::high_resolution_clock::now();
        if (backgroundDb){
            if(desiredDbIs1)
                oPDataTransformerPtr->TransformVideoJSON(item_id, frame_size, vs, &(this->transformed_data_),
                                                &(this->transformed_label_),
                                                datum, &datumBackground);
            else
                std::cout << "2" << std::endl;

        }else{
            oPDataTransformerPtr->TransformVideoJSON(item_id, frame_size, vs, &(this->transformed_data_),
                                            &(this->transformed_label_),
                                            datum);
        }
        const auto end = std::chrono::high_resolution_clock::now();
        mDuration += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();

        // Advance to next data
        if (desiredDbIs1)
            Next();
        else
            NextSecond();
        trans_time += timer.MicroSeconds();
    }

    // Testing Optional
    auto oPDataTransformerPtr = this->mOPDataTransformer;
    oPDataTransformerPtr->Test(frame_size, &(this->transformed_data_), &(this->transformed_label_));
    //boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
    //std::cout << "Loaded Data" << std::endl;

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

INSTANTIATE_CLASS(OPVideoLayer);
REGISTER_LAYER_CLASS(OPVideo);

}  // namespace caffe
