#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
// OpenPose: added
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
// OpenPose: added end
#endif  // USE_OPENCV

// OpenPose: added
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
// OpenPose: added end
#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
// OpenPose: added
#include "caffe/util/benchmark.hpp"
#include "caffe/openpose/cpm_data_transformer.hpp"
// OpenPose: added end

namespace caffe {

// OpenPose: added
const auto MODE = 5; //related to datum.channels(), TODO: make this a parameter in caffe.proto
// Remainder
// const std::map<unsigned int, std::string> POSE_COCO_BODY_PARTS {
//     {0,  "Nose"},
//     {1,  "Neck"},
//     {2,  "RShoulder"},
//     {3,  "RElbow"},
//     {4,  "RWrist"},
//     {5,  "LShoulder"},
//     {6,  "LElbow"},
//     {7,  "LWrist"},
//     {8,  "RHip"},
//     {9,  "RKnee"},
//     {10, "RAnkle"},
//     {11, "LHip"},
//     {12, "LKnee"},
//     {13, "LAnkle"},
//     {14, "REye"},
//     {15, "LEye"},
//     {16, "REar"},
//     {17, "LEar"},
//     {18, "RFoot1"},
//     {19, "LFoot1"},
//     {20, "RFoot2"},
//     {21, "LFoot2"},
//     {20, "Background"},
// };
const std::array<int, (int)Model::Size> NUMBER_PARTS{18, 15, 22};
const std::array<int, (int)Model::Size> NUMBER_PAFS{2*19, 2*14, 2*23};
const std::array<int, (int)Model::Size> NUMBER_HEAT_MAPS{NUMBER_PARTS[0]+NUMBER_PAFS[0],
                                                         NUMBER_PARTS[1]+NUMBER_PAFS[1],
                                                         NUMBER_PARTS[2]+NUMBER_PAFS[2]};
const std::array<std::vector<int>, (int)Model::Size> SWAP_LEFTS_SWAP{
    std::vector<int>{5,6,7,11,12,13,15,17},     std::vector<int>{5,6,7,11,12,13},   std::vector<int>{5,6,7,11,12,13,15,17, 19,21}
};
const std::array<std::vector<int>, (int)Model::Size> SWAP_RIGHTS_SWAP{
    std::vector<int>{2,3,4, 8,9,10,14,16},      std::vector<int>{2,3,4,8,9,10},     std::vector<int>{2,3,4, 8,9,10,14,16, 18,20}
};
const std::array<std::vector<int>, (int)Model::Size> TRANSFORM_MODEL_TO_OURS_A{
    std::vector<int>{0,5, 6,8,10, 5,7,9, 12,14,16, 11,13,15, 2,1,4,3},      std::vector<int>{9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 7},
    std::vector<int>{0,5, 6,8,10, 5,7,9, 12,14,16, 11,13,15, 2,1,4,3, 16,15,16,15}
};
const std::array<std::vector<int>, (int)Model::Size> TRANSFORM_MODEL_TO_OURS_B{
    std::vector<int>{0,6, 6,8,10, 5,7,9, 12,14,16, 11,13,15, 2,1,4,3},      std::vector<int>{9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 6},
    std::vector<int>{0,6, 6,8,10, 5,7,9, 12,14,16, 11,13,15, 2,1,4,3, 16,15,16,15}
};
const std::array<std::vector<int>, (int)Model::Size> LABEL_MAP_A{
    std::vector<int>{1, 8,  9, 1,  11, 12, 1, 2, 3, 2,  1, 5, 6, 5,  1, 0,  0,  14, 15},    std::vector<int>{0, 1, 2, 3, 1, 5, 6, 1, 14, 8, 9,  14, 11, 12},
    std::vector<int>{1, 8,  9, 1,  11, 12, 1, 2, 3, 2,  1, 5, 6, 5,  1, 0,  0,  14, 15, 10, 13, 10, 13}
};
const std::array<std::vector<int>, (int)Model::Size> LABEL_MAP_B{
    std::vector<int>{8, 9, 10, 11, 12, 13, 2, 3, 4, 16, 5, 6, 7, 17, 0, 14, 15, 16, 17},    std::vector<int>{1, 2, 3, 4, 5, 6, 7, 14, 8, 9, 10, 11, 12, 13},
    std::vector<int>{8, 9, 10, 11, 12, 13, 2, 3, 4, 16, 5, 6, 7, 17, 0, 14, 15, 16, 17, 18, 19, 20, 21}
};
std::string getLine(const int line, const std::string& function, const std::string& file)
{
    return std::string{" at " + std::to_string(line) + ", " + function + ", " + file};
}
// OpenPose: added end

template<typename Dtype>
CPMDataTransformer<Dtype>::CPMDataTransformer(const CPMTransformationParameter& param,
        Phase phase)
        : param_(param), phase_(phase) {
    // check if we want to use mean_file
    if (param_.has_mean_file()) {
        CHECK_EQ(param_.mean_value_size(), 0) <<
            "Cannot specify mean_file and mean_value at the same time";
        const string& mean_file = param.mean_file();
        if (Caffe::root_solver()) {
            LOG(INFO) << "Loading mean file from: " << mean_file;
        }
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
        data_mean_.FromProto(blob_proto);
    }
    // check if we want to use mean_value
    if (param_.mean_value_size() > 0) {
        CHECK(param_.has_mean_file() == false) <<
            "Cannot specify mean_file and mean_value at the same time";
        for (int c = 0; c < param_.mean_value_size(); ++c) {
            mean_values_.push_back(param_.mean_value(c));
        }
    }
    // OpenPose: added
    LOG(INFO) << "CPMDataTransformer constructor done.";
    mNumberPartsInLmdb = param_.np_in_lmdb();
    mNumberParts = param_.num_parts();
    mIsTableSet = false;
    // Model
    mModel = Model::Size;
    for (auto i = 0 ; i < (int)Model::Size ; i++)
    {
        if (mNumberParts == NUMBER_HEAT_MAPS[i])
        {
            mModel = (Model)i;
            break;
        }
    }
    if (mModel == Model::Size)
        throw std::runtime_error{"Invalid mModel at " + getLine(__LINE__, __FUNCTION__, __FILE__)};
    // OpenPose: added end
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::Transform(const Datum& datum,
                                          Dtype* transformedData) {
    const string& data = datum.data();
    const int datumChannels = datum.channels();
    const int datumHeight = datum.height();
    const int datumWidth = datum.width();

    const int crop_size = param_.crop_size();
    const Dtype scale = param_.scale();
    const bool do_mirror = param_.mirror() && Rand(2);
    const bool has_mean_file = param_.has_mean_file();
    const bool hasUint8 = data.size() > 0;
    const bool has_mean_values = mean_values_.size() > 0;

    CHECK_GT(datumChannels, 0);
    CHECK_GE(datumHeight, crop_size);
    CHECK_GE(datumWidth, crop_size);

    Dtype* mean = NULL;
    if (has_mean_file) {
        CHECK_EQ(datumChannels, data_mean_.channels());
        CHECK_EQ(datumHeight, data_mean_.height());
        CHECK_EQ(datumWidth, data_mean_.width());
        mean = data_mean_.mutable_cpu_data();
    }
    if (has_mean_values) {
        CHECK(mean_values_.size() == 1 || mean_values_.size() == datumChannels) <<
         "Specify either 1 mean_value or as many as channels: " << datumChannels;
        if (datumChannels > 1 && mean_values_.size() == 1) {
            // Replicate the mean_value for simplicity
            for (auto c = 1; c < datumChannels; ++c) {
                mean_values_.push_back(mean_values_[0]);
            }
        }
    }

    int height = datumHeight;
    int width = datumWidth;

    int h_off = 0;
    int w_off = 0;
    if (crop_size)
    {
        height = crop_size;
        width = crop_size;
        // We only do random crop when we do training.
        if (phase_ == TRAIN)
        {
            h_off = Rand(datumHeight - crop_size + 1);
            w_off = Rand(datumWidth - crop_size + 1);
        }
        else
        {
            h_off = (datumHeight - crop_size) / 2;
            w_off = (datumWidth - crop_size) / 2;
        }
    }

    Dtype datum_element;
    int top_index, data_index;
    for (auto c = 0; c < datumChannels; ++c)
    {
        for (auto h = 0; h < height; ++h)
        {
            for (auto w = 0; w < width; ++w)
            {
                data_index = (c * datumHeight + h_off + h) * datumWidth + w_off + w;
                if (do_mirror)
                    top_index = (c * height + h) * width + (width - 1 - w);
                else
                    top_index = (c * height + h) * width + w;
                if (hasUint8)
                    datum_element = static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
                else
                    datum_element = datum.float_data(data_index);
                if (has_mean_file)
                    transformedData[top_index] = (datum_element - mean[data_index]) * scale;
                else
                {
                    if (has_mean_values)
                        transformedData[top_index] = (datum_element - mean_values_[c]) * scale;
                    else
                        transformedData[top_index] = datum_element * scale;
                }
            }
        }
    }
}

#ifdef USE_OPENCV
template<typename Dtype>
void CPMDataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                          Blob<Dtype>* transformed_blob) {
    const int crop_size = param_.crop_size();
    const int img_channels = cv_img.channels();
    const int img_height = cv_img.rows;
    const int img_width = cv_img.cols;

    // Check dimensions.
    const int channels = transformed_blob->channels();
    const int height = transformed_blob->height();
    const int width = transformed_blob->width();
    const int num = transformed_blob->num();

    CHECK_EQ(channels, img_channels);
    CHECK_LE(height, img_height);
    CHECK_LE(width, img_width);
    CHECK_GE(num, 1);

    CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

    const Dtype scale = param_.scale();
    const bool do_mirror = param_.mirror() && Rand(2);
    const bool has_mean_file = param_.has_mean_file();
    const bool has_mean_values = mean_values_.size() > 0;

    CHECK_GT(img_channels, 0);
    CHECK_GE(img_height, crop_size);
    CHECK_GE(img_width, crop_size);

    Dtype* mean = NULL;
    if (has_mean_file) {
        CHECK_EQ(img_channels, data_mean_.channels());
        CHECK_EQ(img_height, data_mean_.height());
        CHECK_EQ(img_width, data_mean_.width());
        mean = data_mean_.mutable_cpu_data();
    }
    if (has_mean_values) {
        CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
         "Specify either 1 mean_value or as many as channels: " << img_channels;
        if (img_channels > 1 && mean_values_.size() == 1) {
            // Replicate the mean_value for simplicity
            for (auto c = 1; c < img_channels; ++c) {
                mean_values_.push_back(mean_values_[0]);
            }
        }
    }

    int h_off = 0;
    int w_off = 0;
    cv::Mat cv_cropped_img = cv_img;
    if (crop_size)
    {
        CHECK_EQ(crop_size, height);
        CHECK_EQ(crop_size, width);
        // We only do random crop when we do training.
        if (phase_ == TRAIN)
        {
            h_off = Rand(img_height - crop_size + 1);
            w_off = Rand(img_width - crop_size + 1);
        }
        else
        {
            h_off = (img_height - crop_size) / 2;
            w_off = (img_width - crop_size) / 2;
        }
        cv::Rect roi(w_off, h_off, crop_size, crop_size);
        cv_cropped_img = cv_img(roi);
    }
    else
    {
        CHECK_EQ(img_height, height);
        CHECK_EQ(img_width, width);
    }

    CHECK(cv_cropped_img.data);

    Dtype* transformedData = transformed_blob->mutable_cpu_data();
    int top_index;
    for (auto h = 0; h < height; ++h)
    {
        const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
        int img_index = 0;
        for (auto w = 0; w < width; ++w)
        {
            for (auto c = 0; c < img_channels; ++c)
            {
                if (do_mirror)
                    top_index = (c * height + h) * width + (width - 1 - w);
                else
                    top_index = (c * height + h) * width + w;
                // int top_index = (c * height + h) * width + w;
                Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
                if (has_mean_file)
                {
                    int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
                    transformedData[top_index] =
                        (pixel - mean[mean_index]) * scale;
                }
                else
                {
                    if (has_mean_values)
                        transformedData[top_index] = (pixel - mean_values_[c]) * scale;
                    else
                        transformedData[top_index] = pixel * scale;
                }
            }
        }
    }
}
#endif  // USE_OPENCV

template <typename Dtype>
void CPMDataTransformer<Dtype>::InitRand() {
    const bool needs_rand = param_.mirror() ||
            (phase_ == TRAIN && param_.crop_size());
    if (needs_rand)
    {
        const unsigned int rng_seed = caffe_rng_rand();
        rng_.reset(new Caffe::RNG(rng_seed));
    }
    else
        rng_.reset();
}

template <typename Dtype>
int CPMDataTransformer<Dtype>::Rand(int n) {
    CHECK(rng_);
    CHECK_GT(n, 0);
    caffe::rng_t* rng =
            static_cast<caffe::rng_t*>(rng_->generator());
    return ((*rng)() % n);
}

// OpenPose: added
template<typename Dtype>
void CPMDataTransformer<Dtype>::Transform_nv(const Datum& datum, Blob<Dtype>* transformedData, Blob<Dtype>* transformedLabel,
                                             const int counter)
{
    // const auto reps = 50;
    // const auto begin = std::chrono::high_resolution_clock::now();
    // for (auto rep = 0 ; rep < reps ; rep++)
    // {
    //std::cout << "Function 2 is used"; std::cout.flush();
    const int datumChannels = datum.channels();
    //const int datumHeight = datum.height();
    //const int datumWidth = datum.width();

    const int im_channels = transformedData->channels();
    //const int im_height = transformedData->height();
    //const int im_width = transformedData->width();
    const int im_num = transformedData->num();

    //const int lb_channels = transformedLabel->channels();
    //const int lb_height = transformedLabel->height();
    //const int lb_width = transformedLabel->width();
    const int lb_num = transformedLabel->num();

    //LOG(INFO) << "image shape: " << transformedData->num() << " " << transformedData->channels() << " " 
    //                             << transformedData->height() << " " << transformedData->width();
    //LOG(INFO) << "label shape: " << transformedLabel->num() << " " << transformedLabel->channels() << " " 
    //                             << transformedLabel->height() << " " << transformedLabel->width();

    CHECK_EQ(datumChannels, 6);
    CHECK_EQ(im_channels, 6);

    ///CHECK_EQ(im_channels, 4);
    //CHECK_EQ(datumChannels, 4);

    CHECK_EQ(im_num, lb_num);
    //CHECK_LE(im_height, datumHeight);
    //CHECK_LE(im_width, datumWidth);
    CHECK_GE(im_num, 1);

    //const int crop_size = param_.crop_size();

    // if (crop_size)
    // {
    //     CHECK_EQ(crop_size, im_height);
    //     CHECK_EQ(crop_size, im_width);
    // }
    // else
    // {
    //     CHECK_EQ(datumHeight, im_height);
    //     CHECK_EQ(datumWidth, im_width);
    // }

    Dtype* transformedDataPtr = transformedData->mutable_cpu_data();
    Dtype* transformedLabelPtr = transformedLabel->mutable_cpu_data();
    CPUTimer timer;
    timer.Start();
    Transform_nv(transformedDataPtr, transformedLabelPtr, datum, counter); //call function 1
    VLOG(2) << "Transform_nv: " << timer.MicroSeconds() / 1000.0  << " ms";
    // }
    // const auto end = std::chrono::high_resolution_clock::now();
    // std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6 / reps << "ms" << std::endl;
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::generateLabelMap(Dtype* transformedLabel, const cv::Mat& image,
                                                 const MetaData& metaData) const
{
    const auto rezX = (int)image.cols;
    const auto rezY = (int)image.rows;
    const auto stride = (int)param_.stride();
    const auto gridX = rezX / stride;
    const auto gridY = rezY / stride;
    const auto channelOffset = gridY * gridX;

    for (auto gY = 0; gY < gridY; gY++)
        for (auto gX = 0; gX < gridX; gX++)
            for (auto part = mNumberParts+1; part < 2*(mNumberParts+1); part++)
                if (!(MODE == 6 && part == (2*mNumberParts + 1)))
                    transformedLabel[part*channelOffset + gY*gridX + gX] = 0;

    // Parameters
    const auto numberParts = NUMBER_PARTS[(int)mModel];
    const auto offsetInit = NUMBER_PAFS[(int)mModel]+1;
    const auto offsetEnd = NUMBER_HEAT_MAPS[(int)mModel]+1;
    const auto& labelMapA = LABEL_MAP_A[(int)mModel];
    const auto& labelMapB = LABEL_MAP_B[(int)mModel];

    // Common operations
    for (auto part = 0; part < numberParts; part++)
    {
        if (metaData.jointsSelf.isVisible[part] <= 1)
        {
            const auto& centerPoint = metaData.jointsSelf.points[part];
            putGaussianMaps(transformedLabel + (part+mNumberParts+offsetInit)*channelOffset, centerPoint, param_.stride(), 
                            gridX, gridY, param_.sigma()); //self
        }
        //for every other person
        for (auto otherPerson = 0; otherPerson < metaData.numberOtherPeople; otherPerson++)
        {
            if (metaData.jointsOthers[otherPerson].isVisible[part] <= 1)
            {
                const auto& centerPoint = metaData.jointsOthers[otherPerson].points[part];
                putGaussianMaps(transformedLabel + (part+mNumberParts+offsetInit)*channelOffset, centerPoint, param_.stride(), 
                                gridX, gridY, param_.sigma());
            }
        }
    }

    const auto threshold = 1;

    for (auto i = 0 ; i < labelMapA.size() ; i++)
    {
        cv::Mat count = cv::Mat::zeros(gridY, gridX, CV_8UC1);
        const auto& joints = metaData.jointsSelf;
        if (joints.isVisible[labelMapA[i]]<=1 && joints.isVisible[labelMapB[i]]<=1)
        {
            // putVecMaps
            putVecMaps(transformedLabel + (mNumberParts+ 1+ 2*i)*channelOffset,
                       transformedLabel + (mNumberParts+ 2+ 2*i)*channelOffset,
                       count, joints.points[labelMapA[i]], joints.points[labelMapB[i]],
                       param_.stride(), gridX, gridY, param_.sigma(), threshold); //self
        }

        // For every other person
        for (auto otherPerson = 0; otherPerson < metaData.numberOtherPeople; otherPerson++)
        {
            const auto& jointsOthers = metaData.jointsOthers[otherPerson];
            if (jointsOthers.isVisible[labelMapA[i]]<=1 && jointsOthers.isVisible[labelMapB[i]]<=1)
            {
                //putVecMaps
                putVecMaps(transformedLabel + (mNumberParts+ 1+ 2*i)*channelOffset,
                           transformedLabel + (mNumberParts+ 2+ 2*i)*channelOffset,
                           count, jointsOthers.points[labelMapA[i]], jointsOthers.points[labelMapB[i]],
                           param_.stride(), gridX, gridY, param_.sigma(), threshold); //self
            }
        }
    }

    // Put background channel
    for (auto gY = 0; gY < gridY; gY++)
    {
        for (auto gX = 0; gX < gridX; gX++)
        {
            double maximum = 0;
            //second background channel
            for (auto part = mNumberParts+offsetInit ; part < mNumberParts+offsetEnd ; part++)
            {
                maximum = (maximum > transformedLabel[part*channelOffset + gY*gridX + gX]) ? maximum
                            : transformedLabel[part*channelOffset + gY*gridX + gX];
            }
            transformedLabel[(2*mNumberParts+1)*channelOffset + gY*gridX + gX] = std::max(1.0-maximum, 0.0);
        }
    }

    // Visualize
    if (param_.visualize())
    {
        cv::Mat labelMap;
        for (auto part = 0; part < 2*(mNumberParts+1); part++)
        {      
            labelMap = cv::Mat::zeros(gridY, gridX, CV_8UC1);
            // int MPI_index = MPI_to_ours[part];
            // cv::Point2f center = metaData.jointsSelf.points[MPI_index];
            for (auto gY = 0; gY < gridY; gY++)
            {
                //printf("\n");
                for (auto gX = 0; gX < gridX; gX++)
                {
                    labelMap.at<uchar>(gY,gX) = (int)(transformedLabel[part*channelOffset + gY*gridX + gX]*255);
                    //printf("%f ", transformed_label_entry[gY*gridX + gX]*255);
                }
            }
            cv::resize(labelMap, labelMap, cv::Size{}, stride, stride, cv::INTER_LINEAR);
            cv::applyColorMap(labelMap, labelMap, cv::COLORMAP_JET);
            cv::addWeighted(labelMap, 0.5, image, 0.5, 0.0, labelMap);
            
            //center = center * (1.0/(float)param_.stride());
            //cv::circle(labelMap, center, 3, cv::Scalar{255,0,255}, -1);
            char imagename [100];
            sprintf(imagename, "augment_%04d_label_part_%02d.jpg", metaData.writeNumber, part);
            //LOG(INFO) << "filename is " << imagename;
            cv::imwrite(imagename, labelMap);
        }
    }
}

void setLabel(cv::Mat& image, const std::string label, const cv::Point& org)
{
    const int fontface = cv::FONT_HERSHEY_SIMPLEX;
    const double scale = 0.5;
    const int thickness = 1;
    int baseline = 0;
    const cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::rectangle(image, org + cv::Point{0, baseline}, org + cv::Point{text.width, -text.height},
                  cv::Scalar{0,0,0}, CV_FILLED);
    cv::putText(image, label, org, fontface, scale, cv::Scalar{255,255,255}, thickness, 20);
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::visualize(const cv::Mat& image, const MetaData& metaData, const AugmentSelection& augmentSelection) const
{
    cv::Mat imageToVisualize = image.clone();
    static int counter = 0;

    cv::rectangle(imageToVisualize, metaData.objpos-cv::Point2f{3.f,3.f}, metaData.objpos+cv::Point2f{3.f,3.f},
                  cv::Scalar{255,255,0}, CV_FILLED);
    for (auto part = 0 ; part < mNumberParts ; part++)
    {
        const auto currentPoint = metaData.jointsSelf.points[part];
        //LOG(INFO) << "drawing part " << part << ": ";
        //LOG(INFO) << metaData.jointsSelf.points.size();
        //LOG(INFO) << currentPoint;
        //if (metaData.jointsSelf.isVisible[part])
        // hand case
        if (mNumberParts == 21)
        {
            if (part < 4)
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{0,0,255}, -1);
            else if (part < 6 || part == 12 || part == 13)
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,0,0}, -1);
            else if (part < 8 || part == 14 || part == 15)
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,255,0}, -1);
            else if (part < 10|| part == 16 || part == 17)
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,100,0}, -1);
            else if (part < 12|| part == 18 || part == 19)
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,100,100}, -1);
            else 
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{0,100,100}, -1);
        }
        else if (mNumberParts == 9)
        {
            if (part==0 || part==1 || part==2 || part==6)
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{0,0,255}, -1);
            else if (part==3 || part==4 || part==5 || part==7)
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,0,0}, -1);
            else
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,255,0}, -1);
        }
        //body case
        else if (mNumberParts == 14 || mNumberParts == 28)
        {
            if (part < 14)
            {
                if (part==2 || part==3 || part==4 || part==8 || part==9 || part==10)
                    cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{0,0,255}, -1);
                else if (part==5 || part==6 || part==7 || part==11 || part==12 || part==13)
                    cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,0,0}, -1);
                else
                    cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,255,0}, -1);
            }
            else if (part < 16)
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{0,255,0}, -1);
            else
            {
                if (part==17 || part==18 || part==19 || part==23 || part==24)
                    cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,0,0}, -1);
                else if (part==20 || part==21 || part==22 || part==25 || part==26)
                    cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,100,100}, -1);
                else
                    cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,200,200}, -1);
            }
        }
        else
        {
            if (metaData.jointsSelf.isVisible[part] <= 1)
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{200,200,255}, -1);
        }
    }
    
    cv::line(imageToVisualize, metaData.objpos + cv::Point2f{-368/2.f,-368/2.f},
             metaData.objpos + cv::Point2f{368/2.f,-368/2.f}, cv::Scalar{0,255,0}, 2);
    cv::line(imageToVisualize, metaData.objpos + cv::Point2f{368/2.f,-368/2.f},
             metaData.objpos + cv::Point2f{368/2.f,368/2.f}, cv::Scalar{0,255,0}, 2);
    cv::line(imageToVisualize, metaData.objpos + cv::Point2f{368/2.f,368/2.f},
             metaData.objpos + cv::Point2f{-368/2.f,368/2.f}, cv::Scalar{0,255,0}, 2);
    cv::line(imageToVisualize, metaData.objpos + cv::Point2f{-368/2.f,368/2.f},
             metaData.objpos + cv::Point2f{-368/2.f,-368/2.f}, cv::Scalar{0,255,0}, 2);

    for (auto person=0;person<metaData.numberOtherPeople;person++)
    {
        cv::rectangle(imageToVisualize,
                      metaData.objPosOthers[person]-cv::Point2f{3.f,3.f},
                      metaData.objPosOthers[person]+cv::Point2f{3.f,3.f}, cv::Scalar{0,255,255}, CV_FILLED);
        for (auto part = 0 ; part < mNumberParts ; part++)
            if (metaData.jointsOthers[person].isVisible[part] <= 1)
                cv::circle(imageToVisualize, metaData.jointsOthers[person].points[part], 3, cv::Scalar{0,0,255}, -1);
    }
    
    // draw text
    char imagename [100];
    if (phase_ == TRAIN)
    {
        std::stringstream ss;
        // ss << "Augmenting with:" << (augmentSelection.flip ? "flip" : "no flip")
        //    << "; Rotate " << augmentSelection.degree << " deg; scaling: " << augmentSelection.scale << "; crop: " 
        //    << augmentSelection.crop.height << "," << augmentSelection.crop.width;
        ss << metaData.datasetString << " " << metaData.writeNumber << " index:" << metaData.annotationListIndex
           << "; person:" << metaData.peopleIndex << "; o_scale: " << metaData.scaleSelf;
        std::string stringInfo = ss.str();
        setLabel(imageToVisualize, stringInfo, cv::Point{0, 20});

        std::stringstream ss2;
        ss2 << "mult: " << augmentSelection.scale << "; rot: " << augmentSelection.degree << "; flip: "
            << (augmentSelection.flip?"true":"ori");
        stringInfo = ss2.str();
        setLabel(imageToVisualize, stringInfo, cv::Point{0, 40});

        cv::rectangle(imageToVisualize, cv::Point{0, (int)(imageToVisualize.rows)},
                      cv::Point{(int)(param_.crop_size_x()), (int)(param_.crop_size_y()+imageToVisualize.rows)},
                      cv::Scalar{255,255,255}, 1);

        sprintf(imagename, "augment_%04d_epoch_%03d_writenum_%03d.jpg", counter, metaData.epoch, metaData.writeNumber);
    }
    else
    {
        const std::string stringInfo = "no augmentation for testing";
        setLabel(imageToVisualize, stringInfo, cv::Point{0, 20});

        sprintf(imagename, "augment_%04d.jpg", counter);
    }
    //LOG(INFO) << "filename is " << imagename;
    // cv::imwrite(imagename, imageToVisualize);
    if (!imageToVisualize.empty())
    {
        cv::imshow(imagename, imageToVisualize);
        cv::waitKey(0);
    }
    else
        throw std::runtime_error{"Empty image at " + getLine(__LINE__, __FUNCTION__, __FILE__)};
    counter++;
}

template<typename Dtype>
bool CPMDataTransformer<Dtype>::augmentationFlip(cv::Mat& imageAugmented, cv::Mat& maskMiss, cv::Mat& maskAll, MetaData& metaData,
                                                 const cv::Mat& imageSource) const
{
    bool doflip;
    if (param_.aug_way() == "rand")
    {
        const auto dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        doflip = (dice <= param_.flip_prob());
    }
    else if (param_.aug_way() == "table")
        doflip = (mAugmentationFlips[metaData.writeNumber][metaData.epoch % param_.num_total_augs()] == 1);
    else
    {
        doflip = 0;
        LOG(INFO) << "Unhandled exception!!!!!!";
    }

    if (doflip)
    {
        flip(imageSource, imageAugmented, 1);
        const int w = imageSource.cols;
        if (!maskMiss.empty())
            flip(maskMiss, maskMiss, 1);
        if (!maskAll.empty())
            flip(maskAll, maskAll, 1);
        metaData.objpos.x = w - 1 - metaData.objpos.x;
        for (auto part = 0 ; part < mNumberParts ; part++)
            metaData.jointsSelf.points[part].x = w - 1 - metaData.jointsSelf.points[part].x;
        if (param_.transform_body_joint())
            swapLeftRight(metaData.jointsSelf);

        for (auto person = 0 ; person < metaData.numberOtherPeople ; person++)
        {
            metaData.objPosOthers[person].x = w - 1 - metaData.objPosOthers[person].x;
            for (auto part = 0 ; part < mNumberParts ; part++)
                metaData.jointsOthers[person].points[part].x = w - 1 - metaData.jointsOthers[person].points[part].x;
            if (param_.transform_body_joint())
                swapLeftRight(metaData.jointsOthers[person]);
        }
    }
    else if (imageAugmented.data != imageSource.data)
        imageAugmented = imageSource.clone();
    return doflip;
}

template<typename Dtype>
float CPMDataTransformer<Dtype>::augmentationRotate(cv::Mat& imageTarget, cv::Mat& maskMiss, cv::Mat& maskAll, MetaData& metaData,
                                                    const cv::Mat& imageSource) const
{
    float degree;
    if (param_.aug_way() == "rand")
    {
        const float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        degree = (dice - 0.5f) * 2 * param_.max_rotate_degree();
    }
    else if (param_.aug_way() == "table")
        degree = mAugmentationDegs[metaData.writeNumber][metaData.epoch % param_.num_total_augs()];
    else
    {
        degree = 0;
        LOG(INFO) << "Unhandled exception!!!!!!";
    }
    
    const cv::Point2f center(imageSource.cols/2.0, imageSource.rows/2.0);
    const cv::Rect bbox = cv::RotatedRect(center, imageSource.size(), degree).boundingRect();
    // adjust transformation matrix
    cv::Mat R = getRotationMatrix2D(center, degree, 1.0);
    R.at<double>(0,2) += bbox.width/2.0 - center.x;
    R.at<double>(1,2) += bbox.height/2.0 - center.y;
    //LOG(INFO) << "R=[" << R.at<double>(0,0) << " " << R.at<double>(0,1) << " " << R.at<double>(0,2) << ";" 
    //          << R.at<double>(1,0) << " " << R.at<double>(1,1) << " " << R.at<double>(1,2) << "]";
    warpAffine(imageSource, imageTarget, R, bbox.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar{128,128,128});
    if (!maskMiss.empty())
        //Scalar(0) for MPI, cv::Scalar{255} for COCO;
        warpAffine(maskMiss, maskMiss, R, bbox.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar{255});
    if (!maskAll.empty())
        warpAffine(maskAll, maskAll, R, bbox.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar{0});

    //adjust metaData data
    rotatePoint(metaData.objpos, R);
    for (auto part = 0 ; part < mNumberParts ; part++)
        rotatePoint(metaData.jointsSelf.points[part], R);
    for (auto person=0; person<metaData.numberOtherPeople; person++)
    {
        rotatePoint(metaData.objPosOthers[person], R);
        for (auto part = 0; part < mNumberParts ; part++)
            rotatePoint(metaData.jointsOthers[person].points[part], R);
    }
    return degree;
}

// include maskMiss
template<typename Dtype>
float CPMDataTransformer<Dtype>::augmentationScale(cv::Mat& imageTemp, cv::Mat& maskMiss, cv::Mat& maskAll, MetaData& metaData,
                                                   const cv::Mat& imageSource) const
{
    const float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
    float scaleMultiplier;
    // scale: linear shear into [scale_min, scale_max]
    // float scale = (param_.scale_max() - param_.scale_min()) * dice + param_.scale_min();
    if (dice > param_.scale_prob())
        scaleMultiplier = 1.f;
    else
    {
        const float dice2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
        // scaleMultiplier: linear shear into [scale_min, scale_max]
        scaleMultiplier = (param_.scale_max() - param_.scale_min()) * dice2 + param_.scale_min();
    }
    const float scaleAbs = param_.target_dist()/metaData.scaleSelf;
    const float scale = scaleAbs * scaleMultiplier;
    cv::resize(imageSource, imageTemp, cv::Size{}, scale, scale, cv::INTER_CUBIC);
    if (!maskMiss.empty())
        cv::resize(maskMiss, maskMiss, cv::Size{}, scale, scale, cv::INTER_CUBIC);
    if (!maskAll.empty())
        cv::resize(maskAll, maskAll, cv::Size{}, scale, scale, cv::INTER_CUBIC);

    //modify metaData data
    metaData.objpos *= scale;
    for (auto part = 0; part < mNumberParts ; part++)
        metaData.jointsSelf.points[part] *= scale;
    for (auto person=0; person<metaData.numberOtherPeople; person++)
    {
        metaData.objPosOthers[person] *= scale;
        for (auto part = 0; part < mNumberParts ; part++)
            metaData.jointsOthers[person].points[part] *= scale;
    }
    return scaleMultiplier;
}

template<typename Dtype>
cv::Size CPMDataTransformer<Dtype>::augmentationCropped(cv::Mat& imageTarget, cv::Mat& maskMissAugmented, cv::Mat& maskAllAugmented,
                                                        MetaData& metaData, const cv::Mat& imageSource, const cv::Mat& maskMiss,
                                                        const cv::Mat& maskAll) const
{
    const float diceX = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
    const float diceY = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
    const auto cropX = (int) param_.crop_size_x();
    const auto cropY = (int) param_.crop_size_y();

    const cv::Size pointOffset{int((diceX - 0.5f) * 2.f * param_.center_perterb_max()),
                               int((diceY - 0.5f) * 2.f * param_.center_perterb_max())};

    // LOG(INFO) << "Size of imageTemp is " << imageTemp.cols << " " << imageTemp.rows;
    // LOG(INFO) << "ROI: " << pointOffset.width << " " << pointOffset.height
    //           << " " << std::min(800, imageTemp.cols) << " " << std::min(256, imageTemp.rows);
    const cv::Point2i center{
        (int)(metaData.objpos.x + pointOffset.width),
        (int)(metaData.objpos.y + pointOffset.height),
    };
    
    imageTarget = 128*cv::Mat::ones(cropY, cropX, CV_8UC3);
    if (!maskMiss.empty())
        maskMissAugmented = cv::Mat(cropY, cropX, CV_8UC1); //for MPI, COCO with cv::Scalar{255};
    else
        maskMissAugmented = cv::Mat{};
    if (!maskAll.empty())
        maskAllAugmented = cv::Mat(cropY, cropX, CV_8UC1);
    else
        maskAllAugmented = cv::Mat{};
    for (auto y = 0 ; y < cropY ; y++)
    {
        //y,x on cropped
        for (auto x = 0 ; x < cropX ; x++)
        {
            const int coord_x_on_img = center.x - cropX/2 + x;
            const int coord_y_on_img = center.y - cropY/2 + y;
            if (onPlane(cv::Point{coord_x_on_img, coord_y_on_img}, cv::Size{imageSource.cols, imageSource.rows}))
            {
                imageTarget.at<cv::Vec3b>(y,x) = imageSource.at<cv::Vec3b>(coord_y_on_img, coord_x_on_img);
                if (!maskMissAugmented.empty())
                    maskMissAugmented.at<uchar>(y,x) = maskMiss.at<uchar>(coord_y_on_img, coord_x_on_img);
                if (!maskAllAugmented.empty())
                    maskAllAugmented.at<uchar>(y,x) = maskAll.at<uchar>(coord_y_on_img, coord_x_on_img);
            }
        }
    }

    //modify metaData data
    const int offsetLeft = -(center.x - (cropX/2));
    const int offsetUp = -(center.y - (cropY/2));
    const cv::Point2f offsetPoint{(float)offsetLeft, (float)offsetUp};
    metaData.objpos += offsetPoint;
    for (auto part = 0 ; part < mNumberParts ; part++)
        metaData.jointsSelf.points[part] += offsetPoint;
    for (auto person = 0 ; person < metaData.numberOtherPeople ; person++)
    {
        metaData.objPosOthers[person] += offsetPoint;
        for (auto part = 0 ; part < mNumberParts ; part++)
            metaData.jointsOthers[person].points[part] += offsetPoint;
    }

    return pointOffset;
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::rotatePoint(cv::Point2f& point2f, const cv::Mat& R) const
{
    cv::Mat cvMatPoint(3,1, CV_64FC1);
    cvMatPoint.at<double>(0,0) = point2f.x;
    cvMatPoint.at<double>(1,0) = point2f.y;
    cvMatPoint.at<double>(2,0) = 1;
    const cv::Mat new_point = R * cvMatPoint;
    point2f.x = new_point.at<double>(0,0);
    point2f.y = new_point.at<double>(1,0);
}

template<typename Dtype>
bool CPMDataTransformer<Dtype>::onPlane(const cv::Point& point, const cv::Size& imageSize) const
{
    return (point.x >= 0 && point.y >= 0 && point.x < imageSize.width && point.y < imageSize.height);
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::swapLeftRight(Joints& joints) const
{
    const auto& vectorLeft = SWAP_LEFTS_SWAP[(int)mModel];
    const auto& vectorRight = SWAP_RIGHTS_SWAP[(int)mModel];
    for (auto i = 0 ; i < vectorLeft.size() ; i++)
    {   
        const auto li = vectorLeft[i];
        const auto ri = vectorRight[i];
        std::swap(joints.points[ri], joints.points[li]);
        std::swap(joints.isVisible[ri], joints.isVisible[li]);
    }
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::setAugmentationTable(const int numData)
{
    mAugmentationDegs.resize(numData);     
    mAugmentationFlips.resize(numData);  
    for (auto data = 0; data < numData; data++)
    {
        mAugmentationDegs[data].resize(param_.num_total_augs());
        mAugmentationFlips[data].resize(param_.num_total_augs());
    }
    //load table files
    char filename[100];
    sprintf(filename, "../../rotate_%d_%d.txt", param_.num_total_augs(), numData);
    std::ifstream rot_file(filename);
    char filename2[100];
    sprintf(filename2, "../../flip_%d_%d.txt", param_.num_total_augs(), numData);
    std::ifstream flip_file(filename2);

    for (auto data = 0; data < numData; data++)
    {
        for (auto augmentation = 0; augmentation < param_.num_total_augs(); augmentation++)
        {
            rot_file >> mAugmentationDegs[data][augmentation];
            flip_file >> mAugmentationFlips[data][augmentation];
        }
    }
    // for (auto data = 0; data < numData; data++)
    // {
    //     for (auto augmentation = 0; augmentation < param_.num_total_augs(); augmentation++)
    //         printf("%d ", (int)mAugmentationDegs[data][augmentation]);
    //     printf("\n");
    // }
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::Transform_nv(Dtype* transformedData, Dtype* transformedLabel, const Datum& datum, const int counter)
{
    //TODO: some parameter should be set in prototxt
    const int claheTileSize = param_.clahe_tile_size();
    const int claheClipLimit = param_.clahe_clip_limit();

    const string& data = datum.data();
    const int datumChannels = datum.channels();
    const int datumHeight = datum.height();
    const int datumWidth = datum.width();

    //const int crop_size = param_.crop_size();
    const bool hasUint8 = data.size() > 0;
    const int cropX = param_.crop_size_x();
    const int cropY = param_.crop_size_y();

    CHECK_GT(datumChannels, 0);
    //CHECK_GE(datumHeight, crop_size);
    //CHECK_GE(datumWidth, crop_size);
    CPUTimer timer1;
    timer1.Start();
    //before any transformation, get the image from datum
    cv::Mat image(datumHeight, datumWidth, CV_8UC3);
    cv::Mat maskAll, maskMiss;
    if (MODE >= 5)
        maskMiss = cv::Mat(datumHeight, datumWidth, CV_8UC1);
    if (MODE == 6)
        maskAll = cv::Mat(datumHeight, datumWidth, CV_8UC1);

       int offset = image.rows * image.cols;
    int dindex;
    Dtype d_element;
    const bool has_uint8 = data.size() > 0;
    for (int i = 0; i < image.rows; ++i) {
     for (int j = 0; j < image.cols; ++j) {
      cv::Vec3b& rgb = image.at<cv::Vec3b>(i, j);
      for(int c = 0; c < 3; c++){
        dindex = c*offset + i*image.cols + j;
        if (has_uint8)
          d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
        else
          d_element = datum.float_data(dindex);
        rgb[c] = d_element;
      }

      if(MODE >= 5){
        dindex = 4*offset + i*image.cols + j;
        if (has_uint8)
          d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
        else
          d_element = datum.float_data(dindex);
        if (round(d_element/255)!=1 && round(d_element/255)!=0){
          std::cout << d_element << " " << round(d_element/255) << std::endl;
        }
        maskMiss.at<uchar>(i, j) = d_element; //round(d_element/255);
      }

      if(MODE == 6){
        dindex = 5*offset + i*image.cols + j;
        if (has_uint8)
          d_element = static_cast<Dtype>(static_cast<uint8_t>(data[dindex]));
        else
          d_element = datum.float_data(dindex);
        maskAll.at<uchar>(i, j) = d_element;
      }
    }
  }
    VLOG(2) << "  rgb[:] = datum: " << timer1.MicroSeconds()*1e-3 << " ms";
    timer1.Start();

    //color, contract
    if (param_.do_clahe())
        clahe(image, claheTileSize, claheClipLimit);
    if (param_.gray() == 1)
    {
        cv::cvtColor(image, image, CV_BGR2GRAY);
        cv::cvtColor(image, image, CV_GRAY2BGR);
    }
    VLOG(2) << "  color: " << timer1.MicroSeconds()*1e-3 << " ms";
    timer1.Start();

    MetaData metaData;
    readMetaData(metaData, data, 3 * imageArea, datumWidth);
    if (param_.transform_body_joint()) // we expect to transform body joints, and not to transform hand joints
        transformMetaJoints(metaData);

    VLOG(2) << "  ReadMeta+MetaJoints: " << timer1.MicroSeconds()*1e-3 << " ms";
    timer1.Start();
    AugmentSelection augmentSelection;
    // Visualize original
    if (param_.visualize())
    {
        visualize(image, metaData, augmentSelection);
        LOG(INFO) << metaData.jointsSelf.points.size();
        LOG(INFO) << metaData.jointsSelf.points[0];
    }
    //Start transforming
    cv::Mat imageAugmented = cv::Mat::zeros(cropY, cropX, CV_8UC3);
    cv::Mat maskMissAugmented;
    cv::Mat maskAllAugmented;
    cv::Mat imageTemp, imageTemp2; //size determined by scale
    VLOG(2) << "   input size (" << image.cols << ", " << image.rows << ")";
    const int stride = param_.stride();
    // We only do random transform augmentSelection augmentation when training.
    if (phase_ == TRAIN)
    {
        // Scale
        augmentSelection.scale = augmentationScale(imageTemp, maskMiss, maskAll, metaData, image);
        // if (param_.visualize())
        //     visualize(imageTemp, metaData, augmentSelection);
        // Rotation
        augmentSelection.degree = augmentationRotate(imageTemp2, maskMiss, maskAll, metaData, imageTemp);
        // if (param_.visualize())
        //     visualize(imageTemp2, metaData, augmentSelection);
        // Cropping
        augmentSelection.crop = augmentationCropped(imageAugmented, maskMissAugmented, maskAllAugmented, metaData, imageTemp2, maskMiss, maskAll);
        // if (param_.visualize())
        //     visualize(imageAugmented, metaData, augmentSelection);
        // Flipping
        augmentSelection.flip = augmentationFlip(imageAugmented, maskMissAugmented, maskAllAugmented, metaData, imageAugmented);

        // cv::imshow("imageAugmented", imageAugmented);
        // cv::Mat labelMap = maskMissAugmented;
        // cv::applyColorMap(labelMap, labelMap, cv::COLORMAP_JET);
        // cv::addWeighted(labelMap, 0.5, imageAugmented, 0.5, 0.0, labelMap);
        // cv::imshow("maskMissAugmented", labelMap);
        // cv::waitKey(0);

        if (!maskMissAugmented.empty())
            cv::resize(maskMissAugmented, maskMissAugmented, cv::Size{}, 1.0/stride, 1.0/stride, cv::INTER_CUBIC);
        if (!maskAllAugmented.empty())
            cv::resize(maskAllAugmented, maskAllAugmented, cv::Size{}, 1.0/stride, 1.0/stride, cv::INTER_CUBIC);
    }
    // Test
    else
        imageAugmented = image;
    // Visualize final
    if (param_.visualize())
    {
        visualize(imageAugmented, metaData, augmentSelection);
        LOG(INFO) << metaData.jointsSelf.points.size();
        LOG(INFO) << metaData.jointsSelf.points[0];
    }
    VLOG(2) << "  Aug: " << timer1.MicroSeconds()*1e-3 << " ms";
    timer1.Start();
    // LOG(INFO) << "scale: " << augmentSelection.scale << "; crop:(" << augmentSelection.crop.width
    //           << "," << augmentSelection.crop.height 
    //           << "); flip:" << augmentSelection.flip << "; degree: " << augmentSelection.degree;

    //copy transformed image (imageAugmented) into transformedData, do the mean-subtraction here
    const int imageAugmentedArea = imageAugmented.rows * imageAugmented.cols;
    const auto rezX = (int)imageAugmented.cols;
    const auto rezY = (int)imageAugmented.rows;
    const int gridX = rezX / stride;
    const int gridY = rezY / stride;
    const int channelOffset = gridY * gridX;

    for (auto y = 0; y < imageAugmented.rows ; y++)
    {
        const auto rowOffet = y*imageAugmented.cols;
        for (auto x = 0; x < imageAugmented.cols ; x++)
        {
            const auto totalOffet = rowOffet + x;
            const cv::Vec3b& rgb = imageAugmented.at<cv::Vec3b>(y, x);
            transformedData[totalOffet] = (rgb[0] - 128)/256.0;
            transformedData[totalOffet + imageAugmentedArea] = (rgb[1] - 128)/256.0;
            transformedData[totalOffet + 2*imageAugmentedArea] = (rgb[2] - 128)/256.0;
        }
    }
    
    // label size is image size/ stride
    if (MODE > 4)
    {
        for (auto gY = 0; gY < gridY; gY++)
        {
            for (auto gX = 0; gX < gridX; gX++)
            {
                for (auto part = 0; part < mNumberParts; part++)
                {
                    const float weight = float(maskMissAugmented.at<uchar>(gY, gX)) /255; //maskMissAugmented.at<uchar>(gY, gX);
                    if (metaData.jointsSelf.isVisible[part] != 3)
                        transformedLabel[part*channelOffset + gY*gridX + gX] = weight;
                }  
                // background channel
                if (MODE == 5)
                    transformedLabel[mNumberParts*channelOffset + gY*gridX + gX] = float(maskMissAugmented.at<uchar>(gY, gX)) /255;
                else
                {
                    transformedLabel[mNumberParts*channelOffset + gY*gridX + gX] = 1;
                    transformedLabel[(2*mNumberParts+1)*channelOffset + gY*gridX + gX] = float(maskAllAugmented.at<uchar>(gY, gX)) /255;
                }
            }
        }
    }  

    //putGaussianMaps(transformedData + 3*imageAugmentedArea, metaData.objpos, 1, imageAugmented.cols, imageAugmented.rows, param_.sigma_center());
    //LOG(INFO) << "image transformation done!";
    generateLabelMap(transformedLabel, imageAugmented, metaData);

    VLOG(2) << "  putGauss+genLabel: " << timer1.MicroSeconds()*1e-3 << " ms";
}

template<typename Dtype>
Dtype decodeFloats(const string& data, const size_t idx, const size_t len)
{
    Dtype pf;
    memcpy(&pf, const_cast<char*>(&data[idx]), len * sizeof(Dtype));
    return pf;
}

std::string decodeString(const string& data, const size_t idx)
{
    string result = "";
    auto counter = 0;
    while (data[idx+counter] != 0)
    {
        result.push_back(char(data[idx+counter]));
        counter++;
    }
    return result;
}

//very specific to genLMDB.py
template<typename Dtype>
void CPMDataTransformer<Dtype>::readMetaData(MetaData& metaData, const string& data, size_t offset3, size_t offset1)
{
    // ------------------- Dataset name ----------------------
    metaData.datasetString = decodeString(data, offset3);
    // ------------------- Image Dimension -------------------
    metaData.imageSize = cv::Size{(int)decodeFloats<Dtype>(data, offset3+offset1+4, 1),
                                  (int)decodeFloats<Dtype>(data, offset3+offset1, 1)};
    // ----------- Validation, nop, counters -----------------
    metaData.isValidation = (data[offset3+2*offset1]==0 ? false : true);
    metaData.numberOtherPeople = (int)data[offset3+2*offset1+1];
    metaData.peopleIndex = (int)data[offset3+2*offset1+2];
    const auto annotationListIndex = decodeFloats<Dtype>(data, offset3+2*offset1+3, 1);
    metaData.annotationListIndex = (int)annotationListIndex;
    const auto writeNumber = decodeFloats<Dtype>(data, offset3+2*offset1+7, 1);
    metaData.writeNumber = (int)writeNumber;
    const auto totalWriteNumber = decodeFloats<Dtype>(data, offset3+2*offset1+11, 1);
    metaData.totalWriteNumber = (int)totalWriteNumber;

    // count epochs according to counters
    static int currentEpoch = -1;
    if (metaData.writeNumber == 0)
        currentEpoch++;
    metaData.epoch = currentEpoch;
    if (metaData.writeNumber % 1000 == 0)
    {
        LOG(INFO) << "datasetString: " << metaData.datasetString <<"; imageSize: " << metaData.imageSize
                  << "; metaData.annotationListIndex: " << metaData.annotationListIndex
                  << "; metaData.writeNumber: " << metaData.writeNumber
                  << "; metaData.totalWriteNumber: " << metaData.totalWriteNumber
                  << "; metaData.epoch: " << metaData.epoch;
    }
    if (param_.aug_way() == "table" && !mIsTableSet)
    {
        setAugmentationTable(metaData.totalWriteNumber);
        mIsTableSet = true;
    }

    // ------------------- objpos -----------------------
    metaData.objpos.x = decodeFloats<Dtype>(data, offset3+3*offset1, 1);
    metaData.objpos.y = decodeFloats<Dtype>(data, offset3+3*offset1+4, 1);
    metaData.objpos -= cv::Point2f{1.f,1.f};
    // ------------ scaleSelf, jointsSelf --------------
    metaData.scaleSelf = decodeFloats<Dtype>(data, offset3+4*offset1, 1);
    auto& jointSelf = metaData.jointsSelf;
    jointSelf.points.resize(mNumberPartsInLmdb);
    jointSelf.isVisible.resize(mNumberPartsInLmdb);
    for (auto part = 0 ; part < mNumberPartsInLmdb; part++)
    {
        jointSelf.points[part].x = decodeFloats<Dtype>(data, offset3+5*offset1+4*part, 1);
        jointSelf.points[part].y = decodeFloats<Dtype>(data, offset3+6*offset1+4*part, 1);
        jointSelf.points[part] -= cv::Point2f{1.f,1.f}; //from matlab 1-index to c++ 0-index
        const auto isVisible = decodeFloats<Dtype>(data, offset3+7*offset1+4*part, 1);
        if (isVisible == 3)
            jointSelf.isVisible[part] = 3;
        else
        {
            jointSelf.isVisible[part] = (isVisible == 0) ? 0 : 1;
            if (jointSelf.points[part].x < 0 || jointSelf.points[part].y < 0 ||
                 jointSelf.points[part].x >= metaData.imageSize.width || jointSelf.points[part].y >= metaData.imageSize.height)
            {
                jointSelf.isVisible[part] = 2; // 2 means cropped, 0 means occluded by still on image
            }
        }
        //LOG(INFO) << jointSelf.points[part].x << " " << jointSelf.points[part].y << " " << jointSelf.isVisible[part];
    }
    
    //others (7 lines loaded)
    metaData.objPosOthers.resize(metaData.numberOtherPeople);
    metaData.scaleOthers.resize(metaData.numberOtherPeople);
    metaData.jointsOthers.resize(metaData.numberOtherPeople);
    for (auto person = 0 ; person < metaData.numberOtherPeople ; person++)
    {
        metaData.objPosOthers[person].x = decodeFloats<Dtype>(data, offset3+(8+person)*offset1, 1) - 1.f;
        metaData.objPosOthers[person].y = decodeFloats<Dtype>(data, offset3+(8+person)*offset1+4, 1) - 1.f;
        metaData.scaleOthers[person]  = decodeFloats<Dtype>(data, offset3+(8+metaData.numberOtherPeople)*offset1+4*person, 1);
    }
    //8 + numberOtherPeople lines loaded
    for (auto person = 0 ; person < metaData.numberOtherPeople ; person++)
    {
        auto& currentPerson = metaData.jointsOthers[person];
        currentPerson.points.resize(mNumberPartsInLmdb);
        currentPerson.isVisible.resize(mNumberPartsInLmdb);
        for (auto part = 0 ; part < mNumberPartsInLmdb; part++)
        {
            currentPerson.points[part].x = decodeFloats<Dtype>(data, offset3+(9+metaData.numberOtherPeople+3*person)*offset1+4*part, 1) - 1.f;
            currentPerson.points[part].y = decodeFloats<Dtype>(data, offset3+(9+metaData.numberOtherPeople+3*person+1)*offset1+4*part, 1) - 1.f;
            const auto isVisible = decodeFloats<Dtype>(data, offset3+(9+metaData.numberOtherPeople+3*person+2)*offset1+4*part, 1);
            currentPerson.isVisible[part] = (isVisible == 0 ? 0 : 1);
            if (currentPerson.points[part].x < 0 || currentPerson.points[part].y < 0 ||
                 currentPerson.points[part].x >= metaData.imageSize.width || currentPerson.points[part].y >= metaData.imageSize.height)
            {
                currentPerson.isVisible[part] = 2; // 2 means cropped, 1 means occluded by still on image
            }
        }
    }
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::transformMetaJoints(MetaData& metaData) const
{
    //transform joints in metaData from mNumberPartsInLmdb (specified in prototxt) to mNumberParts (specified in prototxt)
    transformJoints(metaData.jointsSelf);
    for (auto& joints : metaData.jointsOthers)
        transformJoints(joints);
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::transformJoints(Joints& joints) const
{
    //transform joints in metaData from mNumberPartsInLmdb (specified in prototxt) to mNumberParts (specified in prototxt)
    auto jointsOld = joints;

    // Common operations
    joints.points.resize(mNumberParts);
    joints.isVisible.resize(mNumberParts);

    // Parameters
    const auto& vectorA = TRANSFORM_MODEL_TO_OURS_A[(int)mModel];
    const auto& vectorB = TRANSFORM_MODEL_TO_OURS_B[(int)mModel];

    for (auto i = 0 ; i < vectorA.size() ; i++)
    {
        joints.points[i] = (jointsOld.points[vectorA[i]] + jointsOld.points[vectorB[i]]) * 0.5f;
        if (jointsOld.isVisible[vectorA[i]] == 2 || jointsOld.isVisible[vectorB[i]] == 2)
            joints.isVisible[i] = 2;
        else if (mModel == Model::COCO_18 && (jointsOld.isVisible[vectorA[i]] == 3 || jointsOld.isVisible[vectorB[i]] == 3))
            joints.isVisible[i] = 3;
        else
            joints.isVisible[i] = jointsOld.isVisible[vectorA[i]] && jointsOld.isVisible[vectorB[i]];
    }
}

template <typename Dtype>
void CPMDataTransformer<Dtype>::clahe(cv::Mat& bgrImage, const int tileSize, const int clipLimit) const
{
    cv::Mat labImage;
    cvtColor(bgrImage, labImage, CV_BGR2Lab);

    // Extract the L channel
    std::vector<cv::Mat> labPlanes(3);
    split(labImage, labPlanes);  // now we have the L image in labPlanes[0]

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = createCLAHE(clipLimit, cv::Size{tileSize, tileSize});
    //clahe->setClipLimit(4);
    cv::Mat dst;
    clahe->apply(labPlanes[0], dst);

    // Merge the the color planes back into an Lab image
    dst.copyTo(labPlanes[0]);
    merge(labPlanes, labImage);

    // convert back to RGB
    cv::Mat image_clahe;
    cvtColor(labImage, bgrImage, CV_Lab2BGR);
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::putGaussianMaps(Dtype* entry, const cv::Point2f& centerPoint, const int stride, const int gridX,
                                                const int gridY, const float sigma) const
{
    //LOG(INFO) << "putGaussianMaps here we start for " << centerPoint.x << " " << centerPoint.y;
    const float start = stride/2.f - 0.5f; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
    for (auto gY = 0; gY < gridY; gY++)
    {
        for (auto gX = 0; gX < gridX; gX++)
        {
            const float x = start + gX * stride;
            const float y = start + gY * stride;
            const float d2 = (x-centerPoint.x)*(x-centerPoint.x) + (y-centerPoint.y)*(y-centerPoint.y);
            const float exponent = d2 / 2.0 / sigma / sigma;
            //ln(100) = -ln(1%)
            if (exponent > 4.6052)
                continue;
            entry[gY*gridX + gX] += exp(-exponent);
            if (entry[gY*gridX + gX] > 1) 
                entry[gY*gridX + gX] = 1;
        }
    }
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::putVecMaps(Dtype* entryX, Dtype* entryY, cv::Mat& count, const cv::Point2f& centerA,
                                           const cv::Point2f& centerB, const int stride, const int gridX, const int gridY,
                                           const float sigma, const int threshold) const
{
    const auto centerAAux = 0.125f * centerA;
    const auto centerBAux = 0.125f * centerB;
    const int minX = std::max( int(round(std::min(centerAAux.x, centerBAux.x) - threshold)), 0);
    const int maxX = std::min( int(round(std::max(centerAAux.x, centerBAux.x) + threshold)), gridX);

    const int minY = std::max( int(round(std::min(centerAAux.y, centerBAux.y) - threshold)), 0);
    const int maxY = std::min( int(round(std::max(centerAAux.y, centerBAux.y) + threshold)), gridY);

    cv::Point2f bc = centerBAux - centerAAux;
    const float norm_bc = sqrt(bc.x*bc.x + bc.y*bc.y);
    bc.x = bc.x /norm_bc;
    bc.y = bc.y /norm_bc;

    // const float x_p = (centerAAux.x + centerBAux.x) / 2;
    // const float y_p = (centerAAux.y + centerBAux.y) / 2;
    // const float angle = atan2f(centerBAux.y - centerAAux.y, centerBAux.x - centerAAux.x);
    // const float sine = sinf(angle);
    // const float cosine = cosf(angle);
    // const float a_sqrt = (centerAAux.x - x_p) * (centerAAux.x - x_p) + (centerAAux.y - y_p) * (centerAAux.y - y_p);
    // const float b_sqrt = 10; //fixed

    for (auto gY = minY; gY < maxY; gY++)
    {
        for (auto gX = minX; gX < maxX; gX++)
        {
            const cv::Point2f ba{gX - centerAAux.x, gY - centerAAux.y};
            const float dist = std::abs(ba.x*bc.y - ba.y*bc.x);

            // const float A = cosine * (gX - x_p) + sine * (gY - y_p);
            // const float B = sine * (gX - x_p) - cosine * (gY - y_p);
            // const float judge = A * A / a_sqrt + B * B / b_sqrt;

            if (dist <= threshold)
            //if (judge <= 1)
            {
                const int counter = count.at<uchar>(gY, gX);
                //LOG(INFO) << "putVecMaps here we start for " << gX << " " << gY;
                if (counter == 0)
                {
                    entryX[gY*gridX + gX] = bc.x;
                    entryY[gY*gridX + gX] = bc.y;
                }
                else
                {
                    entryX[gY*gridX + gX] = (entryX[gY*gridX + gX]*counter + bc.x) / (counter + 1);
                    entryY[gY*gridX + gX] = (entryY[gY*gridX + gX]*counter + bc.y) / (counter + 1);
                    count.at<uchar>(gY, gX) = counter + 1;
                }
            }

        }
    }
}
// OpenPose: added end

INSTANTIATE_CLASS(CPMDataTransformer);

}  // namespace caffe
