#ifdef USE_OPENCV
    #include <opencv2/core/core.hpp>
    // OpenPose: added
    // #include <opencv2/contrib/contrib.hpp>
    // #include <opencv2/contrib/imgproc.hpp>
    // #include <opencv2/highgui/highgui.hpp>
    #include <opencv2/opencv.hpp>
    // OpenPose: added end
#endif  // USE_OPENCV

// OpenPose: added
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
// OpenPose: added end
#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
// OpenPose: added
#include "caffe/util/benchmark.hpp"
#include "caffe/openpose/getLine.hpp"
#include "caffe/openpose/oPDataTransformer.hpp"
// OpenPose: added end

namespace caffe {
// OpenPose: added ended
const auto sAntiStride = 2;
struct AugmentSelection
{
    bool flip = false;
    std::pair<cv::Mat, cv::Size> RotAndFinalSize;
    cv::Point2i cropCenter;
    float scale = 1.f;
};

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

void debugVisualize(const cv::Mat& image, const MetaData& metaData, const AugmentSelection& augmentSelection,
                    const PoseModel poseModel, const Phase& phase_, const OPTransformationParameter& param_)
{
    cv::Mat imageToVisualize = image.clone();

    cv::rectangle(imageToVisualize, metaData.objPos-cv::Point2f{3.f,3.f}, metaData.objPos+cv::Point2f{3.f,3.f},
                  cv::Scalar{255,255,0}, CV_FILLED);
    const auto numberBpPafChannels = getNumberBodyAndPafChannels(poseModel);
    for (auto part = 0 ; part < numberBpPafChannels ; part++)
    {
        const auto currentPoint = metaData.jointsSelf.points[part];
        // Hand case
        if (numberBpPafChannels == 21)
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
        else if (numberBpPafChannels == 9)
        {
            if (part==0 || part==1 || part==2 || part==6)
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{0,0,255}, -1);
            else if (part==3 || part==4 || part==5 || part==7)
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,0,0}, -1);
            else
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,255,0}, -1);
        }
        // Body case (CPM)
        else if (numberBpPafChannels == 14 || numberBpPafChannels == 28)
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

    cv::line(imageToVisualize, metaData.objPos + cv::Point2f{-368/2.f,-368/2.f},
             metaData.objPos + cv::Point2f{368/2.f,-368/2.f}, cv::Scalar{0,255,0}, 2);
    cv::line(imageToVisualize, metaData.objPos + cv::Point2f{368/2.f,-368/2.f},
             metaData.objPos + cv::Point2f{368/2.f,368/2.f}, cv::Scalar{0,255,0}, 2);
    cv::line(imageToVisualize, metaData.objPos + cv::Point2f{368/2.f,368/2.f},
             metaData.objPos + cv::Point2f{-368/2.f,368/2.f}, cv::Scalar{0,255,0}, 2);
    cv::line(imageToVisualize, metaData.objPos + cv::Point2f{-368/2.f,368/2.f},
             metaData.objPos + cv::Point2f{-368/2.f,-368/2.f}, cv::Scalar{0,255,0}, 2);

    for (auto person=0;person<metaData.numberOtherPeople;person++)
    {
        cv::rectangle(imageToVisualize,
                      metaData.objPosOthers[person]-cv::Point2f{3.f,3.f},
                      metaData.objPosOthers[person]+cv::Point2f{3.f,3.f}, cv::Scalar{0,255,255}, CV_FILLED);
        for (auto part = 0 ; part < numberBpPafChannels ; part++)
            if (metaData.jointsOthers[person].isVisible[part] <= 1)
                cv::circle(imageToVisualize, metaData.jointsOthers[person].points[part], 3, cv::Scalar{0,0,255}, -1);
    }

    // Draw text
    char imagename [100];
    if (phase_ == TRAIN)
    {
        std::stringstream ss;
        // ss << "Augmenting with:" << (augmentSelection.flip ? "flip" : "no flip")
        //    << "; Rotate " << augmentSelection.RotAndFinalSize.first << " deg; scaling: "
        //    << augmentSelection.scale << "; crop: " << augmentSelection.cropCenter.height
        //    << "," << augmentSelection.cropCenter.width;
        ss << metaData.datasetString << " " << metaData.writeNumber << " index:" << metaData.annotationListIndex
           << "; person:" << metaData.peopleIndex << "; o_scale: " << metaData.scaleSelf;
        std::string stringInfo = ss.str();
        setLabel(imageToVisualize, stringInfo, cv::Point{0, 20});

        std::stringstream ss2;
        ss2 << "mult: " << augmentSelection.scale << "; rot: " << augmentSelection.RotAndFinalSize.first << "; flip: "
            << (augmentSelection.flip?"true":"ori");
        stringInfo = ss2.str();
        setLabel(imageToVisualize, stringInfo, cv::Point{0, 40});

        cv::rectangle(imageToVisualize, cv::Point{0, (int)(imageToVisualize.rows)},
                      cv::Point{(int)(param_.crop_size_x()), (int)(param_.crop_size_y()+imageToVisualize.rows)},
                      cv::Scalar{255,255,255}, 1);

        sprintf(imagename, "visualize/augment_%04d_epoch_%03d_writenum_%03d.jpg", metaData.writeNumber,
                metaData.epoch, metaData.writeNumber);
    }
    else
    {
        const std::string stringInfo = "no augmentation for testing";
        setLabel(imageToVisualize, stringInfo, cv::Point{0, 20});

        sprintf(imagename, "visualize/augment_%04d.jpg", metaData.writeNumber);
    }
    //LOG(INFO) << "filename is " << imagename;
    cv::imwrite(imagename, imageToVisualize);
}

template<typename Dtype>
int getType(Dtype dtype)
{
    dtype++;
    if (sizeof(Dtype) == sizeof(float))
        return CV_32F;
    else if (sizeof(Dtype) == sizeof(double))
        return CV_64F;
    else
    {
        throw std::runtime_error{"Only float or double"
                                 + getLine(__LINE__, __FUNCTION__, __FILE__)};
        return CV_32F;
    }
}
// OpenPose: added ended

template<typename Dtype>
OPDataTransformer<Dtype>::OPDataTransformer(const OPTransformationParameter& param,
        Phase phase)
        // : param_(param), phase_(phase) {
        : param_(param), phase_(phase), mCurrentEpoch{-1} {
    // OpenPose: commented
    // // check if we want to use mean_file
    // if (param_.has_mean_file()) {
    //     CHECK_EQ(param_.mean_value_size(), 0) <<
    //         "Cannot specify mean_file and mean_value at the same time";
    //     const std::string& mean_file = param.mean_file();
    //     if (Caffe::root_solver()) {
    //         LOG(INFO) << "Loading mean file from: " << mean_file;
    //     }
    //     BlobProto blob_proto;
    //     ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    //     data_mean_.FromProto(blob_proto);
    // }
    // // check if we want to use mean_value
    // if (param_.mean_value_size() > 0) {
    //     CHECK(param_.has_mean_file() == false) <<
    //         "Cannot specify mean_file and mean_value at the same time";
    //     for (int c = 0; c < param_.mean_value_size(); ++c) {
    //         mean_values_.push_back(param_.mean_value(c));
    //     }
    // }
    // OpenPose: commented end
    // OpenPose: added
    LOG(INFO) << "OPDataTransformer constructor done.";
    // PoseModel
    std::tie(mPoseModel, mPoseCategory) = flagsToPoseModel(param_.model());
    // OpenPose: added end
}

// OpenPose: commented
// template <typename Dtype>
// void OPDataTransformer<Dtype>::InitRand() {
//     const bool needs_rand = param_.mirror() ||
//             (phase_ == TRAIN && param_.crop_size());
//     if (needs_rand)
//     {
//         const unsigned int rng_seed = caffe_rng_rand();
//         rng_.reset(new Caffe::RNG(rng_seed));
//     }
//     else
//         rng_.reset();
// }

// OpenPose: added
// template<typename Dtype>
// void OPDataTransformer<Dtype>::Transform(Blob<Dtype>* transformedData, Blob<Dtype>* transformedLabel,
//                                          const Datum& datum, const Datum* datumNegative)
// {
//     // Secuirty checks
//     const int datumChannels = datum.channels();
//     const int imageNum = transformedData->num();
//     const int imageChannels = transformedData->channels();
//     const int labelNum = transformedLabel->num();
//     CHECK_GE(datumChannels, 1);
//     CHECK_EQ(imageChannels, 3);
//     CHECK_EQ(imageNum, labelNum);
//     CHECK_GE(imageNum, 1);

//     auto* transformedDataPtr = transformedData->mutable_cpu_data();
//     auto* transformedLabelPtr = transformedLabel->mutable_cpu_data();
//     CPUTimer timer;
//     timer.Start();
//     generateDataAndLabel(transformedDataPtr, transformedLabelPtr, datum, datumNegative);
//     VLOG(2) << "Transform: " << timer.MicroSeconds() / 1000.0  << " ms";
// }

template<typename Dtype>
void OPDataTransformer<Dtype>::Transform(Blob<Dtype>* transformedData, Blob<Dtype>* transformedLabel,
                                         Blob<Dtype>* transformedLabelBig,
                                         const Datum& datum, const Datum* datumNegative)
{
    // Secuirty checks
    const int datumChannels = datum.channels();
    const int imageNum = transformedData->num();
    const int imageChannels = transformedData->channels();
    const int labelNum = transformedLabel->num();
    CHECK_GE(datumChannels, 1);
    CHECK_EQ(imageChannels, 3);
    CHECK_EQ(imageNum, labelNum);
    CHECK_GE(imageNum, 1);

    auto* transformedDataPtr = transformedData->mutable_cpu_data();
    auto* transformedLabelPtr = transformedLabel->mutable_cpu_data();
    Dtype* transformedLabelBigPtr = nullptr;
    if (transformedLabelBig != nullptr)
        transformedLabelBigPtr = transformedLabelBig->mutable_cpu_data();
    CPUTimer timer;
    timer.Start();
    generateDataAndLabel(transformedDataPtr, transformedLabelPtr, transformedLabelBigPtr,
                         datum, datumNegative);
    VLOG(2) << "Transform: " << timer.MicroSeconds() / 1000.0  << " ms";
}

template <typename Dtype>
int OPDataTransformer<Dtype>::getNumberChannels() const
{
    return 2 * getNumberBodyBkgAndPAF(mPoseModel);
    // // For Distance
    // return 2 * (getNumberBodyBkgAndPAF(mPoseModel) + getNumberPafChannels(mPoseModel)/2);
}
// OpenPose: end

// OpenPose: commented
// template <typename Dtype>
// int OPDataTransformer<Dtype>::Rand(int n) {
//     CHECK(rng_);
//     CHECK_GT(n, 0);
//     caffe::rng_t* rng = static_cast<caffe::rng_t*>(rng_->generator());
//     return ((*rng)() % n);
// }

// OpenPose: added
template<typename Dtype>
void OPDataTransformer<Dtype>::generateDataAndLabel(Dtype* transformedData, Dtype* transformedLabel,
                                                    Dtype* transformedLabelBig, const Datum& datum,
                                                    const Datum* datumNegative)
{
    // Parameters
    const std::string& data = datum.data();
    const int datumHeight = datum.height();
    const int datumWidth = datum.width();
    const auto datumArea = (int)(datumHeight * datumWidth);

    // Time measurement
    CPUTimer timer1;
    timer1.Start();

    // const bool hasUInt8 = data.size() > 0;
    CHECK(data.size() > 0);

    // Read meta data (LMDB channel 3)
    MetaData metaData;
    // DOME
    if (mPoseCategory == PoseCategory::DOME)
        readMetaData<Dtype>(metaData, mCurrentEpoch, data.c_str(), datumWidth, mPoseCategory, mPoseModel);
    // COCO & MPII
    else
        readMetaData<Dtype>(metaData, mCurrentEpoch, &data[3 * datumArea], datumWidth, mPoseCategory, mPoseModel);
    const auto depthEnabled = metaData.depthEnabled;

    // Read image (LMDB channel 1)
    cv::Mat image;
    // DOME
    if (mPoseCategory == PoseCategory::DOME)
    {
        const auto imageFullPath = param_.media_directory() + metaData.imageSource;
        image = cv::imread(imageFullPath, CV_LOAD_IMAGE_COLOR);
        if (image.empty())
            throw std::runtime_error{"Empty image at " + imageFullPath + getLine(__LINE__, __FUNCTION__, __FILE__)};
    }
    // COCO & MPII
    else
    {
        image = cv::Mat(datumHeight, datumWidth, CV_8UC3);
        const auto imageArea = (int)(image.rows * image.cols);
        CHECK_EQ(imageArea, datumArea);
        for (auto y = 0; y < image.rows; ++y)
        {
            const auto yOffset = (int)(y*image.cols);
            for (auto x = 0; x < image.cols; ++x)
            {
                const auto xyOffset = yOffset + x;
                cv::Vec3b& rgb = image.at<cv::Vec3b>(y, x);
                for (auto c = 0; c < 3; c++)
                {
                    const auto dIndex = (int)(c*imageArea + xyOffset);
                    // if (hasUInt8)
                        rgb[c] = static_cast<Dtype>(static_cast<uint8_t>(data[dIndex]));
                    // else
                        // rgb[c] = datum.float_data(dIndex);
                }
            }
        }
    }

    // Read background image
    cv::Mat backgroundImage;
    cv::Mat maskBackgroundImage;
    if (datumNegative != nullptr)
    {
        const std::string& data = datumNegative->data();
        const int datumNegativeHeight = datumNegative->height();
        const int datumNegativeWidth = datumNegative->width();
        const auto datumNegativeArea = (int)(datumNegativeHeight * datumNegativeWidth);
        // Background image
        backgroundImage = cv::Mat(datumNegativeHeight, datumNegativeWidth, CV_8UC3);
        const auto imageArea = (int)(backgroundImage.rows * backgroundImage.cols);
        CHECK_EQ(imageArea, datumNegativeArea);
        for (auto y = 0; y < backgroundImage.rows; ++y)
        {
            const auto yOffset = (int)(y*backgroundImage.cols);
            for (auto x = 0; x < backgroundImage.cols; ++x)
            {
                const auto xyOffset = yOffset + x;
                cv::Vec3b& rgb = backgroundImage.at<cv::Vec3b>(y, x);
                for (auto c = 0; c < 3; c++)
                {
                    const auto dIndex = (int)(c*imageArea + xyOffset);
                    rgb[c] = static_cast<Dtype>(static_cast<uint8_t>(data[dIndex]));
                }
            }
        }
        // Resize
        if (backgroundImage.cols < param_.crop_size_x() || backgroundImage.rows < param_.crop_size_y())
        {
            const auto scaleX = param_.crop_size_x() / (double)backgroundImage.cols;
            const auto scaleY = param_.crop_size_y() / (double)backgroundImage.rows;
            const auto scale = std::max(scaleX, scaleY) * 1.1; // 1.1 to avoid truncating final size down
            cv::Mat backgroundImageTemp;
            cv::resize(backgroundImage, backgroundImageTemp, cv::Size{}, scale, scale, CV_INTER_CUBIC);
            backgroundImage = backgroundImageTemp;
        }
        // Mask fro background image
        // Image size, not backgroundImage
        maskBackgroundImage = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar{0});
    }

    // Read mask miss (LMDB channel 2)
    cv::Mat maskMiss;
    // COCO
    if (mPoseCategory == PoseCategory::COCO)
    {
        maskMiss = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar{0});
        for (auto y = 0; y < maskMiss.rows; y++)
        {
            const auto yOffset = (int)(y*image.cols);
            for (auto x = 0; x < maskMiss.cols; x++)
            {
                const auto xyOffset = yOffset + x;
                const auto dIndex = (int)(4*datumArea + xyOffset);
                Dtype dElement;
                // if (hasUInt8)
                    dElement = static_cast<Dtype>(static_cast<uint8_t>(data[dIndex]));
                // else
                    // dElement = datum.float_data(dIndex);
                if (std::round(dElement/255)!=1 && std::round(dElement/255)!=0)
                    throw std::runtime_error{"Value out of {0,1}" + getLine(__LINE__, __FUNCTION__, __FILE__)};
                maskMiss.at<uchar>(y, x) = dElement; //round(dElement/255);
            }
        }
    }
    // DOME & MPII
    else
        maskMiss = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar{255});

    // Time measurement
    VLOG(2) << "  rgb[:] = datum: " << timer1.MicroSeconds()*1e-3 << " ms";

    // Depth image
    cv::Mat depth;
    if (depthEnabled)
    {
        const auto depthFullPath = param_.media_directory() + metaData.depthSource;
        depth = cv::imread(depthFullPath, CV_LOAD_IMAGE_ANYDEPTH);
        if (image.empty())
            throw std::runtime_error{"Empty depth at " + depthFullPath + getLine(__LINE__, __FUNCTION__, __FILE__)};
    }

    // timer1.Start();
    // // Clahe
    // if (param_.do_clahe())
    //     clahe(image, param_.clahe_tile_size(), param_.clahe_clip_limit());
    // BGR --> Gray --> BGR
    // if image is grey
    // cv::cvtColor(image, image, CV_GRAY2BGR);
    // VLOG(2) << "  cvtColor and CLAHE: " << timer1.MicroSeconds()*1e-3 << " ms";
    timer1.Start();

    VLOG(2) << "  ReadMeta+MetaJoints: " << timer1.MicroSeconds()*1e-3 << " ms";

    // Data augmentation
    timer1.Start();
    AugmentSelection augmentSelection;
    // // Debug - Visualize original
    // debugVisualize(image, metaData, augmentSelection, mPoseModel, phase_, param_);
    // Augmentation
    cv::Mat imageAugmented;
    cv::Mat backgroundImageAugmented;
    cv::Mat maskMissAugmented;
    cv::Mat depthAugmented;
    VLOG(2) << "   input size (" << image.cols << ", " << image.rows << ")";
    const int stride = param_.stride();
    // We only do random transform augmentSelection augmentation when training.
    if (phase_ == TRAIN)
    {
        // Temporary variables
        cv::Mat imageTemp; // Size determined by scale
        cv::Mat backgroundImageTemp;
        cv::Mat maskBackgroundImageTemp;
        cv::Mat maskMissTemp;
        cv::Mat depthTemp;
        // Swap center?
        swapCenterPoint(metaData, param_, mPoseModel);
        // Scale
        augmentSelection.scale = estimateScale(metaData, param_);
        applyScale(imageTemp, augmentSelection.scale, image);
        applyScale(maskBackgroundImageTemp, augmentSelection.scale, maskBackgroundImage);
        applyScale(maskMissTemp, augmentSelection.scale, maskMiss);
        applyScale(depthTemp, augmentSelection.scale, depth);
        applyScale(metaData, augmentSelection.scale, mPoseModel);
        // Rotation
        augmentSelection.RotAndFinalSize = estimateRotation(metaData, imageTemp.size(), param_);
        applyRotation(imageTemp, augmentSelection.RotAndFinalSize, imageTemp, 0);
        applyRotation(maskBackgroundImageTemp, augmentSelection.RotAndFinalSize, maskBackgroundImageTemp, 255);
        applyRotation(maskMissTemp, augmentSelection.RotAndFinalSize, maskMissTemp, 255);
        applyRotation(depthTemp, augmentSelection.RotAndFinalSize, depthTemp, 0);
        applyRotation(metaData, augmentSelection.RotAndFinalSize.first, mPoseModel);
        // Cropping
        augmentSelection.cropCenter = estimateCrop(metaData, param_);
        const cv::Point2i backgroundCropCenter{backgroundImage.cols/2, backgroundImage.rows/2};
        applyCrop(imageAugmented, augmentSelection.cropCenter, imageTemp, 0, param_);
        applyCrop(backgroundImageTemp, backgroundCropCenter, backgroundImage, 0, param_);
        applyCrop(maskBackgroundImage, augmentSelection.cropCenter, maskBackgroundImageTemp, 255, param_);
        applyCrop(maskMissAugmented, augmentSelection.cropCenter, maskMissTemp, 255, param_);
        applyCrop(depthAugmented, augmentSelection.cropCenter, depthTemp, 0, param_);
        applyCrop(metaData, augmentSelection.cropCenter, param_, mPoseModel);
        // Flipping
        augmentSelection.flip = estimateFlip(metaData, param_);
        applyFlip(imageAugmented, augmentSelection.flip, imageAugmented);
        applyFlip(backgroundImageAugmented, augmentSelection.flip, backgroundImageTemp);
        applyFlip(maskBackgroundImage, augmentSelection.flip, maskBackgroundImage);
        applyFlip(maskMissAugmented, augmentSelection.flip, maskMissAugmented);
        applyFlip(depthAugmented, augmentSelection.flip, depthAugmented);
        applyFlip(metaData, augmentSelection.flip, imageAugmented.cols, param_, mPoseModel);
        // Resize mask
        if (!maskMissTemp.empty())
            cv::resize(maskMissAugmented, maskMissAugmented, cv::Size{}, 1./stride, 1./stride, cv::INTER_CUBIC);
        // Final background image - elementwise multiplication
        if (!backgroundImageAugmented.empty() && !maskBackgroundImage.empty())
        {
            // Apply mask to background image
            cv::Mat backgroundImageAugmentedTemp;
            backgroundImageAugmented.copyTo(backgroundImageAugmentedTemp, maskBackgroundImage);
            // Add background image to image augmented
            cv::Mat imageAugmentedTemp;
            addWeighted(imageAugmented, 1., backgroundImageAugmentedTemp, 1., 0., imageAugmentedTemp);
            imageAugmented = imageAugmentedTemp;
        }
        if (depthEnabled && !depthTemp.empty())
            cv::resize(depthAugmented, depthAugmented, cv::Size{}, 1./stride, 1./stride, cv::INTER_CUBIC);
    }
    // Test
    else
    {
        imageAugmented = image;
        maskMissAugmented = maskMiss;
        depthAugmented = depth;
        // Resize mask
        if (!maskMissAugmented.empty())
            cv::resize(maskMissAugmented, maskMissAugmented, cv::Size{}, 1./stride, 1./stride, cv::INTER_AREA);
        if (depthEnabled)
            cv::resize(depthAugmented, depthAugmented, cv::Size{}, 1./stride, 1./stride, cv::INTER_AREA);
    }
    // // Debug - Visualize final (augmented) image
    // debugVisualize(imageAugmented, metaData, augmentSelection, mPoseModel, phase_, param_);
    // Augmentation time
    VLOG(2) << "  Aug: " << timer1.MicroSeconds()*1e-3 << " ms";
    // Data copy
    timer1.Start();
    // Copy imageAugmented into transformedData + mean-subtraction
    const int imageAugmentedArea = imageAugmented.rows * imageAugmented.cols;
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

    // Generate and copy label
    generateLabelMap(transformedLabel, imageAugmented, maskMissAugmented, metaData);
    // If required small and big label
    if (transformedLabelBig != nullptr)
    {
        const auto rezX = (int)imageAugmented.cols;
        const auto rezY = (int)imageAugmented.rows;
        const auto gridX = rezX / stride;
        const auto gridY = rezY / stride;
        const auto area = gridY * gridX;
        const auto areaBig = gridY * gridX * sAntiStride * sAntiStride;
        const auto numberTotalChannels = getNumberBodyBkgAndPAF(mPoseModel);
        const auto type = getType(Dtype(0));
        for (auto c = 0 ; c < 2*numberTotalChannels ; c++)
        {
            cv::Mat cvLabel(gridY, gridX, type, &transformedLabel[c*area]);
            cv::Mat cvLabelBig(gridY*sAntiStride, gridX*sAntiStride, type, &transformedLabelBig[c*areaBig]);
            cv::resize(cvLabel, cvLabelBig, cvLabelBig.size(), 0, 0, cv::INTER_CUBIC);
            // cv::resize(cvLabelBig, cvLabel, cvLabel.size(), 0, 0, cv::INTER_AREA);
        }
    }
    // // Default: only small label
    // else
    //     generateLabelMap(transformedLabel, imageAugmented, maskMissAugmented, metaData);
    if (depthEnabled)
        generateDepthLabelMap(transformedLabel, depthAugmented);
    VLOG(2) << "  AddGaussian+CreateLabel: " << timer1.MicroSeconds()*1e-3 << " ms";

    // // Debugging - Visualize - Write on disk
    // // if (mPoseModel == PoseModel::DOME_59)
    // {
    //     const auto strideToUse = stride / sAntiStride;
    //     auto* transformedLabelToPrint = transformedLabelBig;
    //     // const auto strideToUse = stride;
    //     // auto* transformedLabelToPrint = transformedLabel;
    //     if (metaData.writeNumber < 3)
    //     // if (metaData.writeNumber < 100)
    //     {
    //         // 1. Create `visualize` folder in training folder (where train_pose.sh is located)
    //         // 2. Comment the following if statement
    //         const auto rezX = (int)imageAugmented.cols;
    //         const auto rezY = (int)imageAugmented.rows;
    //         const auto gridX = int(std::round(rezX / strideToUse));
    //         const auto gridY = int(std::round(rezY / strideToUse));
    //         const auto channelOffset = gridY * gridX;
    //         const auto numberTotalChannels = getNumberBodyBkgAndPAF(mPoseModel);
    //         for (auto part = 0; part < numberTotalChannels; part++)
    //         {
    //             // Reduce #images saved (ideally mask images should be the same)
    //             // if (part < 1)
    //             if (part == 59)
    //             // if (part < 3 || part >= numberTotalChannels - 3)
    //             {
    //                 cv::Mat finalImage = cv::Mat::zeros(gridY, 2*gridX, CV_8UC1);
    //                 for (auto subPart = 0; subPart < 2; subPart++)
    //                 {
    //                     cv::Mat labelMap = finalImage(cv::Rect{subPart*gridX, 0, gridX, gridY});
    //                     for (auto gY = 0; gY < gridY; gY++)
    //                     {
    //                         const auto yOffset = gY*gridX;
    //                         for (auto gX = 0; gX < gridX; gX++)
    //                         {
    //                             const auto channelIndex = (part+numberTotalChannels*subPart)*channelOffset;
    //                             labelMap.at<uchar>(gY,gX) = std::min(255, std::max(0, (int)(255*transformedLabelToPrint[channelIndex + yOffset + gX])));
    //                         }
    //                     }
    //                 }
    //                 cv::resize(finalImage, finalImage, cv::Size{2*rezX, rezY}, 0, 0, cv::INTER_LINEAR);
    //                 cv::applyColorMap(finalImage, finalImage, cv::COLORMAP_JET);
    //                 for (auto subPart = 0; subPart < 2; subPart++)
    //                 {
    //                     cv::Mat labelMap = finalImage(cv::Rect{subPart*rezX, 0, rezX, rezY});
    //                     cv::addWeighted(labelMap, 0.5, imageAugmented, 0.5, 0.0, labelMap);
    //                 }
    //                 // Write on disk
    //                 char imagename [100];
    //                 sprintf(imagename, "visualize/%s_augment_%04d_label_part_%02d.jpg", param_.model().c_str(),
    //                         metaData.writeNumber, part);
    //                 cv::imwrite(imagename, finalImage);
    //             }
    //         }
    //         if (depthEnabled)
    //         {
    //             cv::Mat depthMap;
    //             cv::resize(depthAugmented, depthMap, cv::Size{}, strideToUse, strideToUse, cv::INTER_LINEAR);
    //             char imagename [100];
    //             sprintf(imagename, "visualize/%s_augment_%04d_label_part_depth.png", param_.model().c_str(),
    //                     metaData.writeNumber);
    //             cv::imwrite(imagename, depthMap);
    //         }
    //     }
    // }
}

template<typename Dtype>
void OPDataTransformer<Dtype>::generateDataAndLabel(Dtype* transformedData, Dtype* transformedLabel,
                                                    const Datum& datum, const Datum* datumNegative)
{
    generateDataAndLabel(transformedData, transformedLabel, nullptr, datum, datumNegative);
}

template<typename Dtype>
void OPDataTransformer<Dtype>::generateDepthLabelMap(Dtype* transformedLabel, const cv::Mat& depth) const
{
    const auto gridX = (int)depth.cols;
    const auto gridY = (int)depth.rows;
    const auto channelOffset = gridY * gridX;
    const auto numberBpPafChannels = getNumberBodyAndPafChannels(mPoseModel);
    // generate depth
    for (auto gY = 0; gY < gridY; gY++)
    {
        const auto yOffset = gY*gridX;
        for (auto gX = 0; gX < gridX; gX++)
        {
            const auto xyOffset = yOffset + gX;

            auto depth_val = depth.at<uint16_t>(gY, gX);

            transformedLabel[(2*numberBpPafChannels+2)*channelOffset + xyOffset] = (depth_val>0)?1.0:0.0;
            transformedLabel[(2*numberBpPafChannels+3)*channelOffset + xyOffset] = float(depth_val)/1000.0;
        }
    }
}

void keepRoiInside(cv::Rect& roi, const cv::Size& imageSize)
{
    // x,y < 0
    if (roi.x < 0)
    {
        roi.width += roi.x;
        roi.x = 0;
    }
    if (roi.y < 0)
    {
        roi.height += roi.y;
        roi.y = 0;
    }
    // Bigger than image
    if (roi.width + roi.x >= imageSize.width)
        roi.width = imageSize.width - 1 - roi.x;
    if (roi.height + roi.y >= imageSize.height)
        roi.height = imageSize.height - 1 - roi.y;
    // Width/height negative
    roi.width = std::max(0, roi.width);
    roi.height = std::max(0, roi.height);
}

float getNorm(const cv::Point2f& pointA, const cv::Point2f& pointB)
{
    const auto difference = pointA - pointB;
    return std::sqrt(difference.x*difference.x + difference.y*difference.y);
}

void maskHands(cv::Mat& maskMiss, const std::vector<float>& isVisible, const std::vector<cv::Point2f>& points,
              const float stride, const float ratio)
{
    for (auto part = 0 ; part < 2 ; part++)
    {
        const auto shoulderIndex = (part == 0 ? 5:2);
        const auto elbowIndex = shoulderIndex+1;
        const auto wristIndex = elbowIndex+1;
        if (isVisible.at(shoulderIndex) != 2 && isVisible.at(elbowIndex) != 2 && isVisible.at(wristIndex) != 2)
        {
            const auto ratioStride = 1.f / stride;
            const auto wrist = ratioStride * points.at(wristIndex);
            const auto elbow = ratioStride * points.at(elbowIndex);
            const auto shoulder = ratioStride * points.at(shoulderIndex);

            const auto distance = (int)std::round(ratio*std::max(getNorm(wrist, elbow), getNorm(elbow, shoulder)));
            const cv::Point momentum = (wrist-elbow)*0.25f;
            cv::Rect roi{(int)std::round(wrist.x + momentum.x - distance /*- wrist.x/2.f*/),
                         (int)std::round(wrist.y + momentum.y - distance /*- wrist.y/2.f*/),
                         2*distance, 2*distance};
            // Apply ROI
            keepRoiInside(roi, maskMiss.size());
            if (roi.area() > 0)
                maskMiss(roi).setTo(0.f); // For debugging use 0.5f
        }
        // // If there is no visible desired keypoints, mask out the whole background
        // else
        //     maskMiss.setTo(0.f); // For debugging use 0.5f
    }
}

void maskFeet(cv::Mat& maskMiss, const std::vector<float>& isVisible, const std::vector<cv::Point2f>& points,
              const float stride, const float ratio)
{
    for (auto part = 0 ; part < 2 ; part++)
    {
        const auto kneeIndex = 9+part*5;
        const auto ankleIndex = kneeIndex+1;
        if (isVisible.at(kneeIndex) != 2 && isVisible.at(ankleIndex) != 2)
        {
            const auto ratioStride = 1.f / stride;
            const auto knee = ratioStride * points.at(kneeIndex);
            const auto ankle = ratioStride * points.at(ankleIndex);
            const auto distance = (int)std::round(ratio*getNorm(knee, ankle));
            const cv::Point momentum = (ankle-knee)*0.15f;
            cv::Rect roi{(int)std::round(ankle.x + momentum.x)-distance,
                         (int)std::round(ankle.y + momentum.y)-distance,
                         2*distance, 2*distance};
            // Apply ROI
            keepRoiInside(roi, maskMiss.size());
            if (roi.area() > 0)
                maskMiss(roi).setTo(0.f); // For debugging use 0.5f
        }
        // // If there is no visible desired keypoints, mask out the whole background
        // else
        //     maskMiss.setTo(0.f); // For debugging use 0.5f
    }
}

template<typename Dtype>
void OPDataTransformer<Dtype>::generateLabelMap(Dtype* transformedLabel, const cv::Mat& image, const cv::Mat& maskMiss,
                                                const MetaData& metaData) const
{
    const auto rezX = (int)image.cols;
    const auto rezY = (int)image.rows;
    const auto stride = (int)param_.stride();
    const auto gridX = rezX / stride;
    const auto gridY = rezY / stride;
    const auto channelOffset = gridY * gridX;
    const auto numberBodyParts = getNumberBodyParts(mPoseModel); // #BP
    const auto numberPafChannels = getNumberPafChannels(mPoseModel); // 2 x #PAF
    const auto numberTotalChannels = getNumberBodyBkgAndPAF(mPoseModel); // numberBodyParts + numberPafChannels + 1
    // // For Distance
    // const auto numberTotalChannels = getNumberBodyBkgAndPAF(mPoseModel) + (numberPafChannels / 2); // numberBodyParts + numberPafChannels + 1

    // Labels to 0
    std::fill(transformedLabel, transformedLabel + 2*numberTotalChannels * gridY * gridX, 0.f);

    // Initialize labels to 0 or 1 (depending on maskMiss)
    // Label size = image size / stride
    for (auto gY = 0; gY < gridY; gY++)
    {
        const auto yOffset = gY*gridX;
        for (auto gX = 0; gX < gridX; gX++)
        {
            const auto xyOffset = yOffset + gX;
            const float weight = float(maskMiss.at<uchar>(gY, gX)) / 255.f;
            // Body part & PAFs & background channel & distance
            for (auto part = 0; part < numberTotalChannels; part++)
            // // For Distance
            // for (auto part = 0; part < numberTotalChannels - numberPafChannels/2; part++)
                transformedLabel[part*channelOffset + xyOffset] = weight;
        }
    }

    // Masking out channels - For COCO_YY_ZZ models (ZZ < YY)
    if (numberBodyParts > getNumberBodyPartsLmdb(mPoseModel) || mPoseModel == PoseModel::MPII_59)
    {
        // Remove BP/PAF non-labeled channels
        const auto missingChannels = getMissingChannels(mPoseModel, (mPoseModel == PoseModel::MPII_59
                                                                        ? metaData.jointsSelf.isVisible
                                                                        : std::vector<float>{}));
        for (const auto& index : missingChannels)
        {
            std::fill(&transformedLabel[index*channelOffset],
                      &transformedLabel[index*channelOffset + channelOffset], 0);
        }
        // Background
        const auto type = getType(Dtype(0));
        const auto backgroundIndex = numberPafChannels + numberBodyParts;
        cv::Mat maskMiss(gridY, gridX, type, &transformedLabel[backgroundIndex*channelOffset]);
        // If hands
        if (numberBodyParts == 59 && mPoseModel != PoseModel::MPII_59)
        {
            maskHands(maskMiss, metaData.jointsSelf.isVisible, metaData.jointsSelf.points, stride, 0.6f);
            for (const auto& jointsOther : metaData.jointsOthers)
                maskHands(maskMiss, jointsOther.isVisible, jointsOther.points, stride, 0.6f);
        }
        // If foot
        if (numberBodyParts == 23)
        {
            maskFeet(maskMiss, metaData.jointsSelf.isVisible, metaData.jointsSelf.points, stride, 0.6f);
            for (const auto& jointsOther : metaData.jointsOthers)
                maskFeet(maskMiss, jointsOther.isVisible, jointsOther.points, stride, 0.6f);
        }
    }

// TODO: Remove, temporary hack to get foot data, do nicely for 6-keypoint foot
    // Remove if required RBigToe, RSmallToe, LBigToe, LSmallToe, and Background
    if (mPoseModel == PoseModel::COCO_23 || mPoseModel == PoseModel::DOME_23_19 || mPoseModel == PoseModel::COCO_23_17)
    {
        std::vector<int> indexesToRemove;
        // PAFs
        for (const auto& index : {11, 12, 15, 16})
        {
            const auto indexBase = 2*index;
            indexesToRemove.emplace_back(indexBase);
            indexesToRemove.emplace_back(indexBase+1);
        }
        // Body parts
        for (const auto& index : {11, 12, 16, 17})
        {
            const auto indexBase = numberPafChannels + index;
            indexesToRemove.emplace_back(indexBase);
        }
        // Included in code 10-30 lines above...
        // // Dome data: Exclude (unlabeled) foot keypoints
        // if (mPoseModel == PoseModel::DOME_23_19 || mPoseModel == PoseModel::COCO_23_17)
        // {
        //     // Remove those channels
        //     for (const auto& index : indexesToRemove)
        //     {
        //         std::fill(&transformedLabel[index*channelOffset],
        //                   &transformedLabel[index*channelOffset + channelOffset], 0);
        //     }
        // }
        // // Background
        // if (mPoseModel == PoseModel::DOME_23_19 || mPoseModel == PoseModel::COCO_23_17)
        // {
        //     const auto backgroundIndex = numberPafChannels + numberBodyParts;
        //     const auto type = getType(Dtype(0));
        //     cv::Mat maskMiss(gridY, gridX, type, &transformedLabel[backgroundIndex*channelOffset]);
        //     maskFeet(maskMiss, metaData.jointsSelf.isVisible, metaData.jointsSelf.points, stride, 0.6f);
        //     for (const auto& jointsOther : metaData.jointsOthers)
        //         maskFeet(maskMiss, jointsOther.isVisible, jointsOther.points, stride, 0.6f);
        // }
        // Mask foot region over person whose feet are not anotated with a square
        if (mPoseModel == PoseModel::COCO_23)
        {
            // auto visualize = false;
            // // From current annotation
            // const auto& selfPoints = metaData.jointsSelf.points;
            // const auto& selfVisible = metaData.jointsSelf.isVisible;
            // if (selfVisible.at(11) == 2.f && selfVisible.at(12) == 2.f
            //     && selfVisible.at(16) == 2.f && selfVisible.at(17) == 2.f)
            // {
            //     // If knees and ankles visible
            //     if (selfVisible.at(9) != 2 && selfVisible.at(10) != 2
            //         && selfVisible.at(14) != 2 && selfVisible.at(15) != 2)
            //     {
            //         maskFeet(maskMiss, selfVisible, selfPoints, 0.75f);
            //     }
            // }
            // From side annotations
            for (const auto& jointsOther : metaData.jointsOthers)
            {
                const auto& otherPoints = jointsOther.points;
                const auto& otherVisible = jointsOther.isVisible;
                // If no points visible
                if (otherVisible.at(11) == 2.f && otherVisible.at(12) == 2.f
                    && otherVisible.at(16) == 2.f && otherVisible.at(17) == 2.f)
                {
                    // If knees and ankles visible
                    if (otherVisible.at(9) != 2 && otherVisible.at(10) != 2
                        && otherVisible.at(14) != 2 && otherVisible.at(15) != 2)
                    {
                        for (auto index : indexesToRemove)
                        {
                            const auto type = getType(Dtype(0));
                            cv::Mat maskMiss(gridY, gridX, type, &transformedLabel[index*channelOffset]);
                            maskFeet(maskMiss, otherVisible, otherPoints, stride, 0.6f);
                        }
                        // visualize = true;
                    }
                }
            }
            // if (visualize)
            // {
            //     // Visualizing
            //     for (auto part = 0; part < 2*numberTotalChannels; part++)
            //     {
            //         // Reduce #images saved (ideally images from 0 to numberTotalChannels should be the same)
            //         // if (part >= 11*2)
            //         if (part >= 22 && part <= numberTotalChannels)
            //         // if (part < 3 || part >= numberTotalChannels - 3)
            //         {
            //             cv::Mat labelMap = cv::Mat::zeros(gridY, gridX, CV_8UC1);
            //             for (auto gY = 0; gY < gridY; gY++)
            //             {
            //                 const auto yOffset = gY*gridX;
            //                 for (auto gX = 0; gX < gridX; gX++)
            //                     labelMap.at<uchar>(gY,gX) = (int)(transformedLabel[part*channelOffset + yOffset + gX]*255);
            //             }
            //             cv::resize(labelMap, labelMap, cv::Size{}, stride, stride, cv::INTER_LINEAR);
            //             cv::applyColorMap(labelMap, labelMap, cv::COLORMAP_JET);
            //             cv::addWeighted(labelMap, 0.5, image, 0.5, 0.0, labelMap);
            //             // Write on disk
            //             char imagename [100];
            //             sprintf(imagename, "visualize/augment_%04d_label_part_%02d.jpg", metaData.writeNumber, part);
            //             cv::imwrite(imagename, labelMap);
            //         }
            //     }
            // }
        }
    }

    // Parameters
    const auto& labelMapA = getPafIndexA(mPoseModel);
    const auto& labelMapB = getPafIndexB(mPoseModel);

    // PAFs
    const auto threshold = 1;
    for (auto i = 0 ; i < labelMapA.size() ; i++)
    {
        cv::Mat count = cv::Mat::zeros(gridY, gridX, CV_8UC1);
        // Self
        const auto& joints = metaData.jointsSelf;
        if (joints.isVisible[labelMapA[i]] <= 1 && joints.isVisible[labelMapB[i]] <= 1)
        {
            putVectorMaps(transformedLabel + (numberTotalChannels + 2*i)*channelOffset,
                          transformedLabel + (numberTotalChannels + 2*i + 1)*channelOffset,
                          // // For Distance
                          // transformedLabel + (2*numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                          // transformedLabel + (numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                          count, joints.points[labelMapA[i]], joints.points[labelMapB[i]],
                          param_.stride(), gridX, gridY, param_.sigma(), threshold);
        }

        // For every other person
        for (auto otherPerson = 0; otherPerson < metaData.numberOtherPeople; otherPerson++)
        {
            const auto& joints = metaData.jointsOthers[otherPerson];
            if (joints.isVisible[labelMapA[i]] <= 1 && joints.isVisible[labelMapB[i]] <= 1)
            {
                putVectorMaps(transformedLabel + (numberTotalChannels + 2*i)*channelOffset,
                              transformedLabel + (numberTotalChannels + 2*i + 1)*channelOffset,
                              // // For Distance
                              // transformedLabel + (2*numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                              // transformedLabel + (numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                              count, joints.points[labelMapA[i]], joints.points[labelMapB[i]],
                              param_.stride(), gridX, gridY, param_.sigma(), threshold);
            }
        }
    }

    // Body parts
    for (auto part = 0; part < numberBodyParts; part++)
    {
        // Self
        if (metaData.jointsSelf.isVisible[part] <= 1)
        {
            const auto& centerPoint = metaData.jointsSelf.points[part];
            putGaussianMaps(transformedLabel + (part+numberTotalChannels+numberPafChannels)*channelOffset,
                            centerPoint, param_.stride(), gridX, gridY, param_.sigma());
        }
        // For every other person
        for (auto otherPerson = 0; otherPerson < metaData.numberOtherPeople; otherPerson++)
        {
            if (metaData.jointsOthers[otherPerson].isVisible[part] <= 1)
            {
                const auto& centerPoint = metaData.jointsOthers[otherPerson].points[part];
                putGaussianMaps(transformedLabel + (part+numberTotalChannels+numberPafChannels)*channelOffset,
                                centerPoint, param_.stride(), gridX, gridY, param_.sigma());
            }
        }
    }

    // Background channel
    for (auto gY = 0; gY < gridY; gY++)
    {
        const auto yOffset = gY*gridX;
        for (auto gX = 0; gX < gridX; gX++)
        {
            const auto xyOffset = yOffset + gX;
            Dtype maximum = 0.;
            const auto backgroundIndex = numberTotalChannels+numberPafChannels+numberBodyParts;
            for (auto part = numberTotalChannels+numberPafChannels ; part < backgroundIndex ; part++)
            {
                const auto index = part * channelOffset + xyOffset;
                maximum = (maximum > transformedLabel[index]) ? maximum : transformedLabel[index];
            }
            transformedLabel[backgroundIndex*channelOffset + xyOffset] = std::max(Dtype(1.)-maximum, Dtype(0.));
        }
    }
}

template<typename Dtype>
void OPDataTransformer<Dtype>::putGaussianMaps(Dtype* entry, const cv::Point2f& centerPoint, const int stride,
                                               const int gridX, const int gridY, const float sigma) const
{
    //LOG(INFO) << "putGaussianMaps here we start for " << centerPoint.x << " " << centerPoint.y;
    const Dtype start = stride/2.f - 0.5f; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
    for (auto gY = 0; gY < gridY; gY++)
    {
        const auto yOffset = gY*gridX;
        for (auto gX = 0; gX < gridX; gX++)
        {
            const Dtype x = start + gX * stride;
            const Dtype y = start + gY * stride;
            const Dtype d2 = (x-centerPoint.x)*(x-centerPoint.x) + (y-centerPoint.y)*(y-centerPoint.y);
            const Dtype exponent = d2 / 2.0 / sigma / sigma;
            //ln(100) = -ln(1%)
            if (exponent <= 4.6052)
            {
                const auto xyOffset = yOffset + gX;
                // Option a) Max
                entry[xyOffset] = std::min(Dtype(1), std::max(entry[xyOffset], std::exp(-exponent)));
                // // Option b) Average
                // entry[xyOffset] += std::exp(-exponent);
                // if (entry[xyOffset] > 1)
                //     entry[xyOffset] = 1;
            }
        }
    }
}

template<typename Dtype>
void OPDataTransformer<Dtype>::putVectorMaps(Dtype* entryX, Dtype* entryY, cv::Mat& count, const cv::Point2f& centerA,
                                             const cv::Point2f& centerB, const int stride, const int gridX,
                                             const int gridY, const float sigma, const int threshold) const
// void OPDataTransformer<Dtype>::putVectorMaps(Dtype* entryX, Dtype* entryY, Dtype* entryD, Dtype* entryDMask,
//                                              cv::Mat& count, const cv::Point2f& centerA,
//                                              const cv::Point2f& centerB, const int stride, const int gridX,
//                                              const int gridY, const float sigma, const int threshold) const
{
    const auto scaleLabel = Dtype(1)/Dtype(stride);
    const auto centerALabelScale = scaleLabel * centerA;
    const auto centerBLabelScale = scaleLabel * centerB;
    cv::Point2f directionAB = centerBLabelScale - centerALabelScale;
    const auto distanceAB = std::sqrt(directionAB.x*directionAB.x + directionAB.y*directionAB.y);
    directionAB *= (Dtype(1) / distanceAB);

    // // For Distance
    // const auto dMin = Dtype(0);
    // const auto dMax = Dtype(std::sqrt(gridX*gridX + gridY*gridY));
    // const auto dRange = dMax - dMin;
    // const auto entryDValue = 2*(distanceAB - dMin)/dRange - 1; // Main range: [-1, 1], -1 is 0px-distance, 1 is 368 / stride x sqrt(2) px of distance

    // If PAF is not 0 or NaN (e.g. if PAF perpendicular to image plane)
    if (!isnan(directionAB.x) && !isnan(directionAB.y))
    {
        const int minX = std::max(0,
                                  int(std::round(std::min(centerALabelScale.x, centerBLabelScale.x) - threshold)));
        const int maxX = std::min(gridX,
                                  int(std::round(std::max(centerALabelScale.x, centerBLabelScale.x) + threshold)));
        const int minY = std::max(0,
                                  int(std::round(std::min(centerALabelScale.y, centerBLabelScale.y) - threshold)));
        const int maxY = std::min(gridY,
                                  int(std::round(std::max(centerALabelScale.y, centerBLabelScale.y) + threshold)));
        for (auto gY = minY; gY < maxY; gY++)
        {
            const auto yOffset = gY*gridX;
            for (auto gX = minX; gX < maxX; gX++)
            {
                const auto xyOffset = yOffset + gX;
                const cv::Point2f ba{gX - centerALabelScale.x, gY - centerALabelScale.y};
                const float distance = std::abs(ba.x*directionAB.y - ba.y*directionAB.x);
                if (distance <= threshold)
                {
                    auto& counter = count.at<uchar>(gY, gX);
                    if (counter == 0)
                    {
                        entryX[xyOffset] = directionAB.x;
                        entryY[xyOffset] = directionAB.y;
                        // // For Distance
                        // entryD[xyOffset] = entryDValue;
                        // entryDMask[xyOffset] = Dtype(1);
                    }
                    else
                    {
                        entryX[xyOffset] = (entryX[xyOffset]*counter + directionAB.x) / (counter + 1);
                        entryY[xyOffset] = (entryY[xyOffset]*counter + directionAB.y) / (counter + 1);
                        // // For Distance
                        // entryD[xyOffset] = (entryD[xyOffset]*counter + entryDValue) / (counter + 1);
                    }
                    counter++;
                }
            }
        }
    }
}
// OpenPose: added end

INSTANTIATE_CLASS(OPDataTransformer);

}  // namespace caffe
