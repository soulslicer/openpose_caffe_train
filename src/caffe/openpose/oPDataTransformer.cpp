#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
// OpenPose: added
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
// OpenPose: added end
#endif  // USE_OPENCV

// OpenPose: added
#include <atomic>
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

// OpenPose: added
// Remainder
// COCO_BODY_PARTS {
//     {0,  "Nose"},
//     {1,  "LEye"},
//     {2,  "REye"},
//     {3,  "LEar"},
//     {4,  "REar"},
//     {5,  "LShoulder"},
//     {6,  "RShoulder"},
//     {7,  "LElbow"},
//     {8,  "RElbow"},
//     {9,  "LWrist"},
//     {10, "RWrist"},
//     {11, "LHip"},
//     {12, "RHip"},
//     {13, "LKnee"},
//     {14, "RKnee"},
//     {15, "LAnkle"},
//     {16, "RAnkle"},
//     {17, "Background"},
// };
// OPENPOSE_BODY_PARTS {
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
//     {18, "Background"},
// };
// OPENPOSE_DEPTH_BODY_PARTS {
//     {0,  "Nose"},
//     {1,  "RShoulder"},
//     {2,  "RElbow"},
//     {3,  "RWrist"},
//     {4,  "LShoulder"},
//     {5,  "LElbow"},
//     {6,  "LWrist"},
//     {7,  "RHip"},
//     {8,  "RKnee"},
//     {9,  "RAnkle"},
//     {10, "LHip"},
//     {11, "LKnee"},
//     {12, "LAnkle"},
//     {13, "REye"},
//     {14, "LEye"},
//     {15, "REar"},
//     {16, "LEar"},
//     {17, "Background"},
// };
const std::array<int, (int)PoseModel::Size> NUMBER_BODY_PARTS{18, 22, 18};
const std::array<int, (int)PoseModel::Size> NUMBER_PARTS_LMDB{17, 17, 17};
const std::array<int, (int)PoseModel::Size> NUMBER_PAFS{2*19, 2*23, 2*19};
const std::array<int, (int)PoseModel::Size> NUMBER_BODY_AND_PAF_CHANNELS{NUMBER_BODY_PARTS[0]+NUMBER_PAFS[0],
                                                                         NUMBER_BODY_PARTS[1]+NUMBER_PAFS[1],
                                                                         NUMBER_BODY_PARTS[2]+NUMBER_PAFS[2]};
const std::array<std::vector<std::vector<int>>, (int)PoseModel::Size> TRANSFORM_MODEL_TO_OURS{
    std::vector<std::vector<int>>{
        {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}                       // COCO_18
    },
    std::vector<std::vector<int>>{
        {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}, {16},{15},{16},{15}  // BODY_22
    },
    std::vector<std::vector<int>>{
        {0},{1,4}, {1},{2},{3},  {4},{5},{6},  {7}, {8}, {9}, {10},{11},{12},{13},{14},{15},{16}                    // DOME_18
    }
};
const std::array<std::vector<int>, (int)PoseModel::Size> SWAP_LEFTS{
    std::vector<int>{5,6,7,11,12,13,15,17},                                                                 // COCO_18
    std::vector<int>{5,6,7,11,12,13,15,17, 19,21},                                                          // BODY_22
    std::vector<int>{5,6,7,11,12,13,15,17}                                                                  // DOME_18
};
const std::array<std::vector<int>, (int)PoseModel::Size> SWAP_RIGHTS{
    std::vector<int>{2,3,4, 8,9,10,14,16},                                                                  // COCO_18
    std::vector<int>{2,3,4, 8,9,10,14,16, 18,20},                                                           // BODY_22
    std::vector<int>{2,3,4, 8,9,10,14,16}                                                                   // DOME_18
};
const std::array<std::vector<int>, (int)PoseModel::Size> LABEL_MAP_A{
    std::vector<int>{1, 8,  9, 1,  11, 12, 1, 2, 3, 2,  1, 5, 6, 5,  1, 0,  0,  14, 15},                    // COCO_18
    std::vector<int>{1, 8,  9, 1,  11, 12, 1, 2, 3, 2,  1, 5, 6, 5,  1, 0,  0,  14, 15, 10, 13, 10, 13},    // BODY_22
    std::vector<int>{1, 8,  9, 1,  11, 12, 1, 2, 3, 2,  1, 5, 6, 5,  1, 0,  0,  14, 15}                     // DOME_18
};
const std::array<std::vector<int>, (int)PoseModel::Size> LABEL_MAP_B{
    std::vector<int>{8, 9, 10, 11, 12, 13, 2, 3, 4, 16, 5, 6, 7, 17, 0, 14, 15, 16, 17},                    // COCO_18
    std::vector<int>{8, 9, 10, 11, 12, 13, 2, 3, 4, 16, 5, 6, 7, 17, 0, 14, 15, 16, 17, 18, 19, 20, 21},    // BODY_22
    std::vector<int>{8, 9, 10, 11, 12, 13, 2, 3, 4, 16, 5, 6, 7, 17, 0, 14, 15, 16, 17}                     // DOME_18
};
PoseModel flagsToPoseModel(const std::string& poseModeString)
{
    if (poseModeString == "COCO_18")
        return PoseModel::COCO_18;
    else if (poseModeString == "BODY_22")
        return PoseModel::BODY_22;
    else if (poseModeString == "DOME_18")
        return PoseModel::DOME_18;
    // else
    throw std::runtime_error{"String does not correspond to any model (COCO_18, DOME_18, BODY_22, ...)" + getLine(__LINE__, __FUNCTION__, __FILE__)};
    return PoseModel::COCO_18;
}
// OpenPose: added end

template<typename Dtype>
OPDataTransformer<Dtype>::OPDataTransformer(const OPTransformationParameter& param,
        Phase phase)
        : param_(param), phase_(phase) {
    // check if we want to use mean_file
    if (param_.has_mean_file()) {
        CHECK_EQ(param_.mean_value_size(), 0) <<
            "Cannot specify mean_file and mean_value at the same time";
        const std::string& mean_file = param.mean_file();
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
    LOG(INFO) << "OPDataTransformer constructor done.";
    mIsTableSet = false;
    // PoseModel
    mPoseModel = flagsToPoseModel(param_.model());
    if (mPoseModel == PoseModel::Size)
        throw std::runtime_error{"Invalid mPoseModel" + getLine(__LINE__, __FUNCTION__, __FILE__)};
    // OpenPose: added end
}

template <typename Dtype>
void OPDataTransformer<Dtype>::InitRand() {
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

// OpenPose: added
template<typename Dtype>
void OPDataTransformer<Dtype>::Transform(Blob<Dtype>* transformedData, Blob<Dtype>* transformedLabel, const Datum& datum, const Datum* datumNegative)
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
    CPUTimer timer;
    timer.Start();
    generateDataAndLabel(transformedDataPtr, transformedLabelPtr, datum, datumNegative);
    VLOG(2) << "Transform: " << timer.MicroSeconds() / 1000.0  << " ms";
}

template <typename Dtype>
int OPDataTransformer<Dtype>::getNumberChannels() const
{
    return 2 * getNumberBodyBkgAndPAF();
}

template <typename Dtype>
int OPDataTransformer<Dtype>::getNumberBodyBkgAndPAF() const
{
    return NUMBER_BODY_AND_PAF_CHANNELS[(int)mPoseModel] + 1;
}
// OpenPose: end

template <typename Dtype>
int OPDataTransformer<Dtype>::Rand(int n) {
    CHECK(rng_);
    CHECK_GT(n, 0);
    caffe::rng_t* rng = static_cast<caffe::rng_t*>(rng_->generator());
    return ((*rng)() % n);
}

// OpenPose: added
template<typename Dtype>
void OPDataTransformer<Dtype>::generateDataAndLabel(Dtype* transformedData, Dtype* transformedLabel, const Datum& datum, const Datum* datumNegative)
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
    if (mPoseModel == PoseModel::DOME_18)
        readMetaData(metaData, data.c_str(), datumWidth);
    else //if (hasUInt8)
        readMetaData(metaData, &data[3 * datumArea], datumWidth);
    // else
    // {
    //     throw std::runtime_error{"Error" + getLine(__LINE__, __FUNCTION__, __FILE__)};
    //     std::string metadataString(datumArea, '\0');
    //     for (auto y = 0; y < datumHeight; ++y)
    //     {
    //         const auto yOffset = (int)(y*datumWidth);
    //         for (auto x = 0; x < datumWidth; ++x)
    //         {
    //             const auto xyOffset = yOffset + x;
    //             const auto dIndex = (int)(3*datumArea + xyOffset);
    //             metadataString[xyOffset] = datum.float_data(dIndex);
    //         }
    //     }
    //     readMetaData(metaData, metadataString.c_str(), datumWidth);
    // }
    if (param_.transform_body_joint()) // we expect to transform body joints, and not to transform hand joints
        transformMetaJoints(metaData);
    const auto depthEnabled = metaData.depthEnabled;

    // Read image (LMDB channel 1)
    cv::Mat image;
    if (mPoseModel == PoseModel::DOME_18)
    {
        const auto imageFullPath = param_.media_directory() + metaData.imageSource;
        image = cv::imread(imageFullPath, CV_LOAD_IMAGE_COLOR);
        if (image.empty())
            throw std::runtime_error{"Empty image at " + imageFullPath + getLine(__LINE__, __FUNCTION__, __FILE__)};
    }
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
        maskBackgroundImage = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar{0}); // Image size, not backgroundImage
    }

    // Read mask miss (LMDB channel 2)
    cv::Mat maskMiss;
    if (mPoseModel == PoseModel::DOME_18)
        maskMiss = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar{255});
    else
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

    // Time measurement
    VLOG(2) << "  rgb[:] = datum: " << timer1.MicroSeconds()*1e-3 << " ms";
    timer1.Start();

    // Depth image
    cv::Mat depth;
    if (depthEnabled)
    {
        const auto depthFullPath = param_.media_directory() + metaData.depthSource;
        depth = cv::imread(depthFullPath, CV_LOAD_IMAGE_ANYDEPTH);
        if (image.empty())
            throw std::runtime_error{"Empty depth at " + depthFullPath + getLine(__LINE__, __FUNCTION__, __FILE__)};
    }

    // Clahe
    if (param_.do_clahe())
        clahe(image, param_.clahe_tile_size(), param_.clahe_clip_limit());

    // Gray --> BGR
    if (param_.gray() == 1)
    {
        cv::cvtColor(image, image, CV_BGR2GRAY);
        cv::cvtColor(image, image, CV_GRAY2BGR);
    }
    VLOG(2) << "  color: " << timer1.MicroSeconds()*1e-3 << " ms";
    timer1.Start();

    VLOG(2) << "  ReadMeta+MetaJoints: " << timer1.MicroSeconds()*1e-3 << " ms";

    // Data augmentation
    timer1.Start();
    AugmentSelection augmentSelection;
    // Visualize original
    if (param_.visualize())
        visualize(image, metaData, augmentSelection);
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
        // Scale
        augmentSelection.scale = estimateScale(metaData);
        applyScale(imageTemp, augmentSelection.scale, image);
        applyScale(maskBackgroundImageTemp, augmentSelection.scale, maskBackgroundImage);
        applyScale(maskMissTemp, augmentSelection.scale, maskMiss);
        applyScale(depthTemp, augmentSelection.scale, depth);
        applyScale(metaData, augmentSelection.scale);
        // Rotation
        augmentSelection.RotAndFinalSize = estimateRotation(metaData, imageTemp.size());
        applyRotation(imageTemp, augmentSelection.RotAndFinalSize, imageTemp, 0);
        applyRotation(maskBackgroundImageTemp, augmentSelection.RotAndFinalSize, maskBackgroundImageTemp, 255);
        applyRotation(maskMissTemp, augmentSelection.RotAndFinalSize, maskMissTemp, 255);
        applyRotation(depthTemp, augmentSelection.RotAndFinalSize, depthTemp, 0);
        applyRotation(metaData, augmentSelection.RotAndFinalSize.first);
        // Cropping
        augmentSelection.cropCenter = estimateCrop(metaData);
        const cv::Point2i backgroundCropCenter{backgroundImage.cols/2, backgroundImage.rows/2};
        applyCrop(imageAugmented, augmentSelection.cropCenter, imageTemp, 0);
        applyCrop(backgroundImageTemp, backgroundCropCenter, backgroundImage, 0);
        applyCrop(maskBackgroundImage, augmentSelection.cropCenter, maskBackgroundImageTemp, 255);
        applyCrop(maskMissAugmented, augmentSelection.cropCenter, maskMissTemp, 255);
        applyCrop(depthAugmented, augmentSelection.cropCenter, depthTemp, 0);
        applyCrop(metaData, augmentSelection.cropCenter);
        // Flipping
        augmentSelection.flip = estimateFlip(metaData);
        applyFlip(imageAugmented, augmentSelection.flip, imageAugmented);
        applyFlip(backgroundImageAugmented, augmentSelection.flip, backgroundImageTemp);
        applyFlip(maskBackgroundImage, augmentSelection.flip, maskBackgroundImage);
        applyFlip(maskMissAugmented, augmentSelection.flip, maskMissAugmented);
        applyFlip(depthAugmented, augmentSelection.flip, depthAugmented);
        applyFlip(metaData, augmentSelection.flip, imageAugmented.cols);
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
    }
    // Test
    else
    {
        imageAugmented = image;
        maskMissAugmented = maskMiss;
        depthAugmented = depth;
    }
    // Visualize final
    if (param_.visualize())
        visualize(imageAugmented, metaData, augmentSelection);
    VLOG(2) << "  Aug: " << timer1.MicroSeconds()*1e-3 << " ms";
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
    VLOG(2) << "  AddGaussian+CreateLabel: " << timer1.MicroSeconds()*1e-3 << " ms";

    // Visualize
    // 1. Create `visualize` folder in training folder (where train_pose.sh is located)
    // 2. Comment the following if statement
    if (param_.visualize())
    {
        const auto rezX = (int)imageAugmented.cols;
        const auto rezY = (int)imageAugmented.rows;
        const auto stride = (int)param_.stride();
        const auto gridX = rezX / stride;
        const auto gridY = rezY / stride;
        const auto channelOffset = gridY * gridX;
        const auto numberBodyBkgPAFParts = getNumberBodyBkgAndPAF();
        for (auto part = 0; part < 2*numberBodyBkgPAFParts; part++)
        {
            // Original image
            // char imagename [100];
            // sprintf(imagename, "visualize/augment_%04d_label_part_000.jpg", metaData.writeNumber);
            // cv::imwrite(imagename, imageAugmented);
            // Reduce #images saved (ideally images from 0 to numberBodyBkgPAFParts should be the same)
            if (part < 3 || part >= numberBodyBkgPAFParts - 3)
            {
                cv::Mat labelMap = cv::Mat::zeros(gridY, gridX, CV_8UC1);
                for (auto gY = 0; gY < gridY; gY++)
                {
                    const auto yOffset = gY*gridX;
                    for (auto gX = 0; gX < gridX; gX++)
                        labelMap.at<uchar>(gY,gX) = (int)(transformedLabel[part*channelOffset + yOffset + gX]*255);
                }
                cv::resize(labelMap, labelMap, cv::Size{}, stride, stride, cv::INTER_LINEAR);
                cv::applyColorMap(labelMap, labelMap, cv::COLORMAP_JET);
                cv::addWeighted(labelMap, 0.5, imageAugmented, 0.5, 0.0, labelMap);
                // Write on disk
                char imagename [100];
                sprintf(imagename, "visualize/augment_%04d_label_part_%02d.jpg", metaData.writeNumber, part);
                cv::imwrite(imagename, labelMap);
            }
        }
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
    const auto numberBodyAndPAFParts = NUMBER_BODY_AND_PAF_CHANNELS[(int)mPoseModel];
    const auto numberBodyBkgPAFParts = getNumberBodyBkgAndPAF();

    // Labels to 0
    std::fill(transformedLabel, transformedLabel + 2*numberBodyBkgPAFParts * gridY * gridX, 0.f);

    // Label size = image size / stride
    for (auto gY = 0; gY < gridY; gY++)
    {
        const auto yOffset = gY*gridX;
        for (auto gX = 0; gX < gridX; gX++)
        {
            const auto xyOffset = yOffset + gX;
            for (auto part = 0; part < numberBodyAndPAFParts; part++)
            {
                const float weight = float(maskMiss.at<uchar>(gY, gX)) / 255.f;
                transformedLabel[part*channelOffset + xyOffset] = weight;
            }
            // background channel
            transformedLabel[numberBodyAndPAFParts*channelOffset + xyOffset] = float(maskMiss.at<uchar>(gY, gX)) / 255.f;
        }
    }

    // Parameters
    const auto numberBodyParts = NUMBER_BODY_PARTS[(int)mPoseModel];
    const auto numberPAFChannels = NUMBER_PAFS[(int)mPoseModel]+1;
    const auto& labelMapA = LABEL_MAP_A[(int)mPoseModel];
    const auto& labelMapB = LABEL_MAP_B[(int)mPoseModel];

    // PAFs
    const auto threshold = 1;
    for (auto i = 0 ; i < labelMapA.size() ; i++)
    {
        cv::Mat count = cv::Mat::zeros(gridY, gridX, CV_8UC1);
        const auto& joints = metaData.jointsSelf;
        if (joints.isVisible[labelMapA[i]]<=1 && joints.isVisible[labelMapB[i]]<=1)
        {
            // putVecMaps
            putVecMaps(transformedLabel + (numberBodyBkgPAFParts + 2*i)*channelOffset,
                       transformedLabel + (numberBodyBkgPAFParts + 2*i + 1)*channelOffset,
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
                putVecMaps(transformedLabel + (numberBodyBkgPAFParts + 2*i)*channelOffset,
                           transformedLabel + (numberBodyBkgPAFParts + 2*i + 1)*channelOffset,
                           count, jointsOthers.points[labelMapA[i]], jointsOthers.points[labelMapB[i]],
                           param_.stride(), gridX, gridY, param_.sigma(), threshold); //self
            }
        }
    }

    // Body parts
    for (auto part = 0; part < numberBodyParts; part++)
    {
        if (metaData.jointsSelf.isVisible[part] <= 1)
        {
            const auto& centerPoint = metaData.jointsSelf.points[part];
            putGaussianMaps(transformedLabel + (part+numberBodyAndPAFParts+numberPAFChannels)*channelOffset, centerPoint, param_.stride(),
                            gridX, gridY, param_.sigma()); //self
        }
        //for every other person
        for (auto otherPerson = 0; otherPerson < metaData.numberOtherPeople; otherPerson++)
        {
            if (metaData.jointsOthers[otherPerson].isVisible[part] <= 1)
            {
                const auto& centerPoint = metaData.jointsOthers[otherPerson].points[part];
                putGaussianMaps(transformedLabel + (part+numberBodyAndPAFParts+numberPAFChannels)*channelOffset, centerPoint, param_.stride(),
                                gridX, gridY, param_.sigma());
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
            for (auto part = numberBodyAndPAFParts+numberPAFChannels ; part < numberBodyAndPAFParts+numberBodyBkgPAFParts ; part++)
            {
                const auto index = part * channelOffset + xyOffset;
                maximum = (maximum > transformedLabel[index]) ? maximum : transformedLabel[index];
            }
            transformedLabel[(2*numberBodyAndPAFParts+1)*channelOffset + xyOffset] = std::max(Dtype(1.)-maximum, Dtype(0.));
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

std::atomic<int> sVisualizationCounter{0};
template<typename Dtype>
void OPDataTransformer<Dtype>::visualize(const cv::Mat& image, const MetaData& metaData, const AugmentSelection& augmentSelection) const
{
    cv::Mat imageToVisualize = image.clone();

    cv::rectangle(imageToVisualize, metaData.objpos-cv::Point2f{3.f,3.f}, metaData.objpos+cv::Point2f{3.f,3.f},
                  cv::Scalar{255,255,0}, CV_FILLED);
    const auto numberBodyAndPAFParts = NUMBER_BODY_AND_PAF_CHANNELS[(int)mPoseModel];
    for (auto part = 0 ; part < numberBodyAndPAFParts ; part++)
    {
        const auto currentPoint = metaData.jointsSelf.points[part];
        // Hand case
        if (numberBodyAndPAFParts == 21)
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
        else if (numberBodyAndPAFParts == 9)
        {
            if (part==0 || part==1 || part==2 || part==6)
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{0,0,255}, -1);
            else if (part==3 || part==4 || part==5 || part==7)
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,0,0}, -1);
            else
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,255,0}, -1);
        }
        // Body case (CPM)
        else if (numberBodyAndPAFParts == 14 || numberBodyAndPAFParts == 28)
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
        for (auto part = 0 ; part < numberBodyAndPAFParts ; part++)
            if (metaData.jointsOthers[person].isVisible[part] <= 1)
                cv::circle(imageToVisualize, metaData.jointsOthers[person].points[part], 3, cv::Scalar{0,0,255}, -1);
    }

    // Draw text
    char imagename [100];
    if (phase_ == TRAIN)
    {
        std::stringstream ss;
        // ss << "Augmenting with:" << (augmentSelection.flip ? "flip" : "no flip")
        //    << "; Rotate " << augmentSelection.RotAndFinalSize.first << " deg; scaling: " << augmentSelection.scale << "; crop: "
        //    << augmentSelection.cropCenter.height << "," << augmentSelection.cropCenter.width;
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

        sprintf(imagename, "visualize/augment_%04d_epoch_%03d_writenum_%03d.jpg", sVisualizationCounter.load(), metaData.epoch, metaData.writeNumber);
    }
    else
    {
        const std::string stringInfo = "no augmentation for testing";
        setLabel(imageToVisualize, stringInfo, cv::Point{0, 20});

        sprintf(imagename, "visualize/augment_%04d.jpg", sVisualizationCounter.load());
    }
    //LOG(INFO) << "filename is " << imagename;
    cv::imwrite(imagename, imageToVisualize);
    sVisualizationCounter++;
}

template<typename Dtype>
float OPDataTransformer<Dtype>::estimateScale(const MetaData& metaData) const
{
    // Estimate random scale
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
    return scaleAbs * scaleMultiplier;
}

template<typename Dtype>
void OPDataTransformer<Dtype>::applyScale(cv::Mat& imageAugmented, const float scale, const cv::Mat& image) const
{
    // Scale image
    if (!image.empty())
        cv::resize(image, imageAugmented, cv::Size{}, scale, scale, cv::INTER_CUBIC);
}

template<typename Dtype>
void OPDataTransformer<Dtype>::applyScale(MetaData& metaData, const float scale) const
{
    // Update metaData
    metaData.objpos *= scale;
    const auto numberBodyAndPAFParts = NUMBER_BODY_AND_PAF_CHANNELS[(int)mPoseModel];
    for (auto part = 0; part < numberBodyAndPAFParts ; part++)
        metaData.jointsSelf.points[part] *= scale;
    for (auto person=0; person<metaData.numberOtherPeople; person++)
    {
        metaData.objPosOthers[person] *= scale;
        for (auto part = 0; part < numberBodyAndPAFParts ; part++)
            metaData.jointsOthers[person].points[part] *= scale;
    }
}

template<typename Dtype>
std::pair<cv::Mat, cv::Size> OPDataTransformer<Dtype>::estimateRotation(const MetaData& metaData, const cv::Size& imageSize) const
{
    // Estimate random rotation
    float rotation;
    if (param_.aug_way() == "rand")
    {
        const float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        rotation = (dice - 0.5f) * 2 * param_.max_rotate_degree();
    }
    else if (param_.aug_way() == "table")
        rotation = mAugmentationDegs[metaData.writeNumber][metaData.epoch % param_.num_total_augs()];
    else
        throw std::runtime_error{"Unhandled exception" + getLine(__LINE__, __FUNCTION__, __FILE__)};
    // Estimate center & BBox
    const cv::Point2f center{imageSize.width / 2.f, imageSize.height / 2.f};
    const cv::Rect bbox = cv::RotatedRect(center, imageSize, rotation).boundingRect();
    // Adjust transformation matrix
    cv::Mat Rot = cv::getRotationMatrix2D(center, rotation, 1.0);
    Rot.at<double>(0,2) += bbox.width/2. - center.x;
    Rot.at<double>(1,2) += bbox.height/2. - center.y;
    return std::make_pair(Rot, bbox.size());
}

template<typename Dtype>
void OPDataTransformer<Dtype>::applyRotation(cv::Mat& imageAugmented, const std::pair<cv::Mat, cv::Size> RotAndFinalSize,
                                             const cv::Mat& image,
                                             const unsigned char defaultBorderValue) const
{
    // Rotate image
    if (!image.empty())
        cv::warpAffine(image, imageAugmented, RotAndFinalSize.first, RotAndFinalSize.second, cv::INTER_CUBIC, cv::BORDER_CONSTANT,
                       cv::Scalar{(double)defaultBorderValue});
}

template<typename Dtype>
void OPDataTransformer<Dtype>::applyRotation(MetaData& metaData, const cv::Mat& Rot) const
{
    // Update metaData
    rotatePoint(metaData.objpos, Rot);
    const auto numberBodyAndPAFParts = NUMBER_BODY_AND_PAF_CHANNELS[(int)mPoseModel];
    for (auto part = 0 ; part < numberBodyAndPAFParts ; part++)
        rotatePoint(metaData.jointsSelf.points[part], Rot);
    for (auto person = 0; person < metaData.numberOtherPeople; person++)
    {
        rotatePoint(metaData.objPosOthers[person], Rot);
        for (auto part = 0; part < numberBodyAndPAFParts ; part++)
            rotatePoint(metaData.jointsOthers[person].points[part], Rot);
    }
}

bool onPlane(const cv::Point& point, const cv::Size& imageSize)
{
    return (point.x >= 0 && point.y >= 0 && point.x < imageSize.width && point.y < imageSize.height);
}

template<typename Dtype>
cv::Point2i OPDataTransformer<Dtype>::estimateCrop(const MetaData& metaData) const
{
    // Estimate random crop
    const float diceX = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
    const float diceY = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]

    const cv::Size pointOffset{int((diceX - 0.5f) * 2.f * param_.center_perterb_max()),
                               int((diceY - 0.5f) * 2.f * param_.center_perterb_max())};
    const cv::Point2i cropCenter{
        (int)(metaData.objpos.x + pointOffset.width),
        (int)(metaData.objpos.y + pointOffset.height),
    };
    return cropCenter;
}

template<typename Dtype>
void OPDataTransformer<Dtype>::applyCrop(cv::Mat& imageAugmented, const cv::Point2i& cropCenter,
                                         const cv::Mat& image, const unsigned char defaultBorderValue) const
{
    if (!image.empty())
    {
        // Security checks
        if (imageAugmented.data == image.data)
            throw std::runtime_error{"Input and output images must be different" + getLine(__LINE__, __FUNCTION__, __FILE__)};
        // Parameters
        const auto cropX = (int) param_.crop_size_x();
        const auto cropY = (int) param_.crop_size_y();
        // Crop image
        // 1. Allocate memory
        imageAugmented = cv::Mat(cropY, cropX, image.type(), cv::Scalar{(double)defaultBorderValue});
        // 2. Fill memory
        if (imageAugmented.type() == CV_8UC3)
        {
            for (auto y = 0 ; y < cropY ; y++)
            {
                for (auto x = 0 ; x < cropX ; x++)
                {
                    const int xOrigin = cropCenter.x - cropX/2 + x;
                    const int yOrigin = cropCenter.y - cropY/2 + y;
                    if (onPlane(cv::Point{xOrigin, yOrigin}, image.size()))
                        imageAugmented.at<cv::Vec3b>(y,x) = image.at<cv::Vec3b>(yOrigin, xOrigin);
                }
            }
        }
        else if (imageAugmented.type() == CV_8UC1)
        {
            for (auto y = 0 ; y < cropY ; y++)
            {
                for (auto x = 0 ; x < cropX ; x++)
                {
                    const int xOrigin = cropCenter.x - cropX/2 + x;
                    const int yOrigin = cropCenter.y - cropY/2 + y;
                    if (onPlane(cv::Point{xOrigin, yOrigin}, image.size()))
                        imageAugmented.at<uchar>(y,x) = image.at<uchar>(yOrigin, xOrigin);
                }
            }
        }
        else if (imageAugmented.type() == CV_16UC1)
        {
            for (auto y = 0 ; y < cropY ; y++)
            {
                for (auto x = 0 ; x < cropX ; x++)
                {
                    const int xOrigin = cropCenter.x - cropX/2 + x;
                    const int yOrigin = cropCenter.y - cropY/2 + y;
                    if (onPlane(cv::Point{xOrigin, yOrigin}, image.size()))
                        imageAugmented.at<uint16_t>(y,x) = image.at<uint16_t>(yOrigin, xOrigin);
                }
            }
        }
        else
            throw std::runtime_error{"Not implemented for image.type() == " + std::to_string(imageAugmented.type())
                                     + getLine(__LINE__, __FUNCTION__, __FILE__)};
    }
}

template<typename Dtype>
void OPDataTransformer<Dtype>::applyCrop(MetaData& metaData, const cv::Point2i& cropCenter) const
{
    // Update metaData
    const auto cropX = (int) param_.crop_size_x();
    const auto cropY = (int) param_.crop_size_y();
    const int offsetLeft = -(cropCenter.x - (cropX/2));
    const int offsetUp = -(cropCenter.y - (cropY/2));
    const cv::Point2f offsetPoint{(float)offsetLeft, (float)offsetUp};
    metaData.objpos += offsetPoint;
    const auto numberBodyAndPAFParts = NUMBER_BODY_AND_PAF_CHANNELS[(int)mPoseModel];
    for (auto part = 0 ; part < numberBodyAndPAFParts ; part++)
        metaData.jointsSelf.points[part] += offsetPoint;
    for (auto person = 0 ; person < metaData.numberOtherPeople ; person++)
    {
        metaData.objPosOthers[person] += offsetPoint;
        for (auto part = 0 ; part < numberBodyAndPAFParts ; part++)
            metaData.jointsOthers[person].points[part] += offsetPoint;
    }
}

template<typename Dtype>
bool OPDataTransformer<Dtype>::estimateFlip(const MetaData& metaData) const
{
    // Estimate random flip
    bool doflip = false;
    if (param_.aug_way() == "rand")
    {
        const auto dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        doflip = (dice <= param_.flip_prob());
    }
    else if (param_.aug_way() == "table")
        doflip = (mAugmentationFlips[metaData.writeNumber][metaData.epoch % param_.num_total_augs()] == 1);
    else
        throw std::runtime_error{"Unhandled exception" + getLine(__LINE__, __FUNCTION__, __FILE__)};
    return doflip;
}

template<typename Dtype>
void OPDataTransformer<Dtype>::applyFlip(cv::Mat& imageAugmented, const bool flip, const cv::Mat& image) const
{
    // Flip image
    if (flip && !image.empty())
        cv::flip(image, imageAugmented, 1);
    // No flip
    else if (imageAugmented.data != image.data)
        imageAugmented = image.clone();
}

template<typename Dtype>
void OPDataTransformer<Dtype>::applyFlip(MetaData& metaData, const bool flip, const int imageWidth) const
{
    // Update metaData
    if (flip)
    {
        metaData.objpos.x = imageWidth - 1 - metaData.objpos.x;
        const auto numberBodyAndPAFParts = NUMBER_BODY_AND_PAF_CHANNELS[(int)mPoseModel];
        for (auto part = 0 ; part < numberBodyAndPAFParts ; part++)
            metaData.jointsSelf.points[part].x = imageWidth - 1 - metaData.jointsSelf.points[part].x;
        if (param_.transform_body_joint())
            swapLeftRight(metaData.jointsSelf);
        for (auto person = 0 ; person < metaData.numberOtherPeople ; person++)
        {
            metaData.objPosOthers[person].x = imageWidth - 1 - metaData.objPosOthers[person].x;
            for (auto part = 0 ; part < numberBodyAndPAFParts ; part++)
                metaData.jointsOthers[person].points[part].x = imageWidth - 1 - metaData.jointsOthers[person].points[part].x;
            if (param_.transform_body_joint())
                swapLeftRight(metaData.jointsOthers[person]);
        }
    }
}

template<typename Dtype>
void OPDataTransformer<Dtype>::rotatePoint(cv::Point2f& point2f, const cv::Mat& R) const
{
    cv::Mat cvMatPoint(3,1, CV_64FC1);
    cvMatPoint.at<double>(0,0) = point2f.x;
    cvMatPoint.at<double>(1,0) = point2f.y;
    cvMatPoint.at<double>(2,0) = 1;
    const cv::Mat newPoint = R * cvMatPoint;
    point2f.x = newPoint.at<double>(0,0);
    point2f.y = newPoint.at<double>(1,0);
}

template<typename Dtype>
void OPDataTransformer<Dtype>::swapLeftRight(Joints& joints) const
{
    const auto& vectorLeft = SWAP_LEFTS[(int)mPoseModel];
    const auto& vectorRight = SWAP_RIGHTS[(int)mPoseModel];
    for (auto i = 0 ; i < vectorLeft.size() ; i++)
    {
        const auto li = vectorLeft[i];
        const auto ri = vectorRight[i];
        std::swap(joints.points[ri], joints.points[li]);
        std::swap(joints.isVisible[ri], joints.isVisible[li]);
    }
}

template<typename Dtype>
void OPDataTransformer<Dtype>::setAugmentationTable(const int numData)
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
Dtype decodeNumber(const char* charPtr)
{
    Dtype pf;
    memcpy(&pf, charPtr, sizeof(Dtype));
    return pf;
}

std::string decodeString(const char* charPtr)
{
    std::string result = "";
    for (auto index = 0 ; charPtr[index] != 0 ; index++)
        result.push_back(char(charPtr[index]));
    return result;
}

//very specific to genLMDB.py
std::atomic<int> sCurrentEpoch{-1};
template<typename Dtype>
void OPDataTransformer<Dtype>::readMetaData(MetaData& metaData, const char* data, const size_t offsetPerLine)
{
    // Dataset name
    metaData.datasetString = decodeString(data);
    // Image Dimension
    metaData.imageSize = cv::Size{(int)decodeNumber<Dtype>(&data[offsetPerLine+4]),
                                  (int)decodeNumber<Dtype>(&data[offsetPerLine])};

    // Validation, #people, counters
    metaData.isValidation = (data[2*offsetPerLine] != 0);
    metaData.numberOtherPeople = (int)data[2*offsetPerLine+1];
    metaData.peopleIndex = (int)data[2*offsetPerLine+2];
    metaData.annotationListIndex = (int)(decodeNumber<Dtype>(&data[2*offsetPerLine+3]));
    metaData.writeNumber = (int)(decodeNumber<Dtype>(&data[2*offsetPerLine+7]));
    metaData.totalWriteNumber = (int)(decodeNumber<Dtype>(&data[2*offsetPerLine+11]));
    if (metaData.isValidation)
        throw std::runtime_error{"metaData.isValidation == true. Training with val. data????? " + metaData.datasetString
                                 + ", index: " + std::to_string(metaData.annotationListIndex)
                                 + getLine(__LINE__, __FUNCTION__, __FILE__)};

    // Count epochs according to counters
    if (metaData.writeNumber == 0)
        sCurrentEpoch++;
    metaData.epoch = sCurrentEpoch;
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

    // Objpos
    metaData.objpos.x = decodeNumber<Dtype>(&data[3*offsetPerLine]);
    metaData.objpos.y = decodeNumber<Dtype>(&data[3*offsetPerLine+4]);
    metaData.objpos -= cv::Point2f{1.f,1.f};
    // scaleSelf, jointsSelf
    metaData.scaleSelf = decodeNumber<Dtype>(&data[4*offsetPerLine]);
    auto& jointSelf = metaData.jointsSelf;
    const auto numberPartsInLmdb = NUMBER_PARTS_LMDB[(int)mPoseModel];
    jointSelf.points.resize(numberPartsInLmdb);
    jointSelf.isVisible.resize(numberPartsInLmdb);
    for (auto part = 0 ; part < numberPartsInLmdb; part++)
    {
        // Point
        auto& jointPoint = jointSelf.points[part];
        jointPoint.x = decodeNumber<Dtype>(&data[5*offsetPerLine+4*part]);
        jointPoint.y = decodeNumber<Dtype>(&data[6*offsetPerLine+4*part]);
        // Matlab (1-index) to C++ (0-index) --> (0,0 goes to -1,-1)
        if (mPoseModel == PoseModel::COCO_18)
            jointPoint -= cv::Point2f{1.f,1.f};
        // isVisible flag
        const auto isVisible = decodeNumber<Dtype>(&data[7*offsetPerLine+4*part]);
        CHECK_LE(isVisible, 2); // isVisible in range [0, 2]
        jointSelf.isVisible[part] = std::round(isVisible);
        if (jointSelf.isVisible[part] != 2)
            if (jointPoint.x < 0 || jointPoint.y < 0 || jointPoint.x >= metaData.imageSize.width || jointPoint.y >= metaData.imageSize.height)
                jointSelf.isVisible[part] = 2; // 2 means cropped/unlabeled, 0 means occluded but in image
        // LOG(INFO) << jointPoint.x << " " << jointPoint.y << " " << jointSelf.isVisible[part];
    }

    // Others (7 lines loaded)
    metaData.objPosOthers.resize(metaData.numberOtherPeople);
    metaData.scaleOthers.resize(metaData.numberOtherPeople);
    metaData.jointsOthers.resize(metaData.numberOtherPeople);
    for (auto person = 0 ; person < metaData.numberOtherPeople ; person++)
    {
        metaData.objPosOthers[person].x = decodeNumber<Dtype>(&data[(8+person)*offsetPerLine]);
        metaData.objPosOthers[person].y = decodeNumber<Dtype>(&data[(8+person)*offsetPerLine+4]);
        // Matlab (1-index) to C++ (0-index) --> (0,0 goes to -1,-1)
        if (mPoseModel == PoseModel::COCO_18)
            metaData.objPosOthers[person] -= cv::Point2f{1.f,1.f};
        metaData.scaleOthers[person]  = decodeNumber<Dtype>(&data[(8+metaData.numberOtherPeople)*offsetPerLine+4*person]);
    }
    // 8 + numberOtherPeople lines loaded
    for (auto person = 0 ; person < metaData.numberOtherPeople ; person++)
    {
        auto& currentPerson = metaData.jointsOthers[person];
        currentPerson.points.resize(numberPartsInLmdb);
        currentPerson.isVisible.resize(numberPartsInLmdb);
        for (auto part = 0 ; part < numberPartsInLmdb; part++)
        {
            // Point
            // Matlab (1-index) to C++ (0-index) --> (0,0 goes to -1,-1)
            auto& jointPoint = currentPerson.points[part];
            jointPoint.x = decodeNumber<Dtype>(&data[(9+metaData.numberOtherPeople+3*person)*offsetPerLine+4*part]);
            jointPoint.y = decodeNumber<Dtype>(&data[(9+metaData.numberOtherPeople+3*person+1)*offsetPerLine+4*part]);
            // Matlab (1-index) to C++ (0-index) --> (0,0 goes to -1,-1)
            if (mPoseModel == PoseModel::COCO_18)
                jointPoint -= cv::Point2f{1.f,1.f};
            // isVisible flag
            const auto isVisible = decodeNumber<Dtype>(&data[(9+metaData.numberOtherPeople+3*person+2)*offsetPerLine+4*part]);
            currentPerson.isVisible[part] = std::round(isVisible);
            if (currentPerson.isVisible[part] != 2)
                if (jointPoint.x < 0 || jointPoint.y < 0 || jointPoint.x >= metaData.imageSize.width || jointPoint.y >= metaData.imageSize.height)
                    currentPerson.isVisible[part] = 2; // 2 means cropped/unlabeled, 0 means occluded  but in image
        }
    }
    if (mPoseModel == PoseModel::DOME_18)
    {
        // Image path
        int currentLine = 8;
        if (metaData.numberOtherPeople != 0)
            currentLine = 9+4*metaData.numberOtherPeople;
        metaData.imageSource = decodeString(&data[currentLine * offsetPerLine]);
        // Depth enabled
        metaData.depthEnabled = decodeNumber<Dtype>(&data[(currentLine+1) * offsetPerLine]) != Dtype(0);
        // Depth path
        if (metaData.depthEnabled)
            metaData.depthSource = decodeString(&data[(currentLine+2) * offsetPerLine]);
    }
}

template<typename Dtype>
void OPDataTransformer<Dtype>::transformMetaJoints(MetaData& metaData) const
{
    // Transform joints in metaData from NUMBER_PARTS_LMDB[(int)mPoseModel] (specified in prototxt)
    // to NUMBER_BODY_AND_PAF_CHANNELS[(int)mPoseModel] (specified in prototxt)
    transformJoints(metaData.jointsSelf);
    for (auto& joints : metaData.jointsOthers)
        transformJoints(joints);
}

template<typename Dtype>
void OPDataTransformer<Dtype>::transformJoints(Joints& joints) const
{
    // Transform joints in metaData from NUMBER_PARTS_LMDB[(int)mPoseModel] (specified in prototxt)
    // to NUMBER_BODY_AND_PAF_CHANNELS[(int)mPoseModel] (specified in prototxt)
    auto jointsOld = joints;

    // Common operations
    const auto numberBodyAndPAFParts = NUMBER_BODY_AND_PAF_CHANNELS[(int)mPoseModel];
    joints.points.resize(numberBodyAndPAFParts);
    joints.isVisible.resize(numberBodyAndPAFParts);

    // From COCO/DomeDB to OP keypoint indexes
    const auto& modelToOurs = TRANSFORM_MODEL_TO_OURS[(int)mPoseModel];
    for (auto i = 0 ; i < modelToOurs.size() ; i++)
    {
        // Original COCO:
        //     v=0: not labeled
        //     v=1: labeled but not visible
        //     v=2: labeled and visible
        // OpenPose:
        //     v=0: labeled but not visible
        //     v=1: labeled and visible
        //     v=2: out of image / unlabeled
        // Get joints.points[i]
        joints.points[i] = cv::Point2f{0.f, 0.f};
        for (auto& modelToOursIndex : modelToOurs[i])
            joints.points[i] += jointsOld.points[modelToOursIndex];
        joints.points[i] *= (1.f / (float)modelToOurs[i].size());
        // Get joints.isVisible[i]
        joints.isVisible[i] = 1;
        for (auto& modelToOursIndex : modelToOurs[i])
        {
            // If any of them is 2 --> 2 (not in the image or unlabeled)
            if (jointsOld.isVisible[modelToOursIndex] == 2)
            {
                joints.isVisible[i] = 2;
                break;
            }
            // If no 2 but 0 -> 0 (ocluded but located)
            else if (jointsOld.isVisible[modelToOursIndex] == 0)
                joints.isVisible[i] = 0;
            // Else 1 (if all are 1s)
        }
    }
}

template <typename Dtype>
void OPDataTransformer<Dtype>::clahe(cv::Mat& bgrImage, const int tileSize, const int clipLimit) const
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
void OPDataTransformer<Dtype>::putGaussianMaps(Dtype* entry, const cv::Point2f& centerPoint, const int stride, const int gridX,
                                               const int gridY, const float sigma) const
{
    //LOG(INFO) << "putGaussianMaps here we start for " << centerPoint.x << " " << centerPoint.y;
    const float start = stride/2.f - 0.5f; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
    for (auto gY = 0; gY < gridY; gY++)
    {
        const auto yOffset = gY*gridX;
        for (auto gX = 0; gX < gridX; gX++)
        {
            const float x = start + gX * stride;
            const float y = start + gY * stride;
            const float d2 = (x-centerPoint.x)*(x-centerPoint.x) + (y-centerPoint.y)*(y-centerPoint.y);
            const float exponent = d2 / 2.0 / sigma / sigma;
            //ln(100) = -ln(1%)
            if (exponent <= 4.6052)
            {
                const auto xyOffset = yOffset + gX;
                entry[xyOffset] += std::exp(-exponent);
                if (entry[xyOffset] > 1)
                    entry[xyOffset] = 1;
            }
        }
    }
}

template<typename Dtype>
void OPDataTransformer<Dtype>::putVecMaps(Dtype* entryX, Dtype* entryY, cv::Mat& count, const cv::Point2f& centerA,
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

INSTANTIATE_CLASS(OPDataTransformer);

}  // namespace caffe
