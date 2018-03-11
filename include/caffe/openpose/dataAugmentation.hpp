#ifndef CAFFE_OPENPOSE_DATA_AUGMENTATION_HPP
#define CAFFE_OPENPOSE_DATA_AUGMENTATION_HPP
#ifdef USE_OPENCV

#include <vector>
#include <opencv2/core/core.hpp> // cv::Mat, cv::Point, cv::Size
#include "caffe/proto/caffe.pb.h"
#include "metaData.hpp"
#include "poseModel.hpp"

namespace caffe {
    // Swap center point
    void swapCenterPoint(MetaData& metaData, const OPTransformationParameter& param_, const PoseModel poseModel);
    // Scale
    float estimateScale(const MetaData& metaData, const OPTransformationParameter& param_);
    // void applyScale(cv::Mat& imageAugmented, const float scale, const cv::Mat& image);
    void applyScale(MetaData& metaData, const float scale, const PoseModel poseModel);
    // Rotation
    std::pair<cv::Mat, cv::Size> estimateRotation(const MetaData& metaData, const cv::Size& imageSize,
                                                  const OPTransformationParameter& param_);
    // void applyRotation(cv::Mat& imageAugmented, const std::pair<cv::Mat, cv::Size>& RotAndFinalSize,
    //                    const cv::Mat& image, const unsigned char defaultBorderValue);
    void applyRotation(MetaData& metaData, const cv::Mat& Rot, const PoseModel poseModel);
    // Cropping
    cv::Point2i estimateCrop(const MetaData& metaData, const OPTransformationParameter& param_);
    void applyCrop(cv::Mat& imageAugmented, const cv::Point2i& cropCenter, const cv::Mat& image,
                   const unsigned char defaultBorderValue, const cv::Size& cropSize);
    void applyCrop(MetaData& metaData, const cv::Point2i& cropCenter,
                   const cv::Size& cropSize, const PoseModel poseModel);
    // Flipping
    bool estimateFlip(const MetaData& metaData,
                      const OPTransformationParameter& param_);
    void applyFlip(cv::Mat& imageAugmented, const bool flip, const cv::Mat& image);
    void applyFlip(MetaData& metaData, const bool flip, const int imageWidth,
                   const OPTransformationParameter& param_, const PoseModel poseModel);
    void rotatePoint(cv::Point2f& point2f, const cv::Mat& R);
    // Rotation + scale + cropping + flipping
    void applyAllAugmentation(cv::Mat& imageAugmented, const cv::Mat& rotationMatrix,
                              const float scale, const bool flip, const cv::Point2i& cropCenter,
                              const cv::Size& finalSize, const cv::Mat& image,
                              const unsigned char defaultBorderValue);
    // Other functions
    void keepRoiInside(cv::Rect& roi, const cv::Size& imageSize);
    void clahe(cv::Mat& bgrImage, const int tileSize, const int clipLimit);

}  // namespace caffe

#endif  // USE_OPENCV
#endif  // CAFFE_OPENPOSE_DATA_AUGMENTATION_HPP_
