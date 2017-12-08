#ifndef CAFFE_OPENPOSE_DATA_AUGMENTATION_HPP
#define CAFFE_OPENPOSE_DATA_AUGMENTATION_HPP
#ifdef USE_OPENCV

#include <vector>
#include <opencv2/core/core.hpp> // cv::Mat, cv::Point, cv::Size
#include "caffe/proto/caffe.pb.h"
#include "metaData.hpp"
#include "poseModel.hpp"

namespace caffe {
    class DataAugmentation {
    public:
        // Scale
        float estimateScale(const MetaData& metaData, const OPTransformationParameter& param_) const;
        void applyScale(cv::Mat& imageAugmented, const float scale, const cv::Mat& image) const;
        void applyScale(MetaData& metaData, const float scale, const PoseModel poseModel) const;
        // Rotation
        std::pair<cv::Mat, cv::Size> estimateRotation(const MetaData& metaData, const cv::Size& imageSize,
                                                      const OPTransformationParameter& param_) const;
        void applyRotation(cv::Mat& imageAugmented, const std::pair<cv::Mat, cv::Size> RotAndFinalSize,
                           const cv::Mat& image, const unsigned char defaultBorderValue) const;
        void applyRotation(MetaData& metaData, const cv::Mat& Rot, const PoseModel poseModel) const;
        // Cropping
        cv::Point2i estimateCrop(const MetaData& metaData, const OPTransformationParameter& param_) const;
        void applyCrop(cv::Mat& imageAugmented, const cv::Point2i& cropCenter, const cv::Mat& image,
                       const unsigned char defaultBorderValue, const OPTransformationParameter& param_) const;
        void applyCrop(MetaData& metaData, const cv::Point2i& cropCenter,
                       const OPTransformationParameter& param_, const PoseModel poseModel) const;
        // Flipping
        bool estimateFlip(const MetaData& metaData,
                          const OPTransformationParameter& param_) const;
        void applyFlip(cv::Mat& imageAugmented, const bool flip, const cv::Mat& image) const;
        void applyFlip(MetaData& metaData, const bool flip, const int imageWidth,
                       const OPTransformationParameter& param_, const PoseModel poseModel) const;
        void rotatePoint(cv::Point2f& point2f, const cv::Mat& R) const;
        // Other functions
        void clahe(cv::Mat& bgrImage, const int tileSize, const int clipLimit) const;

    private:
        std::vector<std::vector<float>> mAugmentationDegs;
        std::vector<std::vector<int>> mAugmentationFlips;
    };

}  // namespace caffe

#endif  // USE_OPENCV
#endif  // CAFFE_OPENPOSE_DATA_AUGMENTATION_HPP_
