#ifndef CAFFE_OPENPOSE_OP_DATA_TRANSFORMER_HPP
#define CAFFE_OPENPOSE_OP_DATA_TRANSFORMER_HPP

// OpenPose: added
// This function has been originally copied from include/caffe/data_transformer.hpp (both hpp and cpp) at Sep 7th, 2017
// OpenPose: added end

#include <vector>
// OpenPose: added
#ifdef USE_OPENCV
    #include <opencv2/core/core.hpp> // cv::Mat, cv::Point, cv::Size
#endif  // USE_OPENCV
// OpenPose: added end
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// OpenPose: added
enum class PoseModel : unsigned short
{
    COCO_18 = 0,
    DOME_18 = 1,
    COCO_19 = 2,
    DOME_19 = 3,
    COCO_23 = 4,
    DOME_23_19 = 5,
    COCO_23_18 = 6,
    DOME_23 = 7,
    Size,
};
enum class PoseCategory : bool
{
    COCO,
    DOME
};
// OpenPose: added end

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class OPDataTransformer {
public:
    explicit OPDataTransformer(const OPTransformationParameter& param, Phase phase);
    virtual ~OPDataTransformer() {}

    /**
     * @brief Initialize the Random number generations if needed by the
     *    transformation.
     */
    void InitRand();

#ifdef USE_OPENCV
    /**
     * @brief Applies the transformation defined in the data layer's
     * transform_param block to a cv::Mat
     *
     * @param cv_img
     *    cv::Mat containing the data to be transformed.
     * @param transformed_blob
     *    This is destination blob. It can be part of top blob's data if
     *    set_cpu_data() is used. See image_data_layer.cpp for an example.
     */
    // void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob); // OpenPose: commented
#endif  // USE_OPENCV

protected:
     /**
     * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
     *
     * @param n
     *    The upperbound (exclusive) value of the random number.
     * @return
     *    A uniformly random integer value from ({0, 1, ..., n-1}).
     */
    virtual int Rand(int n);

    // void Transform(const Datum& datum, Dtype* transformedData); // OpenPose: commented
    // OpenPose: added
    // Image and label
public:
    void Transform(Blob<Dtype>* transformedData, Blob<Dtype>* transformedLabel, const Datum& datum,
                   const Datum* datumNegative = nullptr);
    int getNumberBodyBkgAndPAF() const;
    int getNumberChannels() const;
protected:
    // OpenPose: added end
    // Tranformation parameters
    // TransformationParameter param_; // OpenPose: commented
    OPTransformationParameter param_; // OpenPose: added


    shared_ptr<Caffe::RNG> rng_;
    Phase phase_;
    Blob<Dtype> data_mean_;
    vector<Dtype> mean_values_;

    // OpenPose: added
protected:
    struct AugmentSelection
    {
        bool flip = false;
        std::pair<cv::Mat, cv::Size> RotAndFinalSize;
        cv::Point2i cropCenter;
        float scale = 1.f;
    };

    struct Joints
    {
        std::vector<cv::Point2f> points;
        std::vector<float> isVisible;
    };

    struct MetaData
    {
        cv::Size imageSize;
        bool isValidation; // Just to check it is false
        int numberOtherPeople;
        int writeNumber;
        int totalWriteNumber;
        int epoch;
        cv::Point2f objpos; //objpos_x(float), objpos_y (float)
        float scaleSelf;
        Joints jointsSelf; //(3*16)
        std::vector<cv::Point2f> objPosOthers; //length is numberOtherPeople
        std::vector<float> scaleOthers; //length is numberOtherPeople
        std::vector<Joints> jointsOthers; //length is numberOtherPeople
        // Only for DomeDB
        std::string imageSource;
        bool depthEnabled = false;
        std::string depthSource;
        // Only for visualization
        std::string datasetString;
        int peopleIndex;
        int annotationListIndex;
    };

    PoseModel mPoseModel;
    PoseCategory mPoseCategory;
    bool mIsTableSet;
    std::vector<std::vector<float>> mAugmentationDegs;
    std::vector<std::vector<int>> mAugmentationFlips;

    void generateDataAndLabel(Dtype* transformedData, Dtype* transformedLabel, const Datum& datum,
                              const Datum* datumNegative);
    void generateLabelMap(Dtype* transformedLabel, const cv::Mat& image, const cv::Mat& maskMiss,
                          const MetaData& metaData) const;
    void generateLabelMap(Dtype* transformedLabel, const cv::Mat& depth) const;
    void writeImageAndKeypoints(const cv::Mat& image, const MetaData& metaData,
                                const AugmentSelection& augmentSelection) const;
    // Scale
    float estimateScale(const MetaData& metaData) const;
    void applyScale(cv::Mat& imageAugmented, const float scale, const cv::Mat& image) const;
    void applyScale(MetaData& metaData, const float scale) const;
    // Rotation
    std::pair<cv::Mat, cv::Size> estimateRotation(const MetaData& metaData, const cv::Size& imageSize) const;
    void applyRotation(cv::Mat& imageAugmented, const std::pair<cv::Mat, cv::Size> RotAndFinalSize,
                       const cv::Mat& image, const unsigned char defaultBorderValue) const;
    void applyRotation(MetaData& metaData, const cv::Mat& Rot) const;
    // Cropping
    cv::Point2i estimateCrop(const MetaData& metaData) const;
    void applyCrop(cv::Mat& imageAugmented, const cv::Point2i& cropCenter, const cv::Mat& image,
                   const unsigned char defaultBorderValue) const;
    void applyCrop(MetaData& metaData, const cv::Point2i& cropCenter) const;
    // Flipping
    bool estimateFlip(const MetaData& metaData) const;
    void applyFlip(cv::Mat& imageAugmented, const bool flip, const cv::Mat& image) const;
    void applyFlip(MetaData& metaData, const bool flip, const int imageWidth) const;
    void rotatePoint(cv::Point2f& point2f, const cv::Mat& R) const;
    void swapLeftRight(Joints& joints) const;
    void setAugmentationTable(const int numData);
    void readMetaData(MetaData& metaData, const char* data, const size_t offsetPerLine);
    void transformMetaJoints(MetaData& metaData) const;
    void transformJoints(Joints& joints) const;
    void clahe(cv::Mat& bgrImage, const int tileSize, const int clipLimit) const;
    void putGaussianMaps(Dtype* entry, const cv::Point2f& center, const int stride, const int gridX, const int gridY,
                         const float sigma) const;
    void putVecMaps(Dtype* entryX, Dtype* entryY, cv::Mat& count, const cv::Point2f& centerA,
                    const cv::Point2f& centerB, const int stride, const int gridX, const int gridY, const float sigma,
                    const int thre) const;
    // OpenPose: added end
};

}  // namespace caffe

#endif  // CAFFE_OPENPOSE_OP_DATA_TRANSFORMER_HPP_
