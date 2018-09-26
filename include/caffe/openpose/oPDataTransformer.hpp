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
#include "dataAugmentation.hpp"
#include "metaData.hpp"
#include "poseModel.hpp"
// OpenPose: added end
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/value.h>
#include <jsoncpp/json/reader.h>
#include <unistd.h>

namespace caffe {

#include  <random>
#include  <iterator>

template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}

template<typename Iter>
Iter select_randomly(Iter start, Iter end) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return select_randomly(start, end, gen);
}

template <class T>
bool vec_contains(std::vector<T> const &v, T const &x) {
    if (v.empty())
         return false;
    if (find(v.begin(), v.end(), x) != v.end())
         return true;
    else
         return false;
}

int getRand(int min, int max);

struct VSeq{
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> imagesAug;
    std::vector<cv::Mat> masks;
    std::vector<Json::Value> jsons;
};

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class OPDataTransformer {
public:
    explicit OPDataTransformer(const std::string& modelString);
    explicit OPDataTransformer(const OPTransformationParameter& param);
    explicit OPDataTransformer(const OPTransformationParameter& param, Phase phase,
                               const std::string& modelString, int tpaf = false, int staf = false, std::vector<int> stafIDS = {}); // OpenPose: Added std::string
    virtual ~OPDataTransformer() {}

    /**
     * @brief Initialize the Random number generations if needed by the
     *    transformation.
     */
    // void InitRand();

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
    // virtual int Rand(int n);

    // void Transform(const Datum& datum, Dtype* transformedData); // OpenPose: commented
    // OpenPose: added
    // Image and label
public:
    void Transform(Blob<Dtype>* transformedData, Blob<Dtype>* transformedLabel, const Datum& datum,
                   const Datum* datumNegative = nullptr,
                   // Extra labels addition
                   Blob<Dtype> extra_transformed_labels[] = nullptr,
                   std::vector<int> extra_strides = std::vector<int>(0),
                   int extra_labels_count = 0);
    void TransformVideoJSON(int vid, int frames, VSeq& vs, Blob<Dtype>* transformedData, Blob<Dtype>* transformedLabel, const Datum& datum,
                   const Datum* datumNegative = nullptr);
    void TransformVideoSF(int vid, int frames, VSeq& vs, Blob<Dtype>* transformedData, Blob<Dtype>* transformedLabel, const Datum& datum,
                   const Datum* datumNegative = nullptr, bool motion = true);
    void Test(int frames, Blob<Dtype>* transformedData, Blob<Dtype>* transformedLabel);
    int getNumberChannels() const;
    int getTotalTaf() const;

    cv::Mat parseBackground(const Datum* background);

    cv::Mat opConvert(const cv::Mat& img, const cv::Mat& bg, std::vector<cv::Rect>& rects);


protected:
    // OpenPose: added end
    // Tranformation parameters
    // TransformationParameter param_; // OpenPose: commented
    OPTransformationParameter param_; // OpenPose: added

    // shared_ptr<Caffe::RNG> rng_; // OpenPose: commented
    Phase phase_;
    // Blob<Dtype> data_mean_; // OpenPose: commented
    // vector<Dtype> mean_values_; // OpenPose: commented

    // OpenPose: added
protected:
    PoseModel mPoseModel;
    PoseCategory mPoseCategory;
    int mCurrentEpoch;
    std::string mModelString;
    int mTpaf, mStaf;
    std::vector<int> mStafIDS;

    // Label generation
    void generateDataAndLabel(Dtype* transformedData, Dtype* transformedLabel, const Datum& datum,
                              const Datum* datumNegative,
                              // Extra labels addition
                              Blob<Dtype> extra_transformed_labels[] = nullptr,
                              std::vector<int> extra_strides = std::vector<int>(0),
                              int extra_labels_count = 0);
    void generateDepthLabelMap(Dtype* transformedLabel, const cv::Mat& depth) const;
    void generateLabelMap(Dtype* transformedLabel, const cv::Size& imageSize, const cv::Mat& maskMiss,
                          const MetaData& metaData, const cv::Mat& img, const int stride) const;
    void generateLabelMapStaf(Dtype* transformedLabel, const cv::Size& imageSize, const cv::Mat& maskMiss,
                          const MetaData& metaData, const cv::Mat& img, const int stride) const;
    void generateLabelMapStafWithPaf(Dtype* transformedLabel, const cv::Size& imageSize, const cv::Mat& maskMiss,
                          const MetaData& metaData, const cv::Mat& img, const int stride) const;
    void generateLabelMapStafNew(Dtype* transformedLabel, const cv::Size& imageSize, const cv::Mat& maskMiss,
                          const MetaData& metaData, const cv::Mat& img, const int stride, bool tracking=false) const;
    void generateLabelMapStafWithPafAndTaf(Dtype* transformedLabel, const cv::Size& imageSize, const cv::Mat& maskMiss,
                          const MetaData& metaData, const cv::Mat& img, const int stride) const;
    void putGaussianMaps(Dtype* entry, const cv::Point2f& center, const int stride, const int gridX, const int gridY,
                         const float sigma) const;
    void putVectorMaps(Dtype* entryX, Dtype* entryY, Dtype* maskX, Dtype* maskY, cv::Mat& count,
                       const cv::Point2f& centerA, const cv::Point2f& centerB, const int stride,
                       const int gridX, const int gridY, const float sigma, const int threshold,
                       const int diagonal, const float diagonalProportion, const bool normalize = true, const bool demask = false, const float tanval = 0) const;

    // // For Distance
    // void putVectorMaps(Dtype* entryX, Dtype* entryY, Dtype* entryD, Dtype* entryDMask, cv::Mat& count,
    //                    const cv::Point2f& centerA, const cv::Point2f& centerB, const int stride, const int gridX,
    //                    const int gridY, const float sigma, const int thre) const;
    // OpenPose: added end
};

}  // namespace caffe

#endif  // CAFFE_OPENPOSE_OP_DATA_TRANSFORMER_HPP
