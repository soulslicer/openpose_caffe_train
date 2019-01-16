#ifndef CAFFE_OPENPOSE_POSE_MODEL_HPP
#define CAFFE_OPENPOSE_POSE_MODEL_HPP

#include <array>
#include <vector>
#include <string>

namespace caffe {

enum class PoseModel : unsigned short
{
    COCO_18 = 0,
    DOME_18,
    COCO_19,
    DOME_19,
    DOME_59,
    COCO_59_17, // 5
    MPII_59,
    COCO_19b,
    COCO_19_V2,
    COCO_25,
    COCO_25_17, // 10
    MPII_65_42,
    CAR_12,
    COCO_25E,
    COCO_25_17E,
    COCO_23,    // 15
    COCO_23_17,
    CAR_22,
    COCO_19E,
    // COCO + MPII + Foot
    COCO_25B_23,
    COCO_25B_17, // 20
    MPII_25B_16,
    // COCO + MPII + Foot + Face
    COCO_95_23,
    COCO_95_17,
    MPII_95_16,
    FACE_95_70,
    // COCO + MPII + Foot + Face + Hand
    COCO_135_23,
    COCO_135_17,
    MPII_135_16,
    HAND_135_21,
    HAND_135_42,
    FACE_135_70,
    Size,
};
enum class PoseCategory : unsigned short
{
    COCO,
    DOME,
    MPII,
    CAR,
    FACE,
    HAND
};

std::pair<PoseModel,PoseCategory> flagsToPoseModel(const std::string& poseModeString);

bool addBkgChannel(const PoseModel poseModel);

int getNumberBodyParts(const PoseModel poseModel);

int getNumberBodyPartsLmdb(const PoseModel poseModel);

int getNumberPafChannels(const PoseModel poseModel);

int getNumberBodyAndPafChannels(const PoseModel poseModel);

int getNumberBodyBkgAndPAF(const PoseModel poseModel);

const std::vector<std::vector<int>>& getLmdbToOpenPoseKeypoints(const PoseModel poseModel);

const std::vector<std::vector<int>>& getMaskedChannels(const PoseModel poseModel);

const std::vector<std::array<int,2>>& getSwapLeftRightKeypoints(const PoseModel poseModel);

const std::vector<int>& getPafIndexA(const PoseModel poseModel);

const std::vector<int>& getPafIndexB(const PoseModel poseModel);

const std::map<unsigned int, std::string>& getMapping(const PoseModel poseModel);

const std::vector<float>& getDistanceAverage(const PoseModel poseModel);

const std::vector<float>& getDistanceSigma(const PoseModel poseModel);

unsigned int getRootIndex();

std::vector<int> getIndexesForParts(const PoseModel poseModel, const std::vector<int>& missingBodyPartsBase,
                                    const std::vector<float>& isVisible, const float minVisibleToBlock = 4.f);

std::vector<int> getEmptyChannels(const PoseModel poseModel, const std::vector<float>& isVisible,
                                  const float minVisibleToBlock = 4.f);

std::vector<int> getMinus1Channels(const PoseModel poseModel, const std::vector<float>& isVisible);

}  // namespace caffe

#endif  // CAFFE_OPENPOSE_POSE_MODEL_HPP
