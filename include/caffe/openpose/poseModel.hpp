#ifndef CAFFE_OPENPOSE_POSE_MODEL_HPP
#define CAFFE_OPENPOSE_POSE_MODEL_HPP

#include <array>
#include <vector>
#include <string>

namespace caffe {

enum class PoseModel : unsigned short
{
    COCO_18 = 0,
    DOME_18 = 1,
    COCO_19 = 2,
    DOME_19 = 3,
    COCO_23 = 4,
    DOME_23_19 = 5,
    COCO_23_17 = 6,
    DOME_23 = 7,
    DOME_59 = 8,
    COCO_59_17 = 9,
    MPII_59 = 10,
    Size,
};
enum class PoseCategory : unsigned short
{
    COCO,
    DOME,
    MPII
};

std::pair<PoseModel,PoseCategory> flagsToPoseModel(const std::string& poseModeString);

int getNumberBodyParts(const PoseModel poseModel);

int getNumberBodyPartsLmdb(const PoseModel poseModel);

int getNumberPafChannels(const PoseModel poseModel);

int getNumberBodyAndPafChannels(const PoseModel poseModel);

int getNumberBodyBkgAndPAF(const PoseModel poseModel);

const std::vector<std::vector<int>>& getLmdbToOpenPoseKeypoints(const PoseModel poseModel);

const std::vector<std::array<int,2>>& getSwapLeftRightKeypoints(const PoseModel poseModel);

const std::vector<int>& getPafIndexA(const PoseModel poseModel);

const std::vector<int>& getPafIndexB(const PoseModel poseModel);

const std::vector<int> getMissingChannels(const PoseModel poseModel, const std::vector<float>& isVisible = {});

}  // namespace caffe

#endif  // CAFFE_OPENPOSE_POSE_MODEL_HPP
