#include <caffe/openpose/poseModel.hpp>
#include <caffe/openpose/getLine.hpp>

namespace caffe {
    std::pair<PoseModel,PoseCategory> flagsToPoseModel(const std::string& poseModeString)
    {
        if (poseModeString == "COCO_18")
            return std::make_pair(PoseModel::COCO_18, PoseCategory::COCO);
        else if (poseModeString == "COCO_19")
            return std::make_pair(PoseModel::COCO_19, PoseCategory::COCO);
        else if (poseModeString == "COCO_23")
            return std::make_pair(PoseModel::COCO_23, PoseCategory::COCO);
        else if (poseModeString == "COCO_23_18")
            return std::make_pair(PoseModel::COCO_23_18, PoseCategory::COCO);
        else if (poseModeString == "DOME_18")
            return std::make_pair(PoseModel::DOME_18, PoseCategory::DOME);
        else if (poseModeString == "DOME_19")
            return std::make_pair(PoseModel::DOME_19, PoseCategory::DOME);
        else if (poseModeString == "DOME_23")
            return std::make_pair(PoseModel::DOME_23, PoseCategory::DOME);
        else if (poseModeString == "DOME_23_19")
            return std::make_pair(PoseModel::DOME_23_19, PoseCategory::DOME);
        else if (poseModeString == "DOME_59")
            return std::make_pair(PoseModel::DOME_59, PoseCategory::DOME);
        // else
        throw std::runtime_error{"String (" + poseModeString
                                 + ") does not correspond to any model (COCO_18, DOME_18, ...)"
                                 + getLine(__LINE__, __FUNCTION__, __FILE__)};
        return std::make_pair(PoseModel::COCO_18, PoseCategory::COCO);
    }

    int getNumberPafChannels(const PoseModel poseModel)
    {
        return 2*(NUMBER_BODY_PARTS[(int)poseModel]+1);
    }

    int getNumberBodyAndPafChannels(const PoseModel poseModel)
    {
        return NUMBER_BODY_PARTS[(int)poseModel] + getNumberPafChannels(poseModel);
    }

    int getNumberBodyBkgAndPAF(const PoseModel poseModel)
    {
        return getNumberBodyAndPafChannels(poseModel) + 1;
    }
}  // namespace caffe
