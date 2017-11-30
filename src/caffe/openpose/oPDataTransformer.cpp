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
// OPENPOSE_DEPTH_BODY_PARTS {
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
// DOME_BODY_PARTS {
//     {0,  "Nose"},
//     {1,  "Neck"},
//     {2,  "RShoulder"},
//     {3,  "RElbow"},
//     {4,  "RWrist"},
//     {5,  "LShoulder"},
//     {6,  "LElbow"},
//     {7,  "LWrist"},
//     {8,  "LowerAbs"},
//     {9,  "RHip"},
//     {10, "RKnee"},
//     {11, "RAnkle"},
//     {12, "LHip"},
//     {13, "LKnee"},
//     {14, "LAnkle"},
//     {15, "REye"},
//     {16, "LEye"},
//     {17, "REar"},
//     {18, "LEar"},
//     {19, "RBigToe"},
//     {20, "RSmallToe"},
//     {21, "LBigToe"},
//     {22, "LSmallToe"},
//     {19/23, "Background"},
// };
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
//     {17-21, "Background"},
//     {17, "LBigToe"},
//     {18, "LSmallToe"},
//     {19, "RBigToe"},
//     {20, "RSmallToe"},
// };
// OPENPOSE_BODY_PARTS_18 {
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
// OPENPOSE_BODY_PARTS_19 {
//     {0,  "Nose"},
//     {1,  "Neck"},
//     {2,  "RShoulder"},
//     {3,  "RElbow"},
//     {4,  "RWrist"},
//     {5,  "LShoulder"},
//     {6,  "LElbow"},
//     {7,  "LWrist"},
//     {8,  "LowerAbs"},
//     {9,  "RHip"},
//     {10, "RKnee"},
//     {11, "RAnkle"},
//     {12, "LHip"},
//     {13, "LKnee"},
//     {14, "LAnkle"},
//     {15, "REye"},
//     {16, "LEye"},
//     {17, "REar"},
//     {18, "LEar"},
//     {19, "Background"},
// };
// OPENPOSE_BODY_PARTS_23 {
//     {0,  "Neck"},
//     {1,  "RShoulder"},
//     {2,  "RElbow"},
//     {3,  "RWrist"},
//     {4,  "LShoulder"},
//     {5,  "LElbow"},
//     {6,  "LWrist"},
//     {7,  "LowerAbs"},
//     {8,  "RHip"},
//     {9,  "RKnee"},
//     {10, "RAnkle"},
//     {11, "RBigToe"},
//     {12, "RSmallToe"},
//     {13, "LHip"},
//     {14, "LKnee"},
//     {15, "LAnkle"},
//     {16, "LBigToe"},
//     {17, "LSmallToe"},
//     {18, "Nose"},
//     {19, "REye"},
//     {20, "REar"},
//     {21, "LEye"},
//     {22, "LEar"},
//     {23, "Background"}
// };
// OPENPOSE_BODY_PARTS_59 {
// // Body
//     {0,  "Nose"},
//     {1,  "Neck"},
//     {2,  "RShoulder"},
//     {3,  "RElbow"},
//     {4,  "RWrist"},
//     {5,  "LShoulder"},
//     {6,  "LElbow"},
//     {7,  "LWrist"},
//     {8,  "LowerAbs"},
//     {9,  "RHip"},
//     {10, "RKnee"},
//     {11, "RAnkle"},
//     {12, "LHip"},
//     {13, "LKnee"},
//     {14, "LAnkle"},
//     {15, "REye"},
//     {16, "LEye"},
//     {17, "REar"},
//     {18, "LEar"},
// // Left hand
//     {19, "LThumb1CMC"},
//     {20, "LThumb2Knuckles"},
//     {21, "LThumb3IP"},
//     {22, "LThumb4FingerTip"},
//     {23, "LIndex1Knuckles"},
//     {24, "LIndex2PIP"},
//     {25, "LIndex3DIP"},
//     {26, "LIndex4FingerTip"},
//     {27, "LMiddle1Knuckles"},
//     {28, "LMiddle2PIP"},
//     {29, "LMiddle3DIP"},
//     {30, "LMiddle4FingerTip"},
//     {31, "LRing1Knuckles"},
//     {32, "LRing2PIP"},
//     {33, "LRing3DIP"},
//     {34, "LRing4FingerTip"},
//     {35, "LPinky1Knuckles"},
//     {36, "LPinky2PIP"},
//     {37, "LPinky3DIP"},
//     {38, "LPinky4FingerTip"},
// // Right hand
//     {39, "RThumb1CMC"},
//     {40, "RThumb2Knuckles"},
//     {41, "RThumb3IP"},
//     {42, "RThumb4FingerTip"},
//     {43, "RIndex1Knuckles"},
//     {44, "RIndex2PIP"},
//     {45, "RIndex3DIP"},
//     {46, "RIndex4FingerTip"},
//     {47, "RMiddle1Knuckles"},
//     {48, "RMiddle2PIP"},
//     {49, "RMiddle3DIP"},
//     {50, "RMiddle4FingerTip"},
//     {51, "RRing1Knuckles"},
//     {52, "RRing2PIP"},
//     {53, "RRing3DIP"},
//     {54, "RRing4FingerTip"},
//     {55, "RPinky1Knuckles"},
//     {56, "RPinky2PIP"},
//     {57, "RPinky3DIP"},
//     {58, "RPinky4FingerTip"},
// // Background
//     {59, "Background"},
// };
// // Hand legend:
// //     - Thumb:
// //         - Carpometacarpal Joints (CMC)
// //         - Interphalangeal Joints (IP)
// //     - Other fingers:
// //         - Knuckles or Metacarpophalangeal Joints (MCP)
// //         - PIP (Proximal Interphalangeal Joints)
// //         - DIP (Distal Interphalangeal Joints)
// //     - All fingers:
// //         - Fingertips
// // More information: Page 6 of http://www.mccc.edu/~behrensb/documents/TheHandbig.pdf
const std::array<int, (int)PoseModel::Size> NUMBER_BODY_PARTS{18, 18, 19, 19, 23, 23, 23, 23, 59};
const std::array<int, (int)PoseModel::Size> NUMBER_PARTS_LMDB{17, 19, 17, 19, 21, 19, 17, 23, 59};
const std::array<int, (int)PoseModel::Size> NUMBER_PAFS{2*(NUMBER_BODY_PARTS[0]+1),
                                                        2*(NUMBER_BODY_PARTS[1]+1),
                                                        2*(NUMBER_BODY_PARTS[2]+1),
                                                        2*(NUMBER_BODY_PARTS[3]+1),
                                                        2*(NUMBER_BODY_PARTS[4]+1),
                                                        2*(NUMBER_BODY_PARTS[5]+1),
                                                        2*(NUMBER_BODY_PARTS[6]+1),
                                                        2*(NUMBER_BODY_PARTS[7]+1),
                                                        2*(NUMBER_BODY_PARTS[8]+1)};
const std::array<int, (int)PoseModel::Size> NUMBER_BODY_AND_PAF_CHANNELS{NUMBER_BODY_PARTS[0]+NUMBER_PAFS[0],
                                                                         NUMBER_BODY_PARTS[1]+NUMBER_PAFS[1],
                                                                         NUMBER_BODY_PARTS[2]+NUMBER_PAFS[2],
                                                                         NUMBER_BODY_PARTS[3]+NUMBER_PAFS[3],
                                                                         NUMBER_BODY_PARTS[4]+NUMBER_PAFS[4],
                                                                         NUMBER_BODY_PARTS[5]+NUMBER_PAFS[5],
                                                                         NUMBER_BODY_PARTS[6]+NUMBER_PAFS[6],
                                                                         NUMBER_BODY_PARTS[7]+NUMBER_PAFS[7],
                                                                         NUMBER_BODY_PARTS[8]+NUMBER_PAFS[8]};
const std::array<std::vector<std::vector<int>>, (int)PoseModel::Size> TRANSFORM_MODEL_TO_OURS{
    std::vector<std::vector<int>>{
        {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}           // COCO_18
    },
    std::vector<std::vector<int>>{
        {0},{1}, {2},{3},{4},  {5},{6},{7},  {9},{10},{11}, {12},{13},{14},  {15},{16},{17},{18}        // DOME_18
    },
    std::vector<std::vector<int>>{
        {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}  // COCO_19
    },
    std::vector<std::vector<int>>{
        {0},{1}, {2},{3},{4},  {5},{6},{7},  {8},  {9},{10},{11},  {12},{13},{14},  {15},{16},{17},{18} // DOME_19
    },
    std::vector<std::vector<int>>{
        {5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16},{19},{20}, {11},{13},{15},{17},{18}, {0},{2},{4},{1},{3}  // COCO_23
    },
    std::vector<std::vector<int>>{
        {1}, {2},{3},{4}, {5},{6},{7}, {8}, {9},{10},{11},{11},{11}, {12},{13},{14},{14},{14}, {0},{15},{17},{16},{18} // DOME_23_19
    },
    std::vector<std::vector<int>>{
        {5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16},{0},{0}, {11},{13},{15},{0},{0}, {0},{2},{4},{1},{3}  // COCO_23_18
    },
    std::vector<std::vector<int>>{
        {1}, {2},{3},{4}, {5},{6},{7}, {8}, {9},{10},{11},{19},{20}, {12},{13},{14},{21},{22}, {0},{15},{17},{16},{18} // DOME_23
    },
    std::vector<std::vector<int>>{                                                                              // DOME_59
        {0},{1}, {2},{3},{4},  {5},{6},{7},  {8},  {9},{10},{11},  {12},{13},{14},  {15},{16},{17},{18},        // Body
        {19},{20},{21},{22}, {23},{24},{25},{26}, {27},{28},{29},{30}, {31},{32},{33},{34}, {35},{36},{37},{38},// Left hand
        {39},{40},{41},{42}, {43},{44},{45},{46}, {47},{48},{49},{50}, {51},{52},{53},{54}, {55},{56},{57},{58} // Right hand
    },
};
const std::array<std::vector<int>, (int)PoseModel::Size> SWAP_LEFTS{
    std::vector<int>{5,6,7,11,12,13,15,17},                                                             // COCO_18
    std::vector<int>{5,6,7,11,12,13,15,17},                                                             // DOME_18
    std::vector<int>{5,6,7,12,13,14,16,18},                                                             // COCO_19
    std::vector<int>{5,6,7,12,13,14,16,18},                                                             // DOME_19
    std::vector<int>{1,2,3, 8,9,10,11,12, 19,20},                                                       // COCO_23
    std::vector<int>{1,2,3, 8,9,10,11,12, 19,20},                                                       // DOME_23_19
    std::vector<int>{1,2,3, 8,9,10,11,12, 19,20},                                                       // COCO_23_18
    std::vector<int>{1,2,3, 8,9,10,11,12, 19,20},                                                       // DOME_23
    std::vector<int>{5,6,7,12,13,14,16,18, 19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38},// DOME_59
};
const std::array<std::vector<int>, (int)PoseModel::Size> SWAP_RIGHTS{
    std::vector<int>{2,3,4, 8,9,10,14,16},                                                              // COCO_18
    std::vector<int>{2,3,4, 8,9,10,14,16},                                                              // DOME_18
    std::vector<int>{2,3,4, 9,10,11,15,17},                                                             // COCO_19
    std::vector<int>{2,3,4, 9,10,11,15,17},                                                             // DOME_19
    std::vector<int>{4,5,6, 13,14,15,16,17, 21,22},                                                     // COCO_23
    std::vector<int>{4,5,6, 13,14,15,16,17, 21,22},                                                     // DOME_23_19
    std::vector<int>{4,5,6, 13,14,15,16,17, 21,22},                                                     // COCO_23_18
    std::vector<int>{4,5,6, 13,14,15,16,17, 21,22},                                                     // DOME_23
    std::vector<int>{2,3,4, 9,10,11,15,17, 39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58},// DOME_59
};
const std::array<std::vector<int>, (int)PoseModel::Size> LABEL_MAP_A{
    std::vector<int>{1, 8,  9, 1,   11, 12, 1, 2, 3,  2, 1, 5, 6, 5,  1, 0,  0,  14, 15},               // COCO_18
    std::vector<int>{1, 8,  9, 1,   11, 12, 1, 2, 3,  2, 1, 5, 6, 5,  1, 0,  0,  14, 15},               // DOME_18
    std::vector<int>{1, 9, 10, 8,8, 12, 13, 1, 2, 3,  2, 1, 5, 6, 5,  1, 0,  0,  15, 16},               // COCO_19
    std::vector<int>{1, 9, 10, 8,8, 12, 13, 1, 2, 3,  2, 1, 5, 6, 5,  1, 0,  0,  15, 16},               // DOME_19
    std::vector<int>{0,0, 1,2, 4,5,  0,7,7,  8,9,10,10, 13,14,15,15,  0,18,18, 19,21,  1,4},            // COCO_23
    std::vector<int>{0,0, 1,2, 4,5,  0,7,7,  8,9,10,10, 13,14,15,15,  0,18,18, 19,21,  1,4},            // DOME_23_19
    std::vector<int>{0,0, 1,2, 4,5,  0,7,7,  8,9,10,10, 13,14,15,15,  0,18,18, 19,21,  1,4},            // COCO_23_18
    std::vector<int>{0,0, 1,2, 4,5,  0,7,7,  8,9,10,10, 13,14,15,15,  0,18,18, 19,21,  1,4},            // DOME_23
    std::vector<int>{1, 9, 10, 8,8, 12, 13, 1, 2, 3,  2, 1, 5, 6, 5,  1, 0,  0,  15, 16,                // DOME_59
                     7,19,20,21, 7,23,24,25, 7,27,28,29, 7,31,32,33, 7,35,36,37, // Left hand
                     4,39,40,41, 4,43,44,45, 4,47,48,49, 4,51,52,53, 4,55,56,57} // Right hand
};
const std::array<std::vector<int>, (int)PoseModel::Size> LABEL_MAP_B{
    std::vector<int>{8, 9, 10, 11,  12, 13, 2, 3, 4, 16, 5, 6, 7, 17, 0, 14, 15, 16, 17},               // COCO_18
    std::vector<int>{8, 9, 10, 11,  12, 13, 2, 3, 4, 16, 5, 6, 7, 17, 0, 14, 15, 16, 17},               // DOME_18
    std::vector<int>{8,10, 11, 9,12,13, 14, 2, 3, 4, 17, 5, 6, 7, 18, 0, 15, 16, 17, 18},               // COCO_19
    std::vector<int>{8,10, 11, 9,12,13, 14, 2, 3, 4, 17, 5, 6, 7, 18, 0, 15, 16, 17, 18},               // DOME_19
    std::vector<int>{1,4, 2,3, 5,6, 7,8,13, 9,10,11,12, 14,15,16,17, 18,19,21, 20,22, 20,22},           // COCO_23
    std::vector<int>{1,4, 2,3, 5,6, 7,8,13, 9,10,11,12, 14,15,16,17, 18,19,21, 20,22, 20,22},           // DOME_23_19
    std::vector<int>{1,4, 2,3, 5,6, 7,8,13, 9,10,11,12, 14,15,16,17, 18,19,21, 20,22, 20,22},           // COCO_23_18
    std::vector<int>{1,4, 2,3, 5,6, 7,8,13, 9,10,11,12, 14,15,16,17, 18,19,21, 20,22, 20,22},           // DOME_23
    std::vector<int>{8,10, 11, 9,12,13, 14, 2, 3, 4, 17, 5, 6, 7, 18, 0, 15, 16, 17, 18,                // DOME_59
                     19,20,21,22, 23,24,25,26, 27,28,29,30, 31,32,33,34, 35,36,37,38, // Left hand
                     39,40,41,42, 43,44,45,46, 47,48,49,50, 51,52,53,54, 55,56,57,58} // Right hand
};
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
    std::tie(mPoseModel, mPoseCategory) = flagsToPoseModel(param_.model());
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
void OPDataTransformer<Dtype>::Transform(Blob<Dtype>* transformedData, Blob<Dtype>* transformedLabel,
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
void OPDataTransformer<Dtype>::generateDataAndLabel(Dtype* transformedData, Dtype* transformedLabel,
                                                    const Datum& datum, const Datum* datumNegative)
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
    if (mPoseCategory == PoseCategory::DOME)
        readMetaData(metaData, data.c_str(), datumWidth);
    else
    {
        readMetaData(metaData, &data[3 * datumArea], datumWidth);
        metaData.depthEnabled = false;
    }
    if (param_.transform_body_joint()) // we expect to transform body joints, and not to transform hand joints
        transformMetaJoints(metaData);
    const auto depthEnabled = metaData.depthEnabled;

    // Read image (LMDB channel 1)
    cv::Mat image;
    if (mPoseCategory == PoseCategory::DOME)
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
        // Image size, not backgroundImage
        maskBackgroundImage = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar{0});
    }

    // Read mask miss (LMDB channel 2)
    cv::Mat maskMiss;
    if (mPoseCategory == PoseCategory::DOME)
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

    // BGR --> Gray --> BGR
    if (param_.gray() == 1)
    {
        cv::cvtColor(image, image, CV_BGR2GRAY);
        cv::cvtColor(image, image, CV_GRAY2BGR);
    }
    VLOG(2) << "  cvtColor and CLAHE: " << timer1.MicroSeconds()*1e-3 << " ms";
    timer1.Start();

    VLOG(2) << "  ReadMeta+MetaJoints: " << timer1.MicroSeconds()*1e-3 << " ms";

    // Data augmentation
    timer1.Start();
    AugmentSelection augmentSelection;
    // Visualize original
    if (param_.visualize())
        writeImageAndKeypoints(image, metaData, augmentSelection);
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
            cv::resize(maskMissAugmented, maskMissAugmented, cv::Size{}, 1./stride, 1./stride, cv::INTER_CUBIC);
        if (depthEnabled)
            cv::resize(depthAugmented, depthAugmented, cv::Size{}, 1./stride, 1./stride, cv::INTER_CUBIC);
    }
    // Visualize final
    if (param_.visualize())
        writeImageAndKeypoints(imageAugmented, metaData, augmentSelection);
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
    if (depthEnabled)
        generateLabelMap(transformedLabel, depthAugmented);
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
            // if (mPoseModel == PoseModel::COCO_23_18)
            {
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
                    sprintf(imagename, "visualize/%s_augment_%04d_label_part_%02d.jpg", param_.model().c_str(),
                            metaData.writeNumber, part);
                    cv::imwrite(imagename, labelMap);
                }
            }
        }
        if (depthEnabled)
        {
            cv::Mat depthMap;
            cv::resize(depthAugmented, depthMap, cv::Size{}, stride, stride, cv::INTER_LINEAR);
            char imagename [100];
            sprintf(imagename, "visualize/%s_augment_%04d_label_part_depth.png", param_.model().c_str(),
                    metaData.writeNumber);
            cv::imwrite(imagename, depthMap);
        }
    }
}

template<typename Dtype>
void OPDataTransformer<Dtype>::generateLabelMap(Dtype* transformedLabel, const cv::Mat& depth) const
{
    const auto gridX = (int)depth.cols;
    const auto gridY = (int)depth.rows;
    const auto channelOffset = gridY * gridX;
    const auto numberBodyPAFParts = NUMBER_BODY_AND_PAF_CHANNELS[(int)mPoseModel];
    // generate depth
    for (auto gY = 0; gY < gridY; gY++)
    {
        const auto yOffset = gY*gridX;
        for (auto gX = 0; gX < gridX; gX++)
        {
            const auto xyOffset = yOffset + gX;

            auto depth_val = depth.at<uint16_t>(gY, gX);

            transformedLabel[(2*numberBodyPAFParts+2)*channelOffset + xyOffset] = (depth_val>0)?1.0:0.0;
            transformedLabel[(2*numberBodyPAFParts+3)*channelOffset + xyOffset] = float(depth_val)/1000.0;
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

void maskFeet(cv::Mat& maskMiss, const std::vector<float>& isVisible, const std::vector<cv::Point2f>& points,
              const float stride, const float ratio)
{
    for (auto part = 0 ; part < 2 ; part++)
    {
        const auto kneeIndex = 9+part*5;
        const auto ankleIndex = kneeIndex+1;
        if (isVisible.at(kneeIndex) != 2 && isVisible.at(ankleIndex) != 2)
        {
            const auto knee = points.at(kneeIndex) * (1.f / stride);
            const auto ankle = points.at(ankleIndex) * (1.f / stride);
            const int distance = (int)std::round(ratio*std::sqrt((knee.x - ankle.x)*(knee.x - ankle.x)
                                                                 + (knee.y - ankle.y)*(knee.y - ankle.y)));
            const cv::Point momentum = (ankle-knee)*0.15f;
            cv::Rect roi{(int)std::round(ankle.x + momentum.x)-distance,
                         (int)std::round(ankle.y + momentum.y)-distance,
                         2*distance, 2*distance};
            // Apply ROI
            keepRoiInside(roi, maskMiss.size());
            if (roi.area() > 0)
                maskMiss(roi).setTo(0.f); // For debugging use 0.5f
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
    const auto numberBodyPAFParts = NUMBER_BODY_AND_PAF_CHANNELS[(int)mPoseModel];
    const auto numberBodyBkgPAFParts = getNumberBodyBkgAndPAF();
    const auto numberPAFChannels = NUMBER_PAFS[(int)mPoseModel];
    const auto numberBodyParts = NUMBER_BODY_PARTS[(int)mPoseModel];

    // Labels to 0
    std::fill(transformedLabel, transformedLabel + 2*numberBodyBkgPAFParts * gridY * gridX, 0.f);

    // Initialize labels to 0 or 1 (depending on maskMiss)
    // Label size = image size / stride
    for (auto gY = 0; gY < gridY; gY++)
    {
        const auto yOffset = gY*gridX;
        for (auto gX = 0; gX < gridX; gX++)
        {
            const auto xyOffset = yOffset + gX;
            const float weight = float(maskMiss.at<uchar>(gY, gX)) / 255.f;
            // Body part & PAFs
            for (auto part = 0; part < numberBodyPAFParts; part++)
                transformedLabel[part*channelOffset + xyOffset] = weight;
            // Background channel
            transformedLabel[numberBodyPAFParts*channelOffset + xyOffset] = weight;
        }
    }

    // Remove if required RBigToe, RSmallToe, LBigToe, LSmallToe, and Background
// TODO: Remove, temporary hack to get foot data
    if (mPoseModel == PoseModel::COCO_23 || mPoseModel == PoseModel::DOME_23_19 || mPoseModel == PoseModel::COCO_23_18)
    {
        std::vector<int> indexesToRemove;
        // PAFs
        for (auto index : {11, 12, 15, 16})
        {
            const auto indexBase = 2*index;
            indexesToRemove.emplace_back(indexBase);
            indexesToRemove.emplace_back(indexBase+1);
        }
        // Body parts
        for (auto index : {11, 12, 16, 17})
        {
            const auto indexBase = numberPAFChannels + index;
            indexesToRemove.emplace_back(indexBase);
        }
        // Dome data: Exclude (unlabeled) foot keypoints
        if (mPoseModel == PoseModel::DOME_23_19 || mPoseModel == PoseModel::COCO_23_18)
        {
            // Remove those channels
            for (auto index : indexesToRemove)
            {
                std::fill(&transformedLabel[index*channelOffset],
                          &transformedLabel[index*channelOffset + channelOffset], 0);
            }
        }
        // Background
        if (mPoseModel == PoseModel::DOME_23_19 || mPoseModel == PoseModel::COCO_23_18)
        {
            const auto backgroundIndex = numberPAFChannels + NUMBER_BODY_PARTS[(int)mPoseModel];
            int type;
            if (sizeof(Dtype) == sizeof(float))
                type = CV_32F;
            else if (sizeof(Dtype) == sizeof(double))
                type = CV_64F;
            else
                throw std::runtime_error{"Only float or double"
                                         + getLine(__LINE__, __FUNCTION__, __FILE__)};
            cv::Mat maskMiss(gridY, gridX, type, &transformedLabel[backgroundIndex*channelOffset]);
            maskFeet(maskMiss, metaData.jointsSelf.isVisible, metaData.jointsSelf.points, stride, 0.6f);
            for (const auto& jointsOther : metaData.jointsOthers)
                maskFeet(maskMiss, jointsOther.isVisible, jointsOther.points, stride, 0.6f);
        }
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
                            int type;
                            if (sizeof(Dtype) == sizeof(float))
                                type = CV_32F;
                            else if (sizeof(Dtype) == sizeof(double))
                                type = CV_64F;
                            else
                                throw std::runtime_error{"Only float or double"
                                                         + getLine(__LINE__, __FUNCTION__, __FILE__)};
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
            //     for (auto part = 0; part < 2*numberBodyBkgPAFParts; part++)
            //     {
            //         // Reduce #images saved (ideally images from 0 to numberBodyBkgPAFParts should be the same)
            //         // if (part >= 11*2)
            //         if (part >= 22 && part <= numberBodyBkgPAFParts)
            //         // if (part < 3 || part >= numberBodyBkgPAFParts - 3)
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
    const auto numberPAFChannelsP1 = NUMBER_PAFS[(int)mPoseModel]+1;
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
            // putVectorMaps
            putVectorMaps(transformedLabel + (numberBodyBkgPAFParts + 2*i)*channelOffset,
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
                //putVectorMaps
                putVectorMaps(transformedLabel + (numberBodyBkgPAFParts + 2*i)*channelOffset,
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
            putGaussianMaps(transformedLabel + (part+numberBodyPAFParts+numberPAFChannelsP1)*channelOffset,
                            centerPoint, param_.stride(), gridX, gridY, param_.sigma()); //self
        }
        //for every other person
        for (auto otherPerson = 0; otherPerson < metaData.numberOtherPeople; otherPerson++)
        {
            if (metaData.jointsOthers[otherPerson].isVisible[part] <= 1)
            {
                const auto& centerPoint = metaData.jointsOthers[otherPerson].points[part];
                putGaussianMaps(transformedLabel + (part+numberBodyPAFParts+numberPAFChannelsP1)*channelOffset,
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
            for (auto p = numberBodyPAFParts+numberPAFChannelsP1 ; p < numberBodyPAFParts+numberBodyBkgPAFParts ; p++)
            {
                const auto index = p * channelOffset + xyOffset;
                maximum = (maximum > transformedLabel[index]) ? maximum : transformedLabel[index];
            }
            transformedLabel[(2*numberBodyPAFParts+1)*channelOffset + xyOffset] = std::max(Dtype(1.)-maximum,
                                                                                           Dtype(0.));
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
void OPDataTransformer<Dtype>::writeImageAndKeypoints(const cv::Mat& image, const MetaData& metaData,
                                                      const AugmentSelection& augmentSelection) const
{
    cv::Mat imageToVisualize = image.clone();

    cv::rectangle(imageToVisualize, metaData.objpos-cv::Point2f{3.f,3.f}, metaData.objpos+cv::Point2f{3.f,3.f},
                  cv::Scalar{255,255,0}, CV_FILLED);
    const auto numberBodyPAFParts = NUMBER_BODY_AND_PAF_CHANNELS[(int)mPoseModel];
    for (auto part = 0 ; part < numberBodyPAFParts ; part++)
    {
        const auto currentPoint = metaData.jointsSelf.points[part];
        // Hand case
        if (numberBodyPAFParts == 21)
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
        else if (numberBodyPAFParts == 9)
        {
            if (part==0 || part==1 || part==2 || part==6)
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{0,0,255}, -1);
            else if (part==3 || part==4 || part==5 || part==7)
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,0,0}, -1);
            else
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,255,0}, -1);
        }
        // Body case (CPM)
        else if (numberBodyPAFParts == 14 || numberBodyPAFParts == 28)
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
        for (auto part = 0 ; part < numberBodyPAFParts ; part++)
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

        sprintf(imagename, "visualize/augment_%04d_epoch_%03d_writenum_%03d.jpg", sVisualizationCounter.load(),
                metaData.epoch, metaData.writeNumber);
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
    const auto numberBodyPAFParts = NUMBER_BODY_AND_PAF_CHANNELS[(int)mPoseModel];
    for (auto part = 0; part < numberBodyPAFParts ; part++)
        metaData.jointsSelf.points[part] *= scale;
    for (auto person=0; person<metaData.numberOtherPeople; person++)
    {
        metaData.objPosOthers[person] *= scale;
        for (auto part = 0; part < numberBodyPAFParts ; part++)
            metaData.jointsOthers[person].points[part] *= scale;
    }
}

template<typename Dtype>
std::pair<cv::Mat, cv::Size> OPDataTransformer<Dtype>::estimateRotation(const MetaData& metaData,
                                                                        const cv::Size& imageSize) const
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
void OPDataTransformer<Dtype>::applyRotation(cv::Mat& imageAugmented, const std::pair<cv::Mat,
                                             cv::Size> RotAndFinalSize, const cv::Mat& image,
                                             const unsigned char defaultBorderValue) const
{
    // Rotate image
    if (!image.empty())
        cv::warpAffine(image, imageAugmented, RotAndFinalSize.first, RotAndFinalSize.second, cv::INTER_CUBIC,
                       cv::BORDER_CONSTANT, cv::Scalar{(double)defaultBorderValue});
}

template<typename Dtype>
void OPDataTransformer<Dtype>::applyRotation(MetaData& metaData, const cv::Mat& Rot) const
{
    // Update metaData
    rotatePoint(metaData.objpos, Rot);
    const auto numberBodyPAFParts = NUMBER_BODY_AND_PAF_CHANNELS[(int)mPoseModel];
    for (auto part = 0 ; part < numberBodyPAFParts ; part++)
        rotatePoint(metaData.jointsSelf.points[part], Rot);
    for (auto person = 0; person < metaData.numberOtherPeople; person++)
    {
        rotatePoint(metaData.objPosOthers[person], Rot);
        for (auto part = 0; part < numberBodyPAFParts ; part++)
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
            throw std::runtime_error{"Input and output images must be different"
                                     + getLine(__LINE__, __FUNCTION__, __FILE__)};
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
    const auto numberBodyPAFParts = NUMBER_BODY_AND_PAF_CHANNELS[(int)mPoseModel];
    for (auto part = 0 ; part < numberBodyPAFParts ; part++)
        metaData.jointsSelf.points[part] += offsetPoint;
    for (auto person = 0 ; person < metaData.numberOtherPeople ; person++)
    {
        metaData.objPosOthers[person] += offsetPoint;
        for (auto part = 0 ; part < numberBodyPAFParts ; part++)
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
        const auto numberBodyPAFParts = NUMBER_BODY_AND_PAF_CHANNELS[(int)mPoseModel];
        for (auto part = 0 ; part < numberBodyPAFParts ; part++)
            metaData.jointsSelf.points[part].x = imageWidth - 1 - metaData.jointsSelf.points[part].x;
        if (param_.transform_body_joint())
            swapLeftRight(metaData.jointsSelf);
        for (auto p = 0 ; p < metaData.numberOtherPeople ; p++)
        {
            metaData.objPosOthers[p].x = imageWidth - 1 - metaData.objPosOthers[p].x;
            for (auto part = 0 ; part < numberBodyPAFParts ; part++)
                metaData.jointsOthers[p].points[part].x = imageWidth - 1 - metaData.jointsOthers[p].points[part].x;
            if (param_.transform_body_joint())
                swapLeftRight(metaData.jointsOthers[p]);
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
    metaData.numberOtherPeople = (int)data[2*offsetPerLine];
    metaData.peopleIndex = (int)data[2*offsetPerLine+1];
    metaData.annotationListIndex = (int)(decodeNumber<Dtype>(&data[2*offsetPerLine+2]));
    metaData.writeNumber = (int)(decodeNumber<Dtype>(&data[2*offsetPerLine+6]));
    metaData.totalWriteNumber = (int)(decodeNumber<Dtype>(&data[2*offsetPerLine+10]));

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
    // Matlab (1-index) to C++ (0-index) --> (0,0 goes to -1,-1)
    if (mPoseCategory == PoseCategory::COCO)
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
        if (mPoseCategory == PoseCategory::COCO)
            jointPoint -= cv::Point2f{1.f,1.f};
        // isVisible flag
        const auto isVisible = decodeNumber<Dtype>(&data[7*offsetPerLine+4*part]);
        CHECK_LE(isVisible, 2); // isVisible in range [0, 2]
        jointSelf.isVisible[part] = std::round(isVisible);
        if (jointSelf.isVisible[part] != 2)
        {
            if (jointPoint.x < 0 || jointPoint.y < 0
                || jointPoint.x >= metaData.imageSize.width || jointPoint.y >= metaData.imageSize.height)
            {
                jointSelf.isVisible[part] = 2; // 2 means cropped/unlabeled, 0 means occluded but in image
            }
        }
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
        if (mPoseCategory == PoseCategory::COCO)
            metaData.objPosOthers[person] -= cv::Point2f{1.f,1.f};
        metaData.scaleOthers[person]  = decodeNumber<Dtype>(&data[(8+metaData.numberOtherPeople)
                                                            *offsetPerLine+4*person]);
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
            jointPoint.x = decodeNumber<Dtype>(&data[(9+metaData.numberOtherPeople+3*person)
                                               *offsetPerLine+4*part]);
            jointPoint.y = decodeNumber<Dtype>(&data[(9+metaData.numberOtherPeople+3*person+1)
                                               *offsetPerLine+4*part]);
            // Matlab (1-index) to C++ (0-index) --> (0,0 goes to -1,-1)
            if (mPoseCategory == PoseCategory::COCO)
                jointPoint -= cv::Point2f{1.f,1.f};
            // isVisible flag
            const auto isVisible = decodeNumber<Dtype>(&data[(9+metaData.numberOtherPeople+3*person+2)
                                                       *offsetPerLine+4*part]);
            currentPerson.isVisible[part] = std::round(isVisible);
            if (currentPerson.isVisible[part] != 2)
            {
                if (jointPoint.x < 0 || jointPoint.y < 0
                    || jointPoint.x >= metaData.imageSize.width || jointPoint.y >= metaData.imageSize.height)
                {
                    currentPerson.isVisible[part] = 2; // 2 means cropped/unlabeled, 0 means occluded  but in image
                }
            }
        }
    }
    if (mPoseCategory == PoseCategory::DOME)
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
    const auto numberBodyPAFParts = NUMBER_BODY_AND_PAF_CHANNELS[(int)mPoseModel];
    joints.points.resize(numberBodyPAFParts);
    joints.isVisible.resize(numberBodyPAFParts);

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
{
    const auto centerAAux = 0.125f * centerA;
    const auto centerBAux = 0.125f * centerB;
    const int minX = std::max( int(round(std::min(centerAAux.x, centerBAux.x) - threshold)), 0);
    const int maxX = std::min( int(round(std::max(centerAAux.x, centerBAux.x) + threshold)), gridX);

    const int minY = std::max( int(round(std::min(centerAAux.y, centerBAux.y) - threshold)), 0);
    const int maxY = std::min( int(round(std::max(centerAAux.y, centerBAux.y) + threshold)), gridY);

    // const cv::Point2f bc = (centerBAux - centerAAux) * (1.f / std::sqrt(bc.x*bc.x + bc.y*bc.y));
    cv::Point2f bc = centerBAux - centerAAux;
    bc *= (1.f / std::sqrt(bc.x*bc.x + bc.y*bc.y));
    // If PAF is not 0 or NaN (e.g. if PAF perpendicular to image plane)
    if (!isnan(bc.x) && !isnan(bc.y))
    {
        // const float x_p = (centerAAux.x + centerBAux.x) / 2;
        // const float y_p = (centerAAux.y + centerBAux.y) / 2;
        // const float angle = atan2f(centerBAux.y - centerAAux.y, centerBAux.x - centerAAux.x);
        // const float sine = sinf(angle);
        // const float cosine = cosf(angle);
        // const float a_sqrt = (centerAAux.x - x_p) * (centerAAux.x - x_p)
        //                    + (centerAAux.y - y_p) * (centerAAux.y - y_p);
        // const float b_sqrt = 10; //fixed

        for (auto gY = minY; gY < maxY; gY++)
        {
            const auto yOffset = gY*gridX;
            for (auto gX = minX; gX < maxX; gX++)
            {
                const auto xyOffset = yOffset + gX;
                const cv::Point2f ba{gX - centerAAux.x, gY - centerAAux.y};
                const float distance = std::abs(ba.x*bc.y - ba.y*bc.x);
                if (distance <= threshold)
                {
                    auto& counter = count.at<uchar>(gY, gX);
                    if (counter == 0)
                    {
                        entryX[xyOffset] = bc.x;
                        entryY[xyOffset] = bc.y;
                    }
                    else
                    {
                        entryX[xyOffset] = (entryX[xyOffset]*counter + bc.x) / (counter + 1);
                        entryY[xyOffset] = (entryY[xyOffset]*counter + bc.y) / (counter + 1);
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
