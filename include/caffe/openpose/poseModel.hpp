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
    COCO_23_18 = 6,
    DOME_23 = 7,
    DOME_59 = 8,
    Size,
};
enum class PoseCategory : bool
{
    COCO,
    DOME
};

// General information:
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

std::pair<PoseModel,PoseCategory> flagsToPoseModel(const std::string& poseModeString);

const std::array<int, (int)PoseModel::Size> NUMBER_BODY_PARTS{18, 18, 19, 19, 23, 23, 23, 23, 59};

const std::array<int, (int)PoseModel::Size> NUMBER_PARTS_LMDB{17, 19, 17, 19, 21, 19, 17, 23, 59};

int getNumberPafChannels(const PoseModel poseModel);

int getNumberBodyAndPafChannels(const PoseModel poseModel);

int getNumberBodyBkgAndPAF(const PoseModel poseModel);

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

}  // namespace caffe

#endif  // CAFFE_OPENPOSE_POSE_MODEL_HPP
