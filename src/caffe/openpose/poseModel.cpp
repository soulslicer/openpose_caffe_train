#include <algorithm>    // std::sort, std::unique, std::distance
#include <iostream>
#include <caffe/openpose/poseModel.hpp>
#include <caffe/openpose/getLine.hpp>

namespace caffe {
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
// OPENPOSE_BODY_PARTS_19(b) OPENPOSE_BODY_PARTS_25 {
//     {0,  "Nose"},
//     {1,  "Neck"},
//     {2,  "RShoulder"},
//     {3,  "RElbow"},
//     {4,  "RWrist"},
//     {5,  "LShoulder"},
//     {6,  "LElbow"},
//     {7,  "LWrist"},
//     {8,  "MHip"},
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
//     {19, "LBigToe"},
//     {20, "LSmallToe"},
//     {21, "LHeel"},
//     {22, "RBigToe"},
//     {23, "RSmallToe"},
//     {24, "RHeel"},
//     {19/25, "Background"},
// };
// OPENPOSE_BODY_PARTS_59 {
//     // Body
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
//     // Left hand
//     {19, "LThumb1CMC"},         {20, "LThumb2Knuckles"},{21, "LThumb3IP"},  {22, "LThumb4FingerTip"},
//     {23, "LIndex1Knuckles"},    {24, "LIndex2PIP"},     {25, "LIndex3DIP"}, {26, "LIndex4FingerTip"},
//     {27, "LMiddle1Knuckles"},   {28, "LMiddle2PIP"},    {29, "LMiddle3DIP"},{30, "LMiddle4FingerTip"},
//     {31, "LRing1Knuckles"},     {32, "LRing2PIP"},      {33, "LRing3DIP"},  {34, "LRing4FingerTip"},
//     {35, "LPinky1Knuckles"},    {36, "LPinky2PIP"},     {37, "LPinky3DIP"}, {38, "LPinky4FingerTip"},
//     // Right hand
//     {39, "RThumb1CMC"},         {40, "RThumb2Knuckles"},{41, "RThumb3IP"},  {42, "RThumb4FingerTip"},
//     {43, "RIndex1Knuckles"},    {44, "RIndex2PIP"},     {45, "RIndex3DIP"}, {46, "RIndex4FingerTip"},
//     {47, "RMiddle1Knuckles"},   {48, "RMiddle2PIP"},    {49, "RMiddle3DIP"},{50, "RMiddle4FingerTip"},
//     {51, "RRing1Knuckles"},     {52, "RRing2PIP"},      {53, "RRing3DIP"},  {54, "RRing4FingerTip"},
//     {55, "RPinky1Knuckles"},    {56, "RPinky2PIP"},     {57, "RPinky3DIP"}, {58, "RPinky4FingerTip"},
//     // Background
//     {59, "Background"},
// };
// OPENPOSE_BODY_PARTS_65 {
//     // Body
//     {0,  "Nose"},
//     {1,  "Neck"},
//     {2,  "RShoulder"},
//     {3,  "RElbow"},
//     {4,  "RWrist"},
//     {5,  "LShoulder"},
//     {6,  "LElbow"},
//     {7,  "LWrist"},
//     {8,  "MidHip"},
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
//     {19, "LBigToe"},
//     {20, "LSmallToe"},
//     {21, "LHeel"},
//     {22, "RBigToe"},
//     {23, "RSmallToe"},
//     {24, "RHeel"},
//     // Left hand
//     {25, "LThumb1CMC"},         {26, "LThumb2Knuckles"},{27, "LThumb3IP"},  {28, "LThumb4FingerTip"},
//     {29, "LIndex1Knuckles"},    {30, "LIndex2PIP"},     {31, "LIndex3DIP"}, {32, "LIndex4FingerTip"},
//     {33, "LMiddle1Knuckles"},   {34, "LMiddle2PIP"},    {35, "LMiddle3DIP"},{36, "LMiddle4FingerTip"},
//     {37, "LRing1Knuckles"},     {38, "LRing2PIP"},      {39, "LRing3DIP"},  {40, "LRing4FingerTip"},
//     {41, "LPinky1Knuckles"},    {42, "LPinky2PIP"},     {43, "LPinky3DIP"}, {44, "LPinky4FingerTip"},
//     // Right hand
//     {45, "RThumb1CMC"},         {46, "RThumb2Knuckles"},{47, "RThumb3IP"},  {48, "RThumb4FingerTip"},
//     {49, "RIndex1Knuckles"},    {50, "RIndex2PIP"},     {51, "RIndex3DIP"}, {52, "RIndex4FingerTip"},
//     {53, "RMiddle1Knuckles"},   {54, "RMiddle2PIP"},    {55, "RMiddle3DIP"},{56, "RMiddle4FingerTip"},
//     {57, "RRing1Knuckles"},     {58, "RRing2PIP"},      {59, "RRing3DIP"},  {60, "RRing4FingerTip"},
//     {61, "RPinky1Knuckles"},    {62, "RPinky2PIP"},     {63, "RPinky3DIP"}, {64, "RPinky4FingerTip"},
//     // Background
//     {65, "Background"},
// };
// CAR_12_PARTS {
//     {0,  "FRWheel"},
//     {1,  "FLWheel"},
//     {2,  "BRWheel"},
//     {3,  "BLWheel"},
//     {4,  "FRLight"},
//     {5,  "FLLight"},
//     {6,  "BRLight"},
//     {7,  "BLLight"},
//     {8,  "FRTop"},
//     {9,  "FLTop"},
//     {10, "BRTop"},
//     {11, "BLTop"},
//     {12, "Background"},
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





    // Auxiliary functions
    int poseModelToIndex(const PoseModel poseModel)
    {
        const auto numberBodyParts = getNumberBodyParts(poseModel);
        if (poseModel == PoseModel::COCO_19b)
            return 3;
        else if (poseModel == PoseModel::COCO_19_V2)
            return 4;
        else if (poseModel == PoseModel::CAR_12)
            return 7;
        else if (poseModel == PoseModel::COCO_25E || poseModel == PoseModel::COCO_25_17E)
            return 8;
        else if (numberBodyParts == 18)
            return 0;
        else if (numberBodyParts == 19)
            return 1;
        else if (numberBodyParts == 25)
            return 5;
        else if (numberBodyParts == 59)
            return 2;
        else if (numberBodyParts == 65)
            return 6;
        // else
        throw std::runtime_error{"PoseModel does not have corresponding index yet."
                                 + getLine(__LINE__, __FUNCTION__, __FILE__)};
        return 0;
    }





    // Parameters and functions to change if new PoseModel
    const std::array<int, (int)PoseModel::Size> NUMBER_BODY_PARTS{18, 18, 19, 19, 59, 59, 59, 19, 19, 25, 25, 65, 12, 25, 25};

    const std::array<int, (int)PoseModel::Size> NUMBER_PARTS_LMDB{17, 19, 17, 19, 59, 17, 59, 17, 17, 23, 17, 42, 14, 23, 17};

    const std::array<std::vector<std::vector<int>>, (int)PoseModel::Size> LMDB_TO_OPENPOSE_KEYPOINTS{
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}                   // COCO_18
        },
        std::vector<std::vector<int>>{
            {0},{1}, {2},{3},{4},  {5},{6},{7},  {9},{10},{11}, {12},{13},{14},  {15},{16},{17},{18}                // DOME_18
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}          // COCO_19
        },
        std::vector<std::vector<int>>{
            {0},{1}, {2},{3},{4},  {5},{6},{7},  {8},  {9},{10},{11},  {12},{13},{14},  {15},{16},{17},{18}         // DOME_19
        },
        std::vector<std::vector<int>>{                                                                              // DOME_59
            {0},{1}, {2},{3},{4},  {5},{6},{7},  {8},  {9},{10},{11},  {12},{13},{14},  {15},{16},{17},{18},        // Body
            {19},{20},{21},{22}, {23},{24},{25},{26}, {27},{28},{29},{30}, {31},{32},{33},{34}, {35},{36},{37},{38},// Left hand
            {39},{40},{41},{42}, {43},{44},{45},{46}, {47},{48},{49},{50}, {51},{52},{53},{54}, {55},{56},{57},{58} // Right hand
        },
        std::vector<std::vector<int>>{                                                                              // COCO_59_17
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3},         // Body
            {},{},{},{}, {},{},{},{}, {},{},{},{}, {},{},{},{}, {},{},{},{},                                        // Left hand
            {},{},{},{}, {},{},{},{}, {},{},{},{}, {},{},{},{}, {},{},{},{}                                         // Right hand
        },
        std::vector<std::vector<int>>{                                                                              // MPII_59
            {},{}, {2},{3},{4},  {5},{6},{7},  {},  {9},{10},{11},  {12},{13},{14},  {},{},{},{},                   // Body
            {19},{20},{21},{22}, {23},{24},{25},{26}, {27},{28},{29},{30}, {31},{32},{33},{34}, {35},{36},{37},{38},// Left hand
            {39},{40},{41},{42}, {43},{44},{45},{46}, {47},{48},{49},{50}, {51},{52},{53},{54}, {55},{56},{57},{58} // Right hand
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}          // COCO_19_b
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}          // COCO_19_V2
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}, {17},{18},{19},{20},{21},{22} // COCO_25
            // {},{5,6}, {},{},{}, {},{},{}, {}, {},{},{16}, {},{},{15}, {},{},{},{}, {17},{18},{19},{20},{21},{22}    // COCO_25
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}, {},{},{},{},{},{} // COCO_25_17
        },
        std::vector<std::vector<int>>{                                                                              // MPII_65_42
            {},{}, {},{},{21}, {},{},{0}, {}, {},{},{}, {},{},{}, {},{},{},{}, {},{},{},{},{},{},                   // Body
            {1},{2},{3},{4}, {5},{6},{7},{8}, {9},{10},{11},{12}, {13},{14},{15},{16}, {17},{18},{19},{20},         // Left hand
            {22},{23},{24},{25}, {26},{27},{28},{29}, {30},{31},{32},{33}, {34},{35},{36},{37}, {38},{39},{40},{41} // Right hand
        },
        std::vector<std::vector<int>>{                                                                              // CAR_12
            {0},{1},{2},{3},{4},{5},{6},{7},{9},{10},{11},{12}                                                      // 8 and 13 are always empty
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}, {17},{18},{19},{20},{21},{22} // COCO_25E
            // {},{5,6}, {},{},{}, {},{},{}, {}, {},{},{16}, {},{},{15}, {},{},{},{}, {17},{18},{19},{20},{21},{22} // COCO_25E
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}, {},{},{},{},{},{} // COCO_25_17E
        },
    };

    // Note: Same than LMDB_TO_OPENPOSE_KEYPOINTS unless some keypoint must be masked out
    // E.g., for foot dataset to avoid overfitting on duplicated body keypoints
    const std::array<std::vector<std::vector<int>>, (int)PoseModel::Size> CHANNELS_TO_MASK{
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}                   // COCO_18
        },
        std::vector<std::vector<int>>{
            {0},{1}, {2},{3},{4},  {5},{6},{7},  {9},{10},{11}, {12},{13},{14},  {15},{16},{17},{18}                // DOME_18
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}          // COCO_19
        },
        std::vector<std::vector<int>>{
            {0},{1}, {2},{3},{4},  {5},{6},{7},  {8},  {9},{10},{11},  {12},{13},{14},  {15},{16},{17},{18}         // DOME_19
        },
        std::vector<std::vector<int>>{                                                                              // DOME_59
            {0},{1}, {2},{3},{4},  {5},{6},{7},  {8},  {9},{10},{11},  {12},{13},{14},  {15},{16},{17},{18},        // Body
            {19},{20},{21},{22}, {23},{24},{25},{26}, {27},{28},{29},{30}, {31},{32},{33},{34}, {35},{36},{37},{38},// Left hand
            {39},{40},{41},{42}, {43},{44},{45},{46}, {47},{48},{49},{50}, {51},{52},{53},{54}, {55},{56},{57},{58} // Right hand
        },
        std::vector<std::vector<int>>{                                                                              // COCO_59_17
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3},         // Body
            {},{},{},{}, {},{},{},{}, {},{},{},{}, {},{},{},{}, {},{},{},{},                                        // Left hand
            {},{},{},{}, {},{},{},{}, {},{},{},{}, {},{},{},{}, {},{},{},{}                                         // Right hand
        },
        std::vector<std::vector<int>>{                                                                              // MPII_59
            {},{}, {2},{3},{4},  {5},{6},{7},  {},  {9},{10},{11},  {12},{13},{14},  {},{},{},{},                   // Body
            {19},{20},{21},{22}, {23},{24},{25},{26}, {27},{28},{29},{30}, {31},{32},{33},{34}, {35},{36},{37},{38},// Left hand
            {39},{40},{41},{42}, {43},{44},{45},{46}, {47},{48},{49},{50}, {51},{52},{53},{54}, {55},{56},{57},{58} // Right hand
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}          // COCO_19_b
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}          // COCO_19_V2
        },
        std::vector<std::vector<int>>{
            // {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}, {17},{18},{19},{20},{21},{22} // COCO_25
            {},{5,6}, {},{},{}, {},{},{}, {11,12}, {},{},{16}, {},{},{15}, {},{},{},{}, {17},{18},{19},{20},{21},{22}// COCO_25
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}, {},{},{},{},{},{} // COCO_25_17
        },
        std::vector<std::vector<int>>{                                                                              // MPII_65_42
            {},{}, {},{},{21}, {},{},{0}, {}, {},{},{}, {},{},{}, {},{},{},{}, {},{},{},{},{},{},                   // Body
            {1},{2},{3},{4}, {5},{6},{7},{8}, {9},{10},{11},{12}, {13},{14},{15},{16}, {17},{18},{19},{20},         // Left hand
            {22},{23},{24},{25}, {26},{27},{28},{29}, {30},{31},{32},{33}, {34},{35},{36},{37}, {38},{39},{40},{41} // Right hand
        },
        std::vector<std::vector<int>>{                                                                              // CAR_12
            {0},{1},{2},{3},{4},{5},{6},{7},{9},{10},{11},{12}                                                      // 8 and 13 are always empty
        },
        std::vector<std::vector<int>>{
            // {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}, {17},{18},{19},{20},{21},{22} // COCO_25E
            {},{5,6}, {},{},{}, {},{},{}, {11,12}, {},{},{16}, {},{},{15}, {},{},{},{}, {17},{18},{19},{20},{21},{22}// COCO_25E
        },
        std::vector<std::vector<int>>{
            {0},{5,6}, {6},{8},{10}, {5},{7},{9}, {11,12}, {12},{14},{16}, {11},{13},{15}, {2},{1},{4},{3}, {},{},{},{},{},{} // COCO_25_17E
        },
    };

    std::pair<PoseModel,PoseCategory> flagsToPoseModel(const std::string& poseModeString)
    {
        // COCO
        if (poseModeString == "COCO_18")
            return std::make_pair(PoseModel::COCO_18, PoseCategory::COCO);
        else if (poseModeString == "COCO_19")
            return std::make_pair(PoseModel::COCO_19, PoseCategory::COCO);
        else if (poseModeString == "COCO_19b")
            return std::make_pair(PoseModel::COCO_19b, PoseCategory::COCO);
        else if (poseModeString == "COCO_19_V2")
            return std::make_pair(PoseModel::COCO_19_V2, PoseCategory::COCO);
        else if (poseModeString == "COCO_25")
            return std::make_pair(PoseModel::COCO_25, PoseCategory::COCO);
        else if (poseModeString == "COCO_25_17")
            return std::make_pair(PoseModel::COCO_25_17, PoseCategory::COCO);
        else if (poseModeString == "COCO_25E")
            return std::make_pair(PoseModel::COCO_25E, PoseCategory::COCO);
        else if (poseModeString == "COCO_25_17E")
            return std::make_pair(PoseModel::COCO_25_17E, PoseCategory::COCO);
        else if (poseModeString == "COCO_59_17")
            return std::make_pair(PoseModel::COCO_59_17, PoseCategory::COCO);
        // Dome
        else if (poseModeString == "DOME_18")
            return std::make_pair(PoseModel::DOME_18, PoseCategory::DOME);
        else if (poseModeString == "DOME_19")
            return std::make_pair(PoseModel::DOME_19, PoseCategory::DOME);
        else if (poseModeString == "DOME_59")
            return std::make_pair(PoseModel::DOME_59, PoseCategory::DOME);
        // MPII
        else if (poseModeString == "MPII_59")
            return std::make_pair(PoseModel::MPII_59, PoseCategory::MPII);
        else if (poseModeString == "MPII_65_42")
            return std::make_pair(PoseModel::MPII_65_42, PoseCategory::MPII);
        else if (poseModeString == "CAR_12")
            return std::make_pair(PoseModel::CAR_12, PoseCategory::CAR);
        // Unknown
        throw std::runtime_error{"String (" + poseModeString
                                 + ") does not correspond to any model (COCO_18, DOME_18, ...)"
                                 + getLine(__LINE__, __FUNCTION__, __FILE__)};
        return std::make_pair(PoseModel::COCO_18, PoseCategory::COCO);
    }





    // Parameters and functions to change if new number body parts
    const std::array<std::vector<std::array<int,2>>, (int)PoseModel::Size> SWAP_LEFT_RIGHT_KEYPOINTS{
        std::vector<std::array<int,2>>{{5,2},{6,3},{7,4},{11,8},{12,9},{13,10},{15,14},{17,16}},                    // 18 (COCO_18, DOME_18)
        std::vector<std::array<int,2>>{{5,2},{6,3},{7,4},{12,9},{13,10},{14,11},{16,15},{18,17}},                   // 19 (COCO_19(b), DOME_19)
        std::vector<std::array<int,2>>{{5,2},{6,3},{7,4},{12,9},{13,10},{14,11},{16,15},{18,17},                    // 59 (DOME_59), COCO_59_17, MPII_59
                                       {19,39},{20,40},{21,41},{22,42},{23,43},{24,44},{25,45},{26,46},     // 2 fingers
                                       {27,47},{28,48},{29,49},{30,50},{31,51},{32,52},{33,53},{34,54},     // 2 fingers
                                       {35,55},{36,56},{37,57},{38,58}},                                    // 1 finger
        std::vector<std::array<int,2>>{{5,2},{6,3},{7,4},{12,9},{13,10},{14,11},{16,15},{18,17}},                   // COCO_19b
        std::vector<std::array<int,2>>{{5,2},{6,3},{7,4},{12,9},{13,10},{14,11},{16,15},{18,17}},                   // COCO_19_V2
        std::vector<std::array<int,2>>{{5,2},{6,3},{7,4},{12,9},{13,10},{14,11},{16,15},{18,17},{19,22},{20,23},{21,24}}, // 25 (COCO_25, COCO_25_17)
        std::vector<std::array<int,2>>{{7,4},                                                               // 65 (MPII_65_42)
                                       {25,45},{26,46},{27,47},{28,48},{29,49},{30,50},{31,51},{32,52},     // 2 fingers
                                       {33,53},{34,54},{35,55},{36,56},{37,57},{38,58},{39,59},{40,60},     // 2 fingers
                                       {41,61},{42,62},{43,63},{44,64}},                                    // 1 finger
        std::vector<std::array<int,2>>{{0,1},{2,3},{4,5},{6,7},{8,9},{10,11}},                                      // CAR_12
        std::vector<std::array<int,2>>{{5,2},{6,3},{7,4},{12,9},{13,10},{14,11},{16,15},{18,17},{19,22},{20,23},{21,24}}, // 25E (COCO_25E, COCO_25_17E)
    };

    const std::array<std::vector<int>, (int)PoseModel::Size> LABEL_MAP_A{
        std::vector<int>{1, 8,  9, 1,   11, 12, 1, 2, 3,  2, 1, 5, 6, 5,  1, 0,  0,  14, 15},                       // 18 (COCO_18, DOME_18)
        std::vector<int>{1, 9, 10, 8,8, 12, 13, 1, 2, 3,  2, 1, 5, 6, 5,  1, 0,  0,  15, 16},                       // 19 (COCO_19, DOME_19)
        std::vector<int>{1, 9, 10, 8,8, 12, 13, 1, 2, 3,  2, 1, 5, 6, 5,  1, 0,  0,  15, 16,                        // 59 (DOME_59), COCO_59_17, MPII_59
                         7,19,20,21, 7,23,24,25, 7,27,28,29, 7,31,32,33, 7,35,36,37, // Left hand
                         4,39,40,41, 4,43,44,45, 4,47,48,49, 4,51,52,53, 4,55,56,57},// Right hand
        std::vector<int>{1, 9, 10, 8,8, 12, 13, 1, 2, 3,  2, 1, 5, 6, 5,  1, 0,  0,  15, 16, 2, 5},                 // COCO_19b
        std::vector<int>{1,1,1,1,1,1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1},                                             // COCO_19_V2
        std::vector<int>{1, 9, 10, 8,8, 12, 13, 1, 2, 3,  2, 1, 5, 6, 5,  1, 0,  0,  15, 16, 14,19,14, 11,22,11},   // 25 (COCO_25, COCO_25_17)
        std::vector<int>{1, 9, 10, 8,8, 12, 13, 1, 2, 3,  2, 1, 5, 6, 5,  1, 0,  0,  15, 16, 14,19,14, 11,22,11,    // 65 (MPII_65_42)
                         7,25,26,27, 7,29,30,31, 7,33,34,35, 7,37,38,39, 7,41,42,43, // Left hand
                         4,45,46,47, 4,49,50,51, 4,53,54,55, 4,57,58,59, 4,61,62,63},// Right hand
        std::vector<int>{4, 4,4,0,4,8, 5,5,1,5,9},                                                                  // CAR_12
        std::vector<int>{1, 9, 10, 8,8, 12, 13, 1, 2, 3,     1, 5, 6,     1, 0,  0,  15, 16, 14,19,14, 11,22,11,    // 25 (COCO_25E, COCO_25_17E)
                         1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1},
    };

    const std::array<std::vector<int>, (int)PoseModel::Size> LABEL_MAP_B{
        std::vector<int>{8, 9, 10, 11,  12, 13, 2, 3, 4, 16, 5, 6, 7, 17, 0, 14, 15, 16, 17},                       // 18 (COCO_18, DOME_18)
        std::vector<int>{8,10, 11, 9,12,13, 14, 2, 3, 4, 17, 5, 6, 7, 18, 0, 15, 16, 17, 18},                       // 19 (COCO_19, DOME_19)
        std::vector<int>{8,10, 11, 9,12,13, 14, 2, 3, 4, 17, 5, 6, 7, 18, 0, 15, 16, 17, 18,                        // 59 (DOME_59), COCO_59_17, MPII_59
                         19,20,21,22, 23,24,25,26, 27,28,29,30, 31,32,33,34, 35,36,37,38, // Left hand
                         39,40,41,42, 43,44,45,46, 47,48,49,50, 51,52,53,54, 55,56,57,58},// Right hand
        std::vector<int>{8,10, 11, 9,12,13, 14, 2, 3, 4, 17, 5, 6, 7, 18, 0, 15, 16, 17, 18, 9, 12},                // COCO_19b
        std::vector<int>{0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18},                                             // COCO_19_V2
        std::vector<int>{8,10, 11, 9,12,13, 14, 2, 3, 4, 17, 5, 6, 7, 18, 0, 15, 16, 17, 18, 19,20,21, 22,23,24},   // 25 (COCO_25, COCO_25_17)
        std::vector<int>{8,10, 11, 9,12,13, 14, 2, 3, 4, 17, 5, 6, 7, 18, 0, 15, 16, 17, 18, 19,20,21, 22,23,24,    // 65 (MPII_65_42)
                         25,26,27,28, 29,30,31,32, 33,34,35,36, 37,38,39,40, 41,42,43,44, // Left hand
                         45,46,47,48, 49,50,51,52, 53,54,55,56, 57,58,59,60, 61,62,63,64},// Right hand
        std::vector<int>{5, 6,0,2,8,10, 7,1,3,9,11},                                                                // CAR_12
        std::vector<int>{8,10, 11, 9,12,13, 14, 2, 3, 4,     5, 6, 7,     0, 15, 16, 17, 18, 19,20,21, 22,23,24,    // 25 (COCO_25E, COCO_25_17E)
                         3,4,6,7,9, 10,11,12,13,14, 15,16,17,18,19, 20,21,22,23,24}, // 0, 2, 5, 8 already done, 1 not required
    };

    const std::array<std::vector<float>, (int)PoseModel::Size> DISTANCE_AVERAGE{
        std::vector<float>{}, // 18 (COCO_18, DOME_18)
        std::vector<float>{0, -2.76364, -1.3345, 0,   -1.95322, 3.95679, -1.20664, 4.76543, // 19 (COCO_19, DOME_19)
                           1.3345, 0, 1.92318, 3.96891,   1.17999, 4.7901, 0, 7.72201,
                           -0.795236, 7.74017, -0.723963,   11.209, -0.651316, 15.6972,
                           0.764623, 7.74869, 0.70755,   11.2307, 0.612832, 15.7281,
                           -0.123134, -3.43515,   0.111775, -3.42761,
                           -0.387066, -3.16603,   0.384038, -3.15951},
        std::vector<float>{}, // 59 (DOME_59), COCO_59_17, MPII_59
        std::vector<float>{}, // COCO_19b
        std::vector<float>{}, // COCO_19_V2
        // std::vector<float>{0, -2.76364, -1.3345, 0,   -1.95322, 3.95679, -1.20664, 4.76543, // 25 (COCO_25, COCO_25_17) // 48 channels
        //                    1.3345, 0, 1.92318, 3.96891,   1.17999, 4.7901, 0, 7.72201,
        //                    -0.795236, 7.74017, -0.723963,   11.209, -0.651316, 15.6972,
        //                    0.764623, 7.74869, 0.70755,   11.2307, 0.612832, 15.7281,
        //                    -0.123134, -3.43515,   0.111775, -3.42761,
        //                    -0.387066, -3.16603,   0.384038, -3.15951,
        //                    0.344764, 12.9666, 0.624157,   12.9057, 0.195454, 12.565,
        //                    -1.06074, 12.9951, -1.2427,   12.9309, -0.800837, 12.5845},
        std::vector<float>{0, -6.55251, // 50 channels
                           0, -4.15062, -1.48818, -4.15506,   -2.22408, -0.312264, -1.42204, 0.588495,
                           1.51044, -4.14629, 2.2113, -0.312283,   1.41081, 0.612377, -0, 3.41112,
                           -0.932306, 3.45504, -0.899812,   6.79837, -0.794223, 11.4972,
                           0.919047, 3.46442, 0.902314,   6.81245, 0.79518, 11.5132,
                           -0.243982, -7.07925,   0.28065, -7.07398,
                           -0.792812, -7.09374,   0.810145, -7.06958,
                           0.582387, 7.46846, 0.889349,   7.40577, 0.465088, 7.03969,
                           -0.96686, 7.46148, -1.20773,   7.38834, -0.762135, 6.99575},
        // std::vector<float>{0.f,0.f,
        //                    0.f,0.f,0.f,0.f,   0.f,0.f,0.f,0.f,
        //                    0.f,0.f,0.f,0.f,   0.f,0.f,0.f,0.f,
        //                    0.f,0.f,0.f,   0.f,0.f,0.f,
        //                    0.f,0.f,0.f,   0.f,0.f,0.f,
        //                    0.f,0.f,   0.f,0.f,
        //                    0.f,0.f,   0.f,0.f,
        //                    0.f,0.f,0.f,   0.f,0.f,0.f,
        //                    0.f,0.f,0.f,   0.f,0.f,0.f},
        std::vector<float>{}, // 65 (MPII_65_42)
        std::vector<float>{}, // CAR_12
        // std::vector<float>{0, -2.76364, -1.3345, 0,   -1.95322, 3.95679, -1.20664, 4.76543, // 25 (COCO_25E, COCO_25_17E) // 48 channels
        //                    1.3345, 0, 1.92318, 3.96891,   1.17999, 4.7901, 0, 7.72201,
        //                    -0.795236, 7.74017, -0.723963,   11.209, -0.651316, 15.6972,
        //                    0.764623, 7.74869, 0.70755,   11.2307, 0.612832, 15.7281,
        //                    -0.123134, -3.43515,   0.111775, -3.42761,
        //                    -0.387066, -3.16603,   0.384038, -3.15951,
        //                    0.344764, 12.9666, 0.624157,   12.9057, 0.195454, 12.565,
        //                    -1.06074, 12.9951, -1.2427,   12.9309, -0.800837, 12.5845},
        std::vector<float>{0, -6.55251, // 50 channels
                           0, -4.15062, -1.48818, -4.15506,   -2.22408, -0.312264, -1.42204, 0.588495,
                           1.51044, -4.14629, 2.2113, -0.312283,   1.41081, 0.612377, -0, 3.41112,
                           -0.932306, 3.45504, -0.899812,   6.79837, -0.794223, 11.4972,
                           0.919047, 3.46442, 0.902314,   6.81245, 0.79518, 11.5132,
                           -0.243982, -7.07925,   0.28065, -7.07398,
                           -0.792812, -7.09374,   0.810145, -7.06958,
                           0.582387, 7.46846, 0.889349,   7.40577, 0.465088, 7.03969,
                           -0.96686, 7.46148, -1.20773,   7.38834, -0.762135, 6.99575},
        // std::vector<float>{0.f,0.f, // 50 channels
        //                    0.f,0.f,0.f,0.f,   0.f,0.f,0.f,0.f,
        //                    0.f,0.f,0.f,0.f,   0.f,0.f,0.f,0.f,
        //                    0.f,0.f,0.f,   0.f,0.f,0.f,
        //                    0.f,0.f,0.f,   0.f,0.f,0.f,
        //                    0.f,0.f,   0.f,0.f,
        //                    0.f,0.f,   0.f,0.f,
        //                    0.f,0.f,0.f,   0.f,0.f,0.f,
        //                    0.f,0.f,0.f,   0.f,0.f,0.f},
    };

    const std::array<std::vector<float>, (int)PoseModel::Size> DISTANCE_SIGMA{
        std::vector<float>{}, // 18 (COCO_18, DOME_18)
        std::vector<float>{3.39629, 3.15605, 3.16913, 1.8234,   5.82252, 5.05674, 7.09876, 6.64574, // 19 (COCO_19, DOME_19)
                           3.16913, 1.8234, 5.79415, 5.01424,   7.03866, 6.62427, 5.52593, 6.75962,
                           5.91224, 6.87241, 8.66473,   10.1792, 11.5871, 13.6565,
                           5.86653, 6.89568, 8.68067,   10.2127, 11.5954, 13.6722,
                           3.3335, 3.49128,   3.34476, 3.50079,
                           2.93982, 3.11151,   2.95006, 3.11004},
        std::vector<float>{}, // 59 (DOME_59), COCO_59_17, MPII_59
        std::vector<float>{}, // COCO_19b
        std::vector<float>{}, // COCO_19_V2
        // std::vector<float>{3.39629, 3.15605, 3.16913, 1.8234,   5.82252, 5.05674, 7.09876, 6.64574, // 25 (COCO_25, COCO_25_17)
        //                    3.16913, 1.8234, 5.79415, 5.01424,   7.03866, 6.62427, 5.52593, 6.75962,
        //                    5.91224, 6.87241, 8.66473,   10.1792, 11.5871, 13.6565,
        //                    5.86653, 6.89568, 8.68067,   10.2127, 11.5954, 13.6722,
        //                    3.3335, 3.49128,   3.34476, 3.50079,
        //                    2.93982, 3.11151,   2.95006, 3.11004,
        //                    9.69408, 7.58921, 9.71193,   7.44185, 9.19343, 7.11157,
        //                    9.16848, 7.86122, 9.07613,   7.83682, 8.91951, 7.33715},
        std::vector<float>{7.26789, 9.70751, // 50 channels
                           6.29588, 8.93472, 6.97401, 9.13746,   7.49632, 9.44757, 8.06695, 9.97319,
                           6.99726, 9.14608, 7.50529, 9.43568,   8.05888, 9.98207, 6.38929, 9.29314,
                           6.71801, 9.39271, 8.00608,   10.6141, 10.3416, 12.7812,
                           6.69875, 9.41407, 8.01876,   10.637, 10.3475, 12.7849,
                           7.30923, 9.7324,   7.27886, 9.73406,
                           7.35978, 9.7289,   7.28914, 9.67711,
                           7.93153, 8.10845, 7.95577,   8.01729, 7.56865, 7.87314,
                           7.4655, 8.25336, 7.43958,   8.26333, 7.33667, 7.97446},
        // std::vector<float>{1.f,1.f,
        //                    1.f,1.f,1.f,1.f,   1.f,1.f,1.f,1.f,
        //                    1.f,1.f,1.f,1.f,   1.f,1.f,1.f,1.f,
        //                    1.f,1.f,1.f,   1.f,1.f,1.f,
        //                    1.f,1.f,1.f,   1.f,1.f,1.f,
        //                    1.f,1.f,   1.f,1.f,
        //                    1.f,1.f,   1.f,1.f,
        //                    1.f,1.f,1.f,   1.f,1.f,1.f,
        //                    1.f,1.f,1.f,   1.f,1.f,1.f},
        std::vector<float>{}, // 65 (MPII_65_42)
        std::vector<float>{}, // CAR_12
        // std::vector<float>{3.39629, 3.15605, 3.16913, 1.8234,   5.82252, 5.05674, 7.09876, 6.64574, // 25 (COCO_25E, COCO_25_17E) // 48 channels
        //                    3.16913, 1.8234, 5.79415, 5.01424,   7.03866, 6.62427, 5.52593, 6.75962,
        //                    5.91224, 6.87241, 8.66473,   10.1792, 11.5871, 13.6565,
        //                    5.86653, 6.89568, 8.68067,   10.2127, 11.5954, 13.6722,
        //                    3.3335, 3.49128,   3.34476, 3.50079,
        //                    2.93982, 3.11151,   2.95006, 3.11004,
        //                    9.69408, 7.58921, 9.71193,   7.44185, 9.19343, 7.11157,
        //                    9.16848, 7.86122, 9.07613,   7.83682, 8.91951, 7.33715},
        std::vector<float>{7.26789, 9.70751, // 50 channels
                           6.29588, 8.93472, 6.97401, 9.13746,   7.49632, 9.44757, 8.06695, 9.97319,
                           6.99726, 9.14608, 7.50529, 9.43568,   8.05888, 9.98207, 6.38929, 9.29314,
                           6.71801, 9.39271, 8.00608,   10.6141, 10.3416, 12.7812,
                           6.69875, 9.41407, 8.01876,   10.637, 10.3475, 12.7849,
                           7.30923, 9.7324,   7.27886, 9.73406,
                           7.35978, 9.7289,   7.28914, 9.67711,
                           7.93153, 8.10845, 7.95577,   8.01729, 7.56865, 7.87314,
                           7.4655, 8.25336, 7.43958,   8.26333, 7.33667, 7.97446},
        // std::vector<float>{1.f,1.f,
        //                    1.f,1.f,1.f,1.f,   1.f,1.f,1.f,1.f,
        //                    1.f,1.f,1.f,1.f,   1.f,1.f,1.f,1.f,
        //                    1.f,1.f,1.f,   1.f,1.f,1.f,
        //                    1.f,1.f,1.f,   1.f,1.f,1.f,
        //                    1.f,1.f,   1.f,1.f,
        //                    1.f,1.f,   1.f,1.f,
        //                    1.f,1.f,1.f,   1.f,1.f,1.f,
        //                    1.f,1.f,1.f,   1.f,1.f,1.f},
    };

    const std::array<unsigned int, (int)PoseModel::Size> ROOT_INDEXES{
        1u,     // 18 (COCO_18, DOME_18)
        1u,     // 19 (COCO_19, DOME_19)
        1u,     // 59 (DOME_59), COCO_59_17, MPII_59
        1u,     // COCO_19b
        1u,     // COCO_19_V2
        1u,     // 25 (COCO_25, COCO_25_17)
        1u,     // 65 (MPII_65_42)
        1u,     // CAR_12
        1u,     // 25 (COCO_25E, COCO_25_17E)
    };





    // Fixed functions
    int getNumberBodyParts(const PoseModel poseModel)
    {
        return NUMBER_BODY_PARTS.at((int)poseModel);
    }

    int getNumberBodyPartsLmdb(const PoseModel poseModel)
    {
        return NUMBER_PARTS_LMDB.at((int)poseModel);
    }

    int getNumberPafChannels(const PoseModel poseModel)
    {
        return (int)(2*getPafIndexA(poseModel).size());
        // return 2*(NUMBER_BODY_PARTS.at((int)poseModel)+1); // Doesn't work for COCO_19b and COCO_19_V2
    }

    int getNumberBodyAndPafChannels(const PoseModel poseModel)
    {
        return NUMBER_BODY_PARTS.at((int)poseModel) + getNumberPafChannels(poseModel);
    }

    int getNumberBodyBkgAndPAF(const PoseModel poseModel)
    {
        return getNumberBodyAndPafChannels(poseModel) + 1;
    }

    const std::vector<std::vector<int>>& getLmdbToOpenPoseKeypoints(const PoseModel poseModel)
    {
        return LMDB_TO_OPENPOSE_KEYPOINTS.at((int)poseModel);
    }

    const std::vector<std::vector<int>>& getMaskedChannels(const PoseModel poseModel)
    {
        return CHANNELS_TO_MASK.at((int)poseModel);
    }

    const std::vector<std::array<int,2>>& getSwapLeftRightKeypoints(const PoseModel poseModel)
    {
        return SWAP_LEFT_RIGHT_KEYPOINTS.at(poseModelToIndex(poseModel));
    }

    const std::vector<int>& getPafIndexA(const PoseModel poseModel)
    {
        return LABEL_MAP_A.at(poseModelToIndex(poseModel));
    }

    const std::vector<int>& getPafIndexB(const PoseModel poseModel)
    {
        return LABEL_MAP_B.at(poseModelToIndex(poseModel));
    }

    const std::vector<float>& getDistanceAverage(const PoseModel poseModel)
    {
        return DISTANCE_AVERAGE.at(poseModelToIndex(poseModel));
    }

    const std::vector<float>& getDistanceSigma(const PoseModel poseModel)
    {
        return DISTANCE_SIGMA.at(poseModelToIndex(poseModel));
    }

    unsigned int getRootIndex(const PoseModel poseModel)
    {
        return ROOT_INDEXES.at(poseModelToIndex(poseModel));
    }

    std::vector<int> getIndexesForParts(const PoseModel poseModel, const std::vector<int>& missingBodyPartsBase,
                                        const std::vector<float>& isVisible)
    {
        auto missingBodyParts = missingBodyPartsBase;
        // If masking also non visible points
        if (!isVisible.empty())
        {
            for (auto i = 0u ; i < isVisible.size() ; i++)
                if (isVisible[i] >= 2.f)
                    missingBodyParts.emplace_back(i);
            std::sort(missingBodyParts.begin(), missingBodyParts.end());
        }
        // Missing PAF channels
        std::vector<int> missingChannels;
        if (!missingBodyParts.empty())
        {
            const auto& pafIndexA = getPafIndexA(poseModel);
            const auto& pafIndexB = getPafIndexB(poseModel);
            for (auto i = 0u ; i < missingBodyParts.size() ; i++)
            {
                for (auto pafId = 0u ; pafId < pafIndexA.size() ; pafId++)
                {
                    if (pafIndexA[pafId] == missingBodyParts[i] || pafIndexB[pafId] == missingBodyParts[i])
                    {
                        missingChannels.emplace_back(2*pafId);
                        missingChannels.emplace_back(2*pafId+1);
                    }
                }
            }
            // Sort indexes (only disordered in PAFs)
            std::sort(missingChannels.begin(), missingChannels.end());
            // Remove duplicates (only possible in PAFs)
            const auto it = std::unique(missingChannels.begin(), missingChannels.end());
            missingChannels.resize(std::distance(missingChannels.begin(), it));
            // Body parts to channel indexes (add #PAF channels)
            std::transform(missingBodyParts.begin(), missingBodyParts.end(), missingBodyParts.begin(),
                           std::bind2nd(std::plus<int>(), getNumberPafChannels(poseModel)));
            missingChannels.insert(missingChannels.end(), missingBodyParts.begin(), missingBodyParts.end());
        }
        // Return result
        return missingChannels;
    }

    std::vector<int> getMissingChannels(const PoseModel poseModel, const std::vector<float>& isVisible)
    {
        // Missing body parts
        std::vector<int> missingBodyParts;
        // const auto& lmdbToOpenPoseKeypoints = getLmdbToOpenPoseKeypoints(poseModel);
        const auto& lmdbToOpenPoseKeypoints = getMaskedChannels(poseModel);
        for (auto i = 0u ; i < lmdbToOpenPoseKeypoints.size() ; i++)
            if (lmdbToOpenPoseKeypoints[i].empty())
                missingBodyParts.emplace_back(i);
        return getIndexesForParts(poseModel, missingBodyParts, isVisible);
    }
}  // namespace caffe
