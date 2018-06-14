#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream> // std::ifstream
#include <iostream>
#include <stdexcept> // std::runtime_error
// #include <opencv2/contrib/contrib.hpp> // cv::CLAHE, CV_Lab2BGR
#include <caffe/openpose/getLine.hpp>
#include <caffe/openpose/dataAugmentation.hpp>

namespace caffe {
    // Private functions
    bool onPlane(const cv::Point& point, const cv::Size& imageSize)
    {
        return (point.x >= 0 && point.y >= 0
                && point.x < imageSize.width && point.y < imageSize.height);
    }

    void swapLeftRightKeypoints(Joints& joints, const PoseModel poseModel)
    {
        const auto& swapLeftRightKeypoints = getSwapLeftRightKeypoints(poseModel);
        for (const auto& swapLeftRightKeypoint : swapLeftRightKeypoints)
        {
            const auto li = swapLeftRightKeypoint[0];
            const auto ri = swapLeftRightKeypoint[1];
            std::swap(joints.points[ri], joints.points[li]);
            std::swap(joints.isVisible[ri], joints.isVisible[li]);
        }
    }

    void flipKeypoints(Joints& joints, cv::Point2f& objPos, const int widthMinusOne, const PoseModel poseModel)
    {
        objPos.x = widthMinusOne - objPos.x;
        for (auto& point : joints.points)
            point.x = widthMinusOne - point.x;
        if(joints.points.size()) swapLeftRightKeypoints(joints, poseModel);
    }

    void flipKeypoints(Joints& joints, const int widthMinusOne, const PoseModel poseModel)
    {
        if(!joints.points.size()) return;
        for (auto& point : joints.points)
            point.x = widthMinusOne - point.x;
        swapLeftRightKeypoints(joints, poseModel);
    }

    // Public functions
    void swapCenterPoint(MetaData& metaData, const OPTransformationParameter& param_, const PoseModel poseModel)
    {
        // Estimate random scale
        const float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
        if (dice < param_.center_swap_prob())
        {
            // const float dice2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
            if (poseModel == PoseModel::DOME_59)
            {
                const auto& isVisible = metaData.jointsSelf.isVisible;
                const auto& points = metaData.jointsSelf.points;
                if (isVisible[4] <= 1 && isVisible[7] <= 1)
                    metaData.objPos = (points[4] + points[7]) * 0.5f;
                else if (isVisible[4] <= 1)
                    metaData.objPos = points[4];
                else if (isVisible[7] <= 1)
                    metaData.objPos = points[7];
            }
            else
                throw std::runtime_error{"Only implemented for DOME_59"
                                         + getLine(__LINE__, __FUNCTION__, __FILE__)};
        }
    }

    float estimateScale(const MetaData& metaData, const OPTransformationParameter& param_)
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

    // void applyScale(cv::Mat& imageAugmented, const float scale, const cv::Mat& image)
    // {
    //     // Scale image
    //     if (!image.empty())
    //         cv::resize(image, imageAugmented, cv::Size{}, scale, scale, cv::INTER_CUBIC);
    //     // Not used given that net makes x8 pooling anyway...
    //     //     // Image sharpening
    //     //     if (scale > 2.5 && imageAugmented.channels() == 3)
    //     //     {
    //     //         cv::Mat gaussianImage;
    //     //         cv::GaussianBlur(imageAugmented, gaussianImage, cv::Size(0, 0), 3);
    //     //         cv::addWeighted(imageAugmented, 1.5, gaussianImage, -0.5, 0, imageAugmented);
    //     //     }
    // }

    void applyScale(MetaData& metaData, const float scale, const PoseModel poseModel)
    {
        // Update metaData
        metaData.objPos *= scale;
        metaData.scaleSelf *= scale;
        for (auto& point : metaData.jointsSelf.points)
            point *= scale;
        for (auto person=0; person<metaData.numberOtherPeople; person++)
        {
            if(metaData.objPosOthers.size()) metaData.objPosOthers[person] *= scale;
            if(metaData.scaleOthers.size()) metaData.scaleOthers[person] *= scale;
            for (auto& point : metaData.jointsOthers[person].points)
                point *= scale;
            if(metaData.jointsOthersPrev.size()){
                for (auto& point : metaData.jointsOthersPrev[person].points)
                    point *= scale;
            }
        }
    }

    float getRotRand(const OPTransformationParameter& param_){
        float rotation;
        const float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        rotation = (dice - 0.5f) * 2 * param_.max_rotate_degree();
        return rotation;
    }

    std::pair<cv::Mat, cv::Size> estimateRotation(const MetaData& metaData, const cv::Size& imageSize,
                                                  const float rotation)
    {
        // Estimate center & BBox
        const cv::Point2f center{imageSize.width / 2.f, imageSize.height / 2.f};
        const cv::Rect bbox = cv::RotatedRect(center, imageSize, rotation).boundingRect();
        // Adjust transformation matrix
        cv::Mat Rot = cv::getRotationMatrix2D(center, rotation, 1.0);
        Rot.at<double>(0,2) += bbox.width/2. - center.x;
        Rot.at<double>(1,2) += bbox.height/2. - center.y;
        return std::make_pair(Rot, bbox.size());
    }

    std::pair<cv::Mat, cv::Size> estimateRotation(const MetaData& metaData, const cv::Size& imageSize,
                                                  const OPTransformationParameter& param_)
    {
        // Estimate random rotation
        float rotation;
        const float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        rotation = (dice - 0.5f) * 2 * param_.max_rotate_degree();
        // Estimate center & BBox
        const cv::Point2f center{imageSize.width / 2.f, imageSize.height / 2.f};
        const cv::Rect bbox = cv::RotatedRect(center, imageSize, rotation).boundingRect();
        // Adjust transformation matrix
        cv::Mat Rot = cv::getRotationMatrix2D(center, rotation, 1.0);
        Rot.at<double>(0,2) += bbox.width/2. - center.x;
        Rot.at<double>(1,2) += bbox.height/2. - center.y;
        return std::make_pair(Rot, bbox.size());
    }

    // void applyRotation(cv::Mat& imageAugmented, const std::pair<cv::Mat, cv::Size>& RotAndFinalSize,
    //                    const cv::Mat& image, const unsigned char defaultBorderValue)
    // {
    //     // Rotate image
    //     if (!image.empty())
    //         cv::warpAffine(image, imageAugmented, RotAndFinalSize.first, RotAndFinalSize.second, cv::INTER_CUBIC,
    //                        cv::BORDER_CONSTANT, cv::Scalar{(double)defaultBorderValue});
    // }

    void applyRotation(MetaData& metaData, const cv::Mat& Rot, const PoseModel poseModel)
    {
        // Update metaData
        rotatePoint(metaData.objPos, Rot);
        for (auto& point : metaData.jointsSelf.points)
            rotatePoint(point, Rot);
        for (auto person = 0; person < metaData.numberOtherPeople; person++)
        {
            if(metaData.objPosOthers.size()) rotatePoint(metaData.objPosOthers[person], Rot);
            for (auto& point : metaData.jointsOthers[person].points)
                rotatePoint(point, Rot);
            if(metaData.jointsOthersPrev.size()){
                for (auto& point : metaData.jointsOthersPrev[person].points)
                    rotatePoint(point, Rot);
            }
        }
    }

    cv::Size estimatePO(const MetaData& metaData, const OPTransformationParameter& param_)
    {
        // Estimate random crop
        const float diceX = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
        const float diceY = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]

        const cv::Size pointOffset{int((diceX - 0.5f) * 2.f * param_.center_perterb_max()),
                                   int((diceY - 0.5f) * 2.f * param_.center_perterb_max())};
        return pointOffset;
    }

    cv::Point2i addPO(const MetaData& metaData, const cv::Size pointOffset)
    {
        const cv::Point2i cropCenter{
            (int)(metaData.objPos.x + pointOffset.width),
            (int)(metaData.objPos.y + pointOffset.height),
        };
        return cropCenter;
    }

    cv::Point2i estimateCrop(const MetaData& metaData, const OPTransformationParameter& param_)
    {
        // Estimate random crop
        const float diceX = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
        const float diceY = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]

        const cv::Size pointOffset{int((diceX - 0.5f) * 2.f * param_.center_perterb_max()),
                                   int((diceY - 0.5f) * 2.f * param_.center_perterb_max())};
        const cv::Point2i cropCenter{
            (int)(metaData.objPos.x + pointOffset.width),
            (int)(metaData.objPos.y + pointOffset.height),
        };
        return cropCenter;
    }

    void applyCrop(cv::Mat& imageAugmented, const cv::Point2i& cropCenter, const cv::Mat& image,
                   const unsigned char defaultBorderValue, const cv::Size& cropSize)
    {
        if (!image.empty())
        {
            // Security checks
            if (imageAugmented.data == image.data)
                throw std::runtime_error{"Input and output images must be different"
                                         + getLine(__LINE__, __FUNCTION__, __FILE__)};
            // Parameters
            const auto cropX = (int)cropSize.width;
            const auto cropY = (int)cropSize.height;
            // Crop image
//             // OpenCV warping - Efficient implementation - x1.5 times faster
//             cv::Mat matrix = cv::Mat::eye(2,3, CV_64F);
//             matrix.at<double>(0,2) = -(cropCenter.x - cropSize.width/2.f);
//             matrix.at<double>(1,2) = -(cropCenter.y - cropSize.height/2.f);
//             // Apply warping
//             cv::warpAffine(image, imageAugmented, matrix, cropSize,
//                            // (scale < 1 ? cv::INTER_AREA : cv::INTER_CUBIC),
//                            cv::INTER_NEAREST, // CUBIC to consider rotations
//                            // cv::INTER_CUBIC, // CUBIC to consider rotations
//                            cv::BORDER_CONSTANT, cv::Scalar{(double)defaultBorderValue});
            // 1. Allocate memory
            imageAugmented = cv::Mat(cropY, cropX, image.type(), cv::Scalar{(double)defaultBorderValue});
            // 2. Fill memory
            // OpenCV wrappering - Efficient implementation - x15 times faster
            // Estimate ROIs
            const auto offsetX = std::max(0, cropCenter.x - cropX/2);
            const auto offsetY = std::max(0, cropCenter.y - cropY/2);
            const auto offsetXAugmented = offsetX - (cropCenter.x - cropX/2);
            const auto offsetYAugmented = offsetY - (cropCenter.y - cropY/2);
            auto width = cropX;
            auto height = cropY;
            if (offsetX + cropX > image.cols)
                width = image.cols - offsetX;
            if (offsetY + cropY > image.rows)
                height = image.rows - offsetY;
            if (offsetXAugmented + cropX > imageAugmented.cols)
                width = image.cols - offsetX;
            if (offsetYAugmented + cropY > imageAugmented.rows)
                height = image.rows - offsetY;
            if (width < 1 || height < 1)
                throw std::runtime_error{"width < 1 || height < 1!!!"
                                         + getLine(__LINE__, __FUNCTION__, __FILE__)};
            cv::Rect roiOrigin{offsetX, offsetY, width, height};
            cv::Rect roiAugmented{offsetXAugmented, offsetYAugmented, width, height};
            // Apply ROI copy
            image(roiOrigin).copyTo(imageAugmented(roiAugmented));
//             // Naive implementation (~x30 times slower)
//             cv::Mat imageAugmented2 = cv::Mat(cropY, cropX, image.type(), cv::Scalar{(double)defaultBorderValue});
//             if (imageAugmented2.type() == CV_8UC3)
//             {
//                 for (auto y = 0 ; y < cropY ; y++)
//                 {
//                     const int yOrigin = cropCenter.y - cropY/2 + y;
//                     for (auto x = 0 ; x < cropX ; x++)
//                     {
//                         const int xOrigin = cropCenter.x - cropX/2 + x;
//                         if (onPlane(cv::Point{xOrigin, yOrigin}, image.size()))
//                             imageAugmented2.at<cv::Vec3b>(y,x) = image.at<cv::Vec3b>(yOrigin, xOrigin);
//                     }
//                 }
//             }
//             else if (imageAugmented2.type() == CV_8UC1)
//             {
//                 for (auto y = 0 ; y < cropY ; y++)
//                 {
//                     const int yOrigin = cropCenter.y - cropY/2 + y;
//                     for (auto x = 0 ; x < cropX ; x++)
//                     {
//                         const int xOrigin = cropCenter.x - cropX/2 + x;
//                         if (onPlane(cv::Point{xOrigin, yOrigin}, image.size()))
//                             imageAugmented2.at<uchar>(y,x) = image.at<uchar>(yOrigin, xOrigin);
//                     }
//                 }
//             }
//             else if (imageAugmented2.type() == CV_16UC1)
//             {
//                 for (auto y = 0 ; y < cropY ; y++)
//                 {
//                     const int yOrigin = cropCenter.y - cropY/2 + y;
//                     for (auto x = 0 ; x < cropX ; x++)
//                     {
//                         const int xOrigin = cropCenter.x - cropX/2 + x;
//                         if (onPlane(cv::Point{xOrigin, yOrigin}, image.size()))
//                             imageAugmented2.at<uint16_t>(y,x) = image.at<uint16_t>(yOrigin, xOrigin);
//                     }
//                 }
//             }
//             else
//                 throw std::runtime_error{"Not implemented for image.type() == " + std::to_string(imageAugmented2.type())
//                                          + getLine(__LINE__, __FUNCTION__, __FILE__)};
//             if(cv::norm(imageAugmented - imageAugmented2) != 0)
//             {
//                 std::cout << "Norm = " << cv::norm(imageAugmented - imageAugmented2) << std::endl;
//                 cv::imwrite("imagename.png", imageAugmented);
//                 cv::imwrite("imagename2.png", imageAugmented2);
//                 throw std::runtime_error{"No 0 norm!!!" + getLine(__LINE__, __FUNCTION__, __FILE__)};
//             }
        }
    }

    void applyCrop(MetaData& metaData, const cv::Point2i& cropCenter,
                   const cv::Size& cropSize, const PoseModel poseModel)
    {
        // Update metaData
        const auto cropX = (int)cropSize.width;
        const auto cropY = (int)cropSize.height;
        const int offsetLeft = -(cropCenter.x - (cropX/2));
        const int offsetUp = -(cropCenter.y - (cropY/2));
        const cv::Point2f offsetPoint{(float)offsetLeft, (float)offsetUp};
        metaData.objPos += offsetPoint;
        for (auto& point : metaData.jointsSelf.points)
            point += offsetPoint;
        for (auto person = 0 ; person < metaData.numberOtherPeople ; person++)
        {
            if(metaData.objPosOthers.size()) metaData.objPosOthers[person] += offsetPoint;
            for (auto& point : metaData.jointsOthers[person].points)
                point += offsetPoint;
            if(metaData.jointsOthersPrev.size()){
                for (auto& point : metaData.jointsOthersPrev[person].points)
                    point += offsetPoint;
            }
        }
    }

    bool estimateFlip(const MetaData& metaData, const OPTransformationParameter& param_)
    {
        // Estimate random flip
        const auto dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        return (dice <= param_.flip_prob());
    }

    void applyFlip(cv::Mat& imageAugmented, const bool flip, const cv::Mat& image)
    {
        // Flip image
        if (flip && !image.empty())
            cv::flip(image, imageAugmented, 1);
        // No flip
        else if (imageAugmented.data != image.data)
            imageAugmented = image.clone();
    }

    void applyFlip(MetaData& metaData, const bool flip, const int imageWidth,
                   const OPTransformationParameter& param_, const PoseModel poseModel)
    {
        // Update metaData
        if (flip)
        {
            const auto widthMinusOne = imageWidth - 1;
            // Main keypoints
            flipKeypoints(metaData.jointsSelf, metaData.objPos, widthMinusOne, poseModel);
            // Other keypoints
            for (auto p = 0 ; p < metaData.numberOtherPeople ; p++){
                if(metaData.objPosOthers.size()){
                    flipKeypoints(metaData.jointsOthers[p], metaData.objPosOthers[p], widthMinusOne, poseModel);
                    if(metaData.jointsOthersPrev.size())
                    flipKeypoints(metaData.jointsOthersPrev[p], metaData.objPosOthers[p], widthMinusOne, poseModel);
                }else {
                    flipKeypoints(metaData.jointsOthers[p], widthMinusOne, poseModel);
                    if(metaData.jointsOthersPrev.size())
                    flipKeypoints(metaData.jointsOthersPrev[p], widthMinusOne, poseModel);
                }
            }
        }
    }

    void rotatePoint(cv::Point2f& point2f, const cv::Mat& R)
    {
        cv::Mat cvMatPoint(3,1, CV_64FC1);
        cvMatPoint.at<double>(0,0) = point2f.x;
        cvMatPoint.at<double>(1,0) = point2f.y;
        cvMatPoint.at<double>(2,0) = 1;
        const cv::Mat newPoint = R * cvMatPoint;
        point2f.x = newPoint.at<double>(0,0);
        point2f.y = newPoint.at<double>(1,0);
    }

    void applyAllAugmentation(cv::Mat& imageAugmented, const cv::Mat& rotationMatrix,
                              const float scale, const bool flip, const cv::Point2i& cropCenter,
                              const cv::Size& finalSize, const cv::Mat& image,
                              const unsigned char defaultBorderValue)
    {
        // Rotate image
        if (!image.empty())
        {
            // Final affine matrix equal to:
            // g_final = g_flip * g_crop * g_rot * g_scale
            // [1||-1, 0, 0||width-1]   [I [x;y]]   [R 0]   [s 0 0]            [sR [x;y]]   [+-sR11 +-sR12 x||(-x+w-1)]
            // [0,     1, 0         ] * [0 1]     * [0 1] * [0 s 0] = g_flip * [0   1]    = [sR21    sR22        y    ]
            // [0,     0, 1         ]                       [0 0 1]                         [  0      0          1    ]
            // Rotation + Scaling + Cropping
            cv::Mat matrix = rotationMatrix.clone();
            matrix.at<double>(0,0) *= scale;
            matrix.at<double>(0,1) *= scale;
            matrix.at<double>(0,2) -= (cropCenter.x - finalSize.width/2);
            // Flipping
            if (flip)
            {
                matrix.at<double>(0,0) *= -1;
                matrix.at<double>(0,1) *= -1;
                matrix.at<double>(0,2) = -matrix.at<double>(0,2) + finalSize.width-1;
            }
            matrix.at<double>(1,0) *= scale;
            matrix.at<double>(1,1) *= scale;
            matrix.at<double>(1,2) -= (cropCenter.y - finalSize.height/2);
            // Apply warping
            cv::warpAffine(image, imageAugmented, matrix, finalSize,
                           // (scale < 1 ? cv::INTER_AREA : cv::INTER_CUBIC),
                           cv::INTER_CUBIC, // CUBIC to consider rotations
                           cv::BORDER_CONSTANT, cv::Scalar{(double)defaultBorderValue});
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

    void clahe(cv::Mat& bgrImage, const int tileSize, const int clipLimit)
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
}  // namespace caffe
