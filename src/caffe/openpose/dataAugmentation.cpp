#include <iostream>
#include <fstream> // std::ifstream
#include <stdexcept> // std::runtime_error
#include <opencv2/opencv.hpp>
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

    void flipKeypoints(Joints& joints, cv::Point2f& objPos, const int numberBodyPAFParts, const int widthMinusOne,
                       const PoseModel poseModel)
    {
        objPos.x = widthMinusOne - objPos.x;
        for (auto part = 0 ; part < numberBodyPAFParts ; part++)
            joints.points[part].x = widthMinusOne - joints.points[part].x;
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

    void applyScale(cv::Mat& imageAugmented, const float scale, const cv::Mat& image)
    {
        // Scale image
        if (!image.empty())
            cv::resize(image, imageAugmented, cv::Size{}, scale, scale, cv::INTER_CUBIC);
        // Not used given that net makes x8 pooling anyway...
        //     // Image sharpening
        //     if (scale > 2.5 && imageAugmented.channels() == 3)
        //     {
        //         cv::Mat gaussianImage;
        //         cv::GaussianBlur(imageAugmented, gaussianImage, cv::Size(0, 0), 3);
        //         cv::addWeighted(imageAugmented, 1.5, gaussianImage, -0.5, 0, imageAugmented);
        //     }
    }

    void applyScale(MetaData& metaData, const float scale, const PoseModel poseModel)
    {
        // Update metaData
        metaData.objPos *= scale;
        metaData.scaleSelf *= scale;
        const auto numberBodyPAFParts = getNumberBodyAndPafChannels(poseModel);
        for (auto part = 0; part < numberBodyPAFParts ; part++)
            metaData.jointsSelf.points[part] *= scale;
        for (auto person=0; person<metaData.numberOtherPeople; person++)
        {
            metaData.objPosOthers[person] *= scale;
            metaData.scaleOthers[person] *= scale;
            for (auto part = 0; part < numberBodyPAFParts ; part++)
                metaData.jointsOthers[person].points[part] *= scale;
        }
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

    void applyRotation(cv::Mat& imageAugmented, const std::pair<cv::Mat, cv::Size> RotAndFinalSize,
                       const cv::Mat& image, const unsigned char defaultBorderValue)
    {
        // Rotate image
        if (!image.empty())
            cv::warpAffine(image, imageAugmented, RotAndFinalSize.first, RotAndFinalSize.second, cv::INTER_CUBIC,
                           cv::BORDER_CONSTANT, cv::Scalar{(double)defaultBorderValue});
    }

    void applyRotation(MetaData& metaData, const cv::Mat& Rot, const PoseModel poseModel)
    {
        // Update metaData
        rotatePoint(metaData.objPos, Rot);
        const auto numberBodyPAFParts = getNumberBodyAndPafChannels(poseModel);
        for (auto part = 0 ; part < numberBodyPAFParts ; part++)
            rotatePoint(metaData.jointsSelf.points[part], Rot);
        for (auto person = 0; person < metaData.numberOtherPeople; person++)
        {
            rotatePoint(metaData.objPosOthers[person], Rot);
            for (auto part = 0; part < numberBodyPAFParts ; part++)
                rotatePoint(metaData.jointsOthers[person].points[part], Rot);
        }
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
        const auto numberBodyPAFParts = getNumberBodyAndPafChannels(poseModel);
        for (auto part = 0 ; part < numberBodyPAFParts ; part++)
            metaData.jointsSelf.points[part] += offsetPoint;
        for (auto person = 0 ; person < metaData.numberOtherPeople ; person++)
        {
            metaData.objPosOthers[person] += offsetPoint;
            for (auto part = 0 ; part < numberBodyPAFParts ; part++)
                metaData.jointsOthers[person].points[part] += offsetPoint;
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
            const auto numberBodyPAFParts = getNumberBodyAndPafChannels(poseModel);
            const auto widthMinusOne = imageWidth - 1;
            // Main keypoints
            flipKeypoints(metaData.jointsSelf, metaData.objPos, numberBodyPAFParts, widthMinusOne, poseModel);
            // Other keypoints
            for (auto p = 0 ; p < metaData.numberOtherPeople ; p++)
                flipKeypoints(metaData.jointsOthers[p], metaData.objPosOthers[p], numberBodyPAFParts, widthMinusOne,
                              poseModel);
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
