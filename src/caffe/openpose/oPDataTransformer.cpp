#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
// OpenPose: added
// #include <opencv2/contrib/contrib.hpp>
// #include <opencv2/contrib/imgproc.hpp>
// #include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
// OpenPose: added end
#endif  // USE_OPENCV

// OpenPose: added
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
using namespace std;
// OpenPose: added end

namespace caffe {
// OpenPose: added ended
struct AugmentSelection
{
    bool flip = false;
    std::pair<cv::Mat, cv::Size> RotAndFinalSize;
    cv::Point2i cropCenter;
    float scale = 1.f;
    float rotation = 0.;
    cv::Size pointOffset;
};

void doOcclusions(cv::Mat& imageAugmented, cv::Mat& backgroundImageAugmented, const MetaData& metaData,
                  const unsigned int numberMaxOcclusions, const PoseModel poseModel)
{
    // For all visible keypoints --> [0, numberMaxOcclusions] oclusions
    // For 1/n visible keypoints --> [0, numberMaxOcclusions/n] oclusions
    const float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
    const auto numberBodyParts = getNumberBodyParts(poseModel);
    int detectedParts = 0;
    for (auto i = 0 ; i < numberBodyParts ; i++)
        if (metaData.jointsSelf.isVisible[i] < 1.5f)
            detectedParts++;
    const auto numberOcclusions = (int)std::round(numberMaxOcclusions * dice * detectedParts / numberBodyParts);
    if (numberOcclusions > 0)
    {
        for (auto i = 0 ; i < numberOcclusions ; i++)
        {
            // Select occluded part
            int occludedPart = -1;
            do
                occludedPart = std::rand() % numberBodyParts; // [0, #BP-1]
            while (metaData.jointsSelf.isVisible[occludedPart] > 1.5f);
            // Select random cropp around it
            const auto width = (int)std::round(imageAugmented.cols * metaData.scaleSelf/2
                                               * (1+(std::rand() % 1001 - 500)/1000.)); // +- [0.5-1.5] random
            const auto height = (int)std::round(imageAugmented.rows * metaData.scaleSelf/2
                                                * (1+(std::rand() % 1001 - 500)/1000.)); // +- [0.5-1.5] random
            const auto random = 1+(std::rand() % 1001 - 500)/500.; // +- [0-2] random
            // Estimate ROI rectangle to apply
            const auto point = metaData.jointsSelf.points[occludedPart];
            cv::Rect rectangle{(int)std::round(point.x - width/2*random),
                        (int)std::round(point.y - height/2*random), width, height};
            keepRoiInside(rectangle, imageAugmented.size());
            // Apply crop
            if (rectangle.area() > 0)
                backgroundImageAugmented(rectangle).copyTo(imageAugmented(rectangle));
        }
    }
}

void setLabel(cv::Mat& image, const std::string& label, const cv::Point& org)
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

void debugVisualize(const cv::Mat& image, const MetaData& metaData, const AugmentSelection& augmentSelection,
                    const PoseModel poseModel, const Phase& phase_, const OPTransformationParameter& param_)
{
    cv::Mat imageToVisualize = image.clone();

    cv::rectangle(imageToVisualize, metaData.objPos-cv::Point2f{3.f,3.f}, metaData.objPos+cv::Point2f{3.f,3.f},
                  cv::Scalar{255,255,0}, CV_FILLED);
    const auto numberBpPafChannels = getNumberBodyAndPafChannels(poseModel);
    for (auto part = 0 ; part < numberBpPafChannels ; part++)
    {
        const auto currentPoint = metaData.jointsSelf.points[part];
        // Hand case
        if (numberBpPafChannels == 21)
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
        else if (numberBpPafChannels == 9)
        {
            if (part==0 || part==1 || part==2 || part==6)
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{0,0,255}, -1);
            else if (part==3 || part==4 || part==5 || part==7)
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,0,0}, -1);
            else
                cv::circle(imageToVisualize, currentPoint, 3, cv::Scalar{255,255,0}, -1);
        }
        // Body case (CPM)
        else if (numberBpPafChannels == 14 || numberBpPafChannels == 28)
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

    cv::line(imageToVisualize, metaData.objPos + cv::Point2f{-368/2.f,-368/2.f},
             metaData.objPos + cv::Point2f{368/2.f,-368/2.f}, cv::Scalar{0,255,0}, 2);
    cv::line(imageToVisualize, metaData.objPos + cv::Point2f{368/2.f,-368/2.f},
             metaData.objPos + cv::Point2f{368/2.f,368/2.f}, cv::Scalar{0,255,0}, 2);
    cv::line(imageToVisualize, metaData.objPos + cv::Point2f{368/2.f,368/2.f},
             metaData.objPos + cv::Point2f{-368/2.f,368/2.f}, cv::Scalar{0,255,0}, 2);
    cv::line(imageToVisualize, metaData.objPos + cv::Point2f{-368/2.f,368/2.f},
             metaData.objPos + cv::Point2f{-368/2.f,-368/2.f}, cv::Scalar{0,255,0}, 2);

    for (auto person=0;person<metaData.numberOtherPeople;person++)
    {
        cv::rectangle(imageToVisualize,
                      metaData.objPosOthers[person]-cv::Point2f{3.f,3.f},
                      metaData.objPosOthers[person]+cv::Point2f{3.f,3.f}, cv::Scalar{0,255,255}, CV_FILLED);
        for (auto part = 0 ; part < numberBpPafChannels ; part++)
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

        sprintf(imagename, "visualize/augment_%04d_epoch_%03d_writenum_%03d.jpg", metaData.writeNumber,
                metaData.epoch, metaData.writeNumber);
    }
    else
    {
        const std::string stringInfo = "no augmentation for testing";
        setLabel(imageToVisualize, stringInfo, cv::Point{0, 20});

        sprintf(imagename, "visualize/augment_%04d.jpg", metaData.writeNumber);
    }
    //LOG(INFO) << "filename is " << imagename;
    cv::imwrite(imagename, imageToVisualize);
}
template<typename Dtype>
int getType(Dtype dtype)
{
    (void)dtype;
    if (sizeof(Dtype) == sizeof(float))
        return CV_32F;
    else if (sizeof(Dtype) == sizeof(double))
        return CV_64F;
    else
    {
        throw std::runtime_error{"Only float or double" + getLine(__LINE__, __FUNCTION__, __FILE__)};
        return CV_32F;
    }
}
// OpenPose: added ended

template<typename Dtype>
OPDataTransformer<Dtype>::OPDataTransformer(const std::string& modelString){
    LOG(INFO) << "OPDataTransformer constructor done.";
    // PoseModel
    std::tie(mPoseModel, mPoseCategory) = flagsToPoseModel(modelString);
    mModelString = modelString;
    srand(time(NULL));
}

template<typename Dtype>
OPDataTransformer<Dtype>::OPDataTransformer(const OPTransformationParameter& param) // OpenPose: Added std::string
// : param_(param), phase_(phase) {
    : param_(param), mCurrentEpoch{-1} {

    LOG(INFO) << "OPDataTransformer constructor done.";
    // PoseModel
    std::tie(mPoseModel, mPoseCategory) = flagsToPoseModel(param_.model());
    mModelString = param_.model();
    srand(time(NULL));
    // OpenPose: added end
}

template<typename Dtype>
OPDataTransformer<Dtype>::OPDataTransformer(const OPTransformationParameter& param,
                                            Phase phase, const std::string& modelString, int tpaf, int staf, std::vector<int> stafIDS) // OpenPose: Added std::string
// : param_(param), phase_(phase) {
    : param_(param), phase_(phase), mCurrentEpoch{-1} {
    // OpenPose: commented
    // // check if we want to use mean_file
    // if (param_.has_mean_file()) {
    //     CHECK_EQ(param_.mean_value_size(), 0) <<
    //         "Cannot specify mean_file and mean_value at the same time";
    //     const std::string& mean_file = param.mean_file();
    //     if (Caffe::root_solver()) {
    //         LOG(INFO) << "Loading mean file from: " << mean_file;
    //     }
    //     BlobProto blob_proto;
    //     ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    //     data_mean_.FromProto(blob_proto);
    // }
    // // check if we want to use mean_value
    // if (param_.mean_value_size() > 0) {
    //     CHECK(param_.has_mean_file() == false) <<
    //         "Cannot specify mean_file and mean_value at the same time";
    //     for (int c = 0; c < param_.mean_value_size(); ++c) {
    //         mean_values_.push_back(param_.mean_value(c));
    //     }
    // }
    // OpenPose: commented end
    // OpenPose: added
    mTpaf = tpaf;
    mStaf = staf;
    mStafIDS = stafIDS;
    LOG(INFO) << "OPDataTransformer constructor done.";
    // PoseModel
    std::tie(mPoseModel, mPoseCategory) = flagsToPoseModel(modelString);
    mModelString = modelString;
    srand(time(NULL));
    // OpenPose: added end
}

// OpenPose: commented
// template <typename Dtype>
// void OPDataTransformer<Dtype>::InitRand() {
//     const bool needs_rand = param_.mirror() ||
//             (phase_ == TRAIN && param_.crop_size());
//     if (needs_rand)
//     {
//         const unsigned int rng_seed = caffe_rng_rand();
//         rng_.reset(new Caffe::RNG(rng_seed));
//     }
//     else
//         rng_.reset();
// }

int getRand(int min, int max){
    int randNum = rand()%(max-min + 1) + min;
    return randNum;
}

std::string getCurrDir(){
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL)
        return std::string(cwd);
    else
        throw std::runtime_error("getCurrDir Error");
}

template<typename Dtype>
void vizDebug(const cv::Mat& imageAugmented, const MetaData& metaData, const Dtype* transformedLabel, const int rezX, const int rezY, const int gridX, const int gridY, const int stride, const PoseModel mPoseModel, const std::string mModelString, const int numberTotalChannels, bool basic=false, std::string fname="visualize"){

    std::string vizDir = getCurrDir() + "/" + fname;
    std::string rmCommand = "rm -rf " + vizDir;
    std::string mkdirCommand = "mkdir " + vizDir;
    system(rmCommand.c_str());
    system(mkdirCommand.c_str());

    // Write metadata
    cv::Mat imageAugCloned = imageAugmented.clone();
    int i=0;
    for(cv::Point2f p : metaData.jointsSelf.points){
        if(metaData.jointsSelf.isVisible[i] <= 1){
        cv::circle(imageAugCloned, p, 3, cv::Scalar(25,255,255),CV_FILLED);
        cv::putText(imageAugCloned, std::to_string(i), p, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255), 1);
        }
        i++;

        int j=0;
        if(metaData.jointsSelfPrev.points.size()){
            for(cv::Point2f px : metaData.jointsSelfPrev.points){
                cv::circle(imageAugCloned, px, 3, cv::Scalar(25,25,255),CV_FILLED);
                if(metaData.jointsSelf.points.size()){
                    if(metaData.jointsSelf.isVisible[j] <= 1 && metaData.jointsSelfPrev.isVisible[j] <= 1)
                        cv::line(imageAugCloned, px, metaData.jointsSelf.points[j], cv::Scalar(25,25,255));
                }
                //i++;
            }
        }
    }
    int pid=0;
    for(const Joints& j : metaData.jointsOthers){
        int i=0;
        for(cv::Point2f p : j.points){
            if(j.isVisible[i] <= 1){
            cv::circle(imageAugCloned, p, 3, cv::Scalar(25,255,255),CV_FILLED);
            cv::putText(imageAugCloned, std::to_string(i), p, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255), 1);
            }
            i++;
        }

        i=0;
        if(metaData.jointsOthersPrev.size()){
            for(cv::Point2f p : metaData.jointsOthersPrev[pid].points){
                cv::circle(imageAugCloned, p, 3, cv::Scalar(25,25,255),CV_FILLED);
                if(metaData.jointsOthers[pid].points.size()){
                    if(metaData.jointsOthers[pid].isVisible[i] <= 1 && metaData.jointsOthersPrev[pid].isVisible[i] <= 1)
                        cv::line(imageAugCloned, p, metaData.jointsOthers[pid].points[i], cv::Scalar(25,25,255));
                }
                i++;
            }
        }

        pid++;
    }

    if(basic){
        std::cout << vizDir + "/basic.png" << std::endl;
        cv::imwrite(vizDir + "/basic.png", imageAugCloned);
        return;
    }

    // 1. Create `visualize` folder in training folder (where train_pose.sh is located)
    // 2. Comment the following if statement
    const auto channelOffset = gridY * gridX;
    for (auto part = 0; part < numberTotalChannels; part++)
    {
        // Reduce #images saved (ideally mask images should be the same)
        // if (part < 1)
        //                 if (part == numberTotalChannels-1)
        // if (part < 3 || part >= numberTotalChannels - 3)
        {
            cv::Mat finalImage = cv::Mat::zeros(gridY, 2*gridX, CV_8UC1);
            for (auto subPart = 0; subPart < 2; subPart++)
            {
                cv::Mat labelMap = finalImage(cv::Rect{subPart*gridX, 0, gridX, gridY});
                for (auto gY = 0; gY < gridY; gY++)
                {
                    const auto yOffset = gY*gridX;
                    for (auto gX = 0; gX < gridX; gX++)
                    {
                        const auto channelIndex = (part+numberTotalChannels*subPart)*channelOffset;
                        labelMap.at<uchar>(gY,gX) = (int)(255.*transformedLabel[channelIndex + yOffset + gX]);
                    }
                }
            }
            cv::resize(finalImage, finalImage, cv::Size{}, stride, stride, cv::INTER_LINEAR);
            cv::applyColorMap(finalImage, finalImage, cv::COLORMAP_JET);
            for (auto subPart = 0; subPart < 2; subPart++)
            {
                cv::Mat labelMap = finalImage(cv::Rect{subPart*rezX, 0, rezX, rezY});
                cv::addWeighted(labelMap, 0.5, imageAugCloned, 0.5, 0.0, labelMap);
            }
            // Write on disk
            char imagename [200];
            sprintf(imagename, "%s/%s_augment_%04d_label_part_%02d.jpg", vizDir.c_str(), mModelString.c_str(),
                    metaData.writeNumber, part);
            cv::imwrite(imagename, finalImage);
        }
    }
}

template<typename Dtype>
void matToCaffe(Dtype* caffeImg, const cv::Mat& imgAug){
    const int imageAugmentedArea = imgAug.rows * imgAug.cols;
    auto* uCharPtrCvMat = (unsigned char*)(imgAug.data);
    //caffeImg = new Dtype[imgAug.channels()*imgAug.size().width*imgAug.size().height];
    for (auto y = 0; y < imgAug.rows; y++)
    {
        const auto yOffset = y*imgAug.cols;
        for (auto x = 0; x < imgAug.cols; x++)
        {
            const auto xyOffset = yOffset + x;
            // const cv::Vec3b& bgr = imageAugmented.at<cv::Vec3b>(y, x);
            auto* bgr = &uCharPtrCvMat[3*xyOffset];
            caffeImg[xyOffset] = (bgr[0] - 128) / 256.0;
            caffeImg[xyOffset + imageAugmentedArea] = (bgr[1] - 128) / 256.0;
            caffeImg[xyOffset + 2*imageAugmentedArea] = (bgr[2] - 128) / 256.0;
        }
    }
}

template<typename Dtype>
void caffeToMat(cv::Mat& img, const Dtype* caffeImg, cv::Size imageSize){
    // Need a function to convert back
    img = cv::Mat(imageSize, CV_8UC3);
    const int imageAugmentedArea = img.rows * img.cols;
    auto* imgPtr = (unsigned char*)(img.data);
    for (auto y = 0; y < img.rows; y++)
    {
        const auto yOffset = y*img.cols;
        for (auto x = 0; x < img.cols; x++)
        {
            const auto xyOffset = yOffset + x;
            auto* bgr = &imgPtr[3*xyOffset];
            bgr[0] = (caffeImg[xyOffset]*256.) + 128;
            bgr[1] = (caffeImg[xyOffset + imageAugmentedArea]*256.) + 128;
            bgr[2] = (caffeImg[xyOffset + 2*imageAugmentedArea]*256.) + 128;
        }
    }
}

void maskBetween(const cv::Point2i& a, const cv::Point2i& b, cv::Mat& maskMiss, cv::Mat& img, float sscale){
    if((a.x >= 0 && a.x < img.size().width && a.y >= 0 && a.y < img.size().height)
            && (b.x >= 0 && b.x < img.size().width && b.y >= 0 && b.y < img.size().height)){
        float x1 = a.x; float x2 = b.x;
        float y1 = a.y; float y2 = b.y;
        float xc = (x1 + x2)/2  ;  float yc = (y1 + y2)/2  ;    // Center point
        float xd = (x1 - x2)/2  ;  float yd = (y1 - y2)/2  ;    // Half-diagonal
        float x3 = xc - yd  ;  float y3 = yc + xd;              // Third corner
        float x4 = xc + yd  ;  float y4 = yc - xd;              // Fourth corner
        float dist = sqrt(pow(x2-x1,2)+pow(y2-y1,2));
        cv::Point2i rp(x3,y3);
        cv::Point2i lp(x4,y4);
        cv::Point2i v(rp.x-lp.x, rp.y-lp.y);
        float iscale = (float)img.size().width/(float)maskMiss.size().width;
        cv::Point2i rpn(rp.x - v.x*sscale, rp.y - v.y*sscale);
        cv::Point2i lpn(lp.x + v.x*sscale, lp.y + v.y*sscale);
        cv::line(maskMiss, cv::Point2i(lpn.x/iscale, lpn.y/iscale), cv::Point2i(rpn.x/iscale, rpn.y/iscale), cv::Scalar(0), dist/2/iscale);
        cv::line(img, cv::Point2i(lpn.x/1, lpn.y/1), cv::Point2i(rpn.x/1, rpn.y/1), cv::Scalar(255,255,255), dist/2/1);
    }
}

void maskFaceMPII(cv::Mat& maskMiss, const std::vector<float>& isVisible, const std::vector<cv::Point2f>& points, cv::Mat& img){
    // Mask Face
    if(isVisible[20] == 2 || isVisible[19] == 2) return;
    cv::Point topPoint = points[20];
    cv::Point neckPoint = points[19];
    maskBetween(topPoint, neckPoint, maskMiss, img, 0.7);
    // maybe handle if top missing ?

}

void maskRealNeckCOCO(cv::Mat& maskMiss, const std::vector<float>& isVisible, const std::vector<cv::Point2f>& points, cv::Mat& img){
    if(isVisible[0] == 2 || isVisible[1] == 2) return;
    cv::Point nosePoint = points[0];
    cv::Point fakeNeckPoint = points[1];
    maskBetween(nosePoint, fakeNeckPoint, maskMiss, img, 0.5);
}

float l2(cv::Point& p, cv::Point& q) {
    cv::Point diff = p - q;
    return sqrt(diff.x*diff.x + diff.y*diff.y);
}

void maskFaceCOCO(cv::Mat& maskMiss, const std::vector<float>& isVisible, const std::vector<cv::Point2f>& points, cv::Mat& img)
{
    int neckIndex = 1;
    int noseIndex = 0;
    int lEarIndex = 18;
    int rEarIndex = 19;
    int lEyeIndex = 16;
    int rEyeIndex = 17;
    float threshold = 0.25f;
    bool neckVisible = (isVisible[neckIndex] < 2);
    bool noseVisible = (isVisible[noseIndex] < 2);
    bool lEarVisible = (isVisible[lEarIndex] < 2);
    bool rEarVisible = (isVisible[rEarIndex] < 2);
    bool lEyeVisible = (isVisible[lEyeIndex] < 2);
    bool rEyeVisible = (isVisible[rEyeIndex] < 2);
    cv::Point neckPoint = points[neckIndex];
    cv::Point nosePoint = points[noseIndex];
    cv::Point lEarPoint = points[lEarIndex];
    cv::Point rEarPoint = points[rEarIndex];
    cv::Point lEyePoint = points[lEyeIndex];
    cv::Point rEyePoint = points[rEyeIndex];

    cv::Point pointTopLeft(0,0);
    float faceSize = 0.;
    int counter = 0;

    // Neck and Nose Visible
    if(noseVisible && neckVisible){
        // Only Left Eye and Ear Visible
        if(lEyeVisible && lEarVisible && !rEyeVisible && !rEarVisible){
            pointTopLeft.x += (lEyePoint.x + lEarPoint.x + nosePoint.x) / 3.f;
            pointTopLeft.y += (lEyePoint.y + lEarPoint.y + nosePoint.y) / 3.f;
            faceSize += 0.85f * l2(nosePoint, lEyePoint) + l2(nosePoint, lEarPoint) + l2(nosePoint, neckPoint);
        }
        // Only Right Eye and Ear Visible
        else if(rEyeVisible && rEarVisible && !lEyeVisible && !lEarVisible){
            pointTopLeft.x += (rEyePoint.x + rEarPoint.x + nosePoint.x) / 3.f;
            pointTopLeft.y += (rEyePoint.y + rEarPoint.y + nosePoint.y) / 3.f;
            faceSize += 0.85f * l2(nosePoint, rEyePoint) + l2(nosePoint, rEarPoint) + l2(nosePoint, neckPoint);
        }
        // Neck and Nose only
        else{
            pointTopLeft.x += (neckPoint.x + nosePoint.x) / 2.f;
            pointTopLeft.y += (neckPoint.y + nosePoint.y) / 2.f;
            faceSize += 2.f * l2(neckPoint, nosePoint);
        }
        counter++;
    }
    // LEye and REye
    if(lEyeVisible && rEyeVisible){
        pointTopLeft.x += (lEyePoint.x + rEyePoint.x) / 2.f;
        pointTopLeft.y += (lEyePoint.y + rEyePoint.y) / 2.f;
        faceSize += 3.f * l2(lEyePoint, rEyePoint);
        counter++;
    }
    // LEar and REar
    if(lEarVisible && rEarVisible){
        pointTopLeft.x += (lEarPoint.x + rEarPoint.x) / 2.f;
        pointTopLeft.y += (lEarPoint.y + rEarPoint.y) / 2.f;
        faceSize += 2.f * l2(lEarPoint, rEarPoint);
        counter++;
    }
    if (counter > 0)
    {
        pointTopLeft.x =  pointTopLeft.x / (float)counter;
        pointTopLeft.y =  pointTopLeft.y / (float)counter;
        faceSize /= counter;
    }else{
        return;
    }

    cv::Rect bigRect(pointTopLeft.x - faceSize / 2, pointTopLeft.y - faceSize / 2, faceSize, faceSize);
    float iscale = (float)img.size().width/(float)maskMiss.size().width;
    cv::Rect smallRect(bigRect.x/iscale,bigRect.y/iscale,bigRect.width/iscale,bigRect.height/iscale);

    float xScale = 0.5;
    float yScale = 1.0;
    cv::Point a = cv::Point(bigRect.x,bigRect.y);
    cv::Point b = cv::Point(bigRect.x+bigRect.width,bigRect.y+bigRect.height);
    cv::Point centroid = cv::Point((a.x+b.x)/2,(a.y+b.y)/2);
    cv::Point an = cv::Point(((a.x-centroid.x)*xScale)+centroid.x,((a.y-centroid.y)*yScale)+centroid.y);
    cv::Rect dropRect(an.x, an.y, bigRect.width*xScale, (bigRect.height*yScale)/4);

    // Eye ear check
    if(!rEarVisible) rEarPoint.y = std::numeric_limits<int>::max();
    if(!lEarVisible) lEarPoint.y = std::numeric_limits<int>::max();
    if(!rEyeVisible) rEyePoint.y = std::numeric_limits<int>::max();
    if(!lEyeVisible) lEyePoint.y = std::numeric_limits<int>::max();
    if(lEarVisible || rEarVisible || lEyeVisible || rEyeVisible){
        int minPixY = std::min(lEarPoint.y, std::min(rEarPoint.y, std::min(lEyePoint.y, rEyePoint.y)));
        minPixY-=5;
        if(dropRect.y < minPixY)
            dropRect.height = minPixY - dropRect.y;
    }

    cv::Rect dropRectScaled(dropRect.x/iscale,dropRect.y/iscale,dropRect.width/iscale,dropRect.height/iscale);
    cv::rectangle(img, dropRect, cv::Scalar(255,0,0));
    cv::rectangle(maskMiss, dropRectScaled, cv::Scalar(0), CV_FILLED);
}

template<typename T>
bool contains(vector<T> v, T x)
{
      if (v.empty())
           return false;
      if (find(v.begin(), v.end(), x) != v.end())
           return true;
      else
           return false;
}

// OpenPose: added
template<typename Dtype>
cv::Mat OPDataTransformer<Dtype>::opConvert(const cv::Mat &img, const cv::Mat &bg, std::vector<cv::Rect> &rects){

    // Data
    const auto stride = (int)param_.stride();
    const cv::Size finalCropSize{(int)param_.crop_size_x(), (int)param_.crop_size_y()};
    const auto finalImageWidth = (int)param_.crop_size_x();
    const auto finalImageHeight = (int)param_.crop_size_y();

    // Force resize to 1280?
    float ratio = (float)img.size().width / (float)img.size().height;
    cv::Mat convImg;
    cv::resize(img, convImg, cv::Size(720*ratio,720));
    float xChange = (float)convImg.size().width / (float)img.size().width;
    float yChange = (float)convImg.size().height / (float)img.size().height;
    // Resize the rects also
    for(cv::Rect& rect : rects){
        rect.x *= xChange;
        rect.y *= yChange;
        rect.width *= xChange;
        rect.height *= yChange;
    }

    // Setup a metaData object
    MetaData metaData;
    metaData.imageSize = convImg.size();
    metaData.isValidation = false;
    metaData.numberOtherPeople = rects.size();
    int selectedRect = getRand(0, rects.size()-1);
    //metaData.objPos.x = rects[selectedRect].x + (rects[selectedRect].width/2);
    //metaData.objPos.y = rects[selectedRect].y + (rects[selectedRect].height/2);
    metaData.objPos.x = metaData.imageSize.width/2.; metaData.objPos.y = metaData.imageSize.height/2; // Set to rotate around centre
    metaData.jointsOthers.resize(rects.size());
    metaData.scaleSelf = 1;

    // Store Rect as joints
    int i=0;
    for(cv::Rect& rect : rects){
        //metaData.jointsOthers[i].isVisible.emplace_back(1);
        //metaData.jointsOthers[i].isVisible.emplace_back(1);
        cv::Point2f a = rect.tl();
        cv::Point2f b = cv::Point(a.x + rect.width, a.y);
        cv::Point2f c = rect.br();
        cv::Point2f d = cv::Point(c.x - rect.width, c.y);
        metaData.jointsOthers[i].points.emplace_back(a);
        metaData.jointsOthers[i].points.emplace_back(b);
        metaData.jointsOthers[i].points.emplace_back(c);
        metaData.jointsOthers[i].points.emplace_back(d);
        i++;
    }

    // Augment
    cv::Mat imgAug = convImg.clone();
    AugmentSelection augmentSelection;
    augmentSelection.scale = estimateScale(metaData, param_);
    applyScale(metaData, augmentSelection.scale, mPoseModel);
    augmentSelection.RotAndFinalSize = estimateRotation(
                    metaData,
                    cv::Size{(int)std::round(metaData.imageSize.width * augmentSelection.scale),
                             (int)std::round(metaData.imageSize.height * augmentSelection.scale)},
                    param_);
    applyRotation(metaData, augmentSelection.RotAndFinalSize.first, mPoseModel);
    augmentSelection.cropCenter = estimateCrop(metaData, param_);
    applyCrop(metaData, augmentSelection.cropCenter, finalCropSize, mPoseModel);
    augmentSelection.flip = estimateFlip(metaData, param_);
    augmentSelection.flip = 1;
    applyFlip(metaData, augmentSelection.flip, finalImageWidth, param_, mPoseModel, false);
    applyAllAugmentation(imgAug, augmentSelection.RotAndFinalSize.first, augmentSelection.scale,
                             augmentSelection.flip, augmentSelection.cropCenter, finalCropSize, convImg, 0);

    // Background
    if(!bg.empty()){
        cv::Mat backgroundImage;
        applyCrop(backgroundImage, {finalImageWidth/2, finalImageHeight/2}, bg, 0, finalCropSize);

        cv::Mat maskImg(convImg.size(), CV_8UC1, cv::Scalar(255));

        cv::Mat maskImgAug;
        applyAllAugmentation(maskImgAug, augmentSelection.RotAndFinalSize.first, augmentSelection.scale,
                                 augmentSelection.flip, augmentSelection.cropCenter, finalCropSize, maskImg, 0);
        cv::bitwise_not(maskImgAug, maskImgAug);

        cv::Mat finalImg;
        cv::add(imgAug, backgroundImage, imgAug, maskImgAug);
    }

    // Modify Rect back
    i = 0;
    for(cv::Rect& rect : rects){
        rect = cv::boundingRect(metaData.jointsOthers[i].points);
        i++;
    }

    return imgAug;
}

// OpenPose: added
template<typename Dtype>
void OPDataTransformer<Dtype>::TransformVideoJSON(int vid, int frames, VSeq& vs, Blob<Dtype>* transformedData, Blob<Dtype>* transformedLabel,
                                                  const Datum& datum, const Datum* datumNegative)
{
    // Secuirty checks
    const int datumChannels = datum.channels();
    const int imageNum = transformedData->num();
    const int imageChannels = transformedData->channels();
    const int labelNum = transformedLabel->num();
    //cout << datumChannels << " " << imageNum << " " << imageChannels << " " << labelNum << endl;

    // [31 x 20000 x 1]
    //cout << datum.channels() << " " << datum.height() << " " << datum.width() << " " << labelNum << endl;

    // Skip Vector
    std::vector<int> skip;

    // Load json data
    std::vector<Json::Value> jsonVideoData;
    const std::string& data = datum.data();
    for(int i=0; i<datum.channels(); i++){
        std::string jsonString(datum.height(),' ');
        for(int j=0; j<datum.height(); j++){
            jsonString[j] = data[i*datum.height()*datum.width() + j];
        }

        // Load string
        Json::Value root;
        Json::Reader reader;
        bool parsingSuccessful = reader.parse( jsonString.c_str(), root );     //parse process
        if ( !parsingSuccessful )
        {
            std::cout  << "Failed to parse" << reader.getFormattedErrorMessages();
            throw;
        }
        jsonVideoData.emplace_back(root);

        if(root.isMember("skip")){
            if(i==0 && root["skip"].size())
            {
                for(int f=0; f<root["skip"].size(); f++){
                    skip.push_back(root["skip"][f].asInt());
                }
            }
        }
    }

    // Sample for flip
    bool rev = 1;
    rev = getRand(0,1);

    // Sample for step frames
    int step = 1;
    if(!skip.size()) step = getRand(1,2);

    // Sample the start frame (We can timestep too) NOT DONE!!! Try to do cache also
    int startIndex = getRand(0,datum.channels()-(frames*step)-1);

    // Dont train on Skip frames
    int skipCounter = 0;
    if(skip.size()){
        while(true){
            bool mustSkip = false;
            for(int i=startIndex; i<startIndex+(frames*step); i+=step){
                if(contains(skip, i)) mustSkip = true;
            }
            if(mustSkip){
                startIndex = getRand(0,datum.channels()-(frames*step)-1);
            }else break;
            skipCounter++;
            if(skipCounter > 500){
                std::cout << "BAD" << std::endl;
                break;
            }
        }
    }

    vs.images.clear();
    vs.masks.clear();
    vs.jsons.clear();
    for(int i=startIndex; i<startIndex+(frames*step); i+=step){
        //cout << "Looking at frame: " << i << endl;
        vs.images.emplace_back(cv::imread(jsonVideoData[i]["image_path_full"].asString()));
        vs.masks.emplace_back(cv::imread(jsonVideoData[i]["mask_path_full"].asString(),0));
        vs.jsons.emplace_back(jsonVideoData[i]);
    }
    //cout << "Loaded Images" << endl;

    if(rev){
        std::reverse(vs.images.begin(), vs.images.end());
        std::reverse(vs.masks.begin(), vs.masks.end());
        std::reverse(vs.jsons.begin(), vs.jsons.end());
    }

    // Sample person to focus?
    std::vector<int> trackIds;
    if(jsonVideoData[0]["annorect"].size()){
        if(jsonVideoData[0]["annorect"][0].isMember("track_id")){
            for(int i=0; i<jsonVideoData[0]["annorect"].size(); i++){
                int track_id = jsonVideoData[0]["annorect"][i]["track_id"].asInt();
                trackIds.emplace_back(track_id);
            }
        }
    }
    // First select a possible track
    std::vector<cv::Rect> bboxes;
    if(trackIds.size()){
        std::random_shuffle(trackIds.begin(), trackIds.end());
        int selected_track = trackIds[0];
        // Iterate each frame
        for(int i=0; i<jsonVideoData.size(); i++){
            // If frame has people
            if(jsonVideoData[i]["annorect"].size()){
                // Iterate each person and select bbox for that id
                bool found = false;
                cv::Rect bbox;
                for(int j=0; j<jsonVideoData[i]["annorect"].size(); j++){
                    int track_id = jsonVideoData[i]["annorect"][j]["track_id"].asInt();
                    if(track_id == selected_track){
                        found = true;
                        bbox.x = jsonVideoData[i]["annorect"][j]["rect"][0].asInt();
                        bbox.y = jsonVideoData[i]["annorect"][j]["rect"][1].asInt();
                        bbox.width = jsonVideoData[i]["annorect"][j]["rect"][2].asInt() - bbox.x;
                        bbox.height = jsonVideoData[i]["annorect"][j]["rect"][3].asInt() - bbox.y;
                        break;
                    }
                }
                // Person found for this frame, push back
                if(found) bboxes.emplace_back(bbox);
                // Person not found for this frame
                else{
                    // Take last if available
                    if(bboxes.size()) bboxes.emplace_back(bboxes.back());
                    // If not
                    bboxes.emplace_back(cv::Rect(0,0,0,0));
                }
            }
            else{
                throw std::runtime_error("Handle this");
            }
        }
    }
    for(cv::Rect& r : bboxes){
        if(r.x == 0 && r.y == 0){
            bboxes.clear();
            break;
        }
    }
    int sampleToFocus = getRand(0,1);
    if(!sampleToFocus) bboxes.clear();

    // CHECK THIS PART MAKE SURE IT IS ALSO IMAGE CENTRERD!

    // Params
    //const auto rezX = (int)metaData.imageSize.width;
    //const auto rezY = (int)metaData.imageSize.height;
    const auto stride = (int)param_.stride();
    const cv::Size finalCropSize{(int)param_.crop_size_x(), (int)param_.crop_size_y()};
    const auto finalImageWidth = (int)param_.crop_size_x();
    const auto finalImageHeight = (int)param_.crop_size_y();
    const auto gridX = finalImageWidth / stride;
    const auto gridY = finalImageHeight / stride;

    // Read background image
    cv::Mat backgroundImage;
    cv::Mat maskBackgroundImage = (datumNegative != nullptr
            ? cv::Mat(vs.images[0].size().height, vs.images[0].size().width, CV_8UC1, cv::Scalar{0}) : cv::Mat());
    if (datumNegative != nullptr)
    {
        const std::string& data = datumNegative->data();
        const int datumNegativeWidth = datumNegative->width();
        const int datumNegativeHeight = datumNegative->height();
        const auto datumNegativeArea = (int)(datumNegativeHeight * datumNegativeWidth);
        const cv::Mat b(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[0]);
        const cv::Mat g(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[datumNegativeArea]);
        const cv::Mat r(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[2*datumNegativeArea]);
        std::vector<cv::Mat> bgr = {b,g,r};
        cv::merge(bgr, backgroundImage);
        if (datumNegativeWidth > finalImageWidth && datumNegativeHeight > finalImageHeight)
        {
            const auto xDiff = datumNegativeWidth - finalImageWidth;
            const auto yDiff = datumNegativeHeight - finalImageHeight;
            const auto minX = (xDiff <= 0 ? 0 :
                                            (int)std::round(xDiff * float(std::rand()) / float(RAND_MAX)) // [0,1]
                                            );
            const auto minY = (xDiff <= 0 ? 0 :
                                            (int)std::round(yDiff * float(std::rand()) / float(RAND_MAX)) // [0,1]
                                            );
            cv::Mat backgroundImageTemp;
            std::swap(backgroundImage, backgroundImageTemp);
            const cv::Point2i backgroundCropCenter{minX + finalImageWidth/2, minY + finalImageHeight/2};
            applyCrop(backgroundImage, backgroundCropCenter, backgroundImageTemp, 0, finalCropSize);
        }
        // Resize (if smaller than final crop size)
        // if (datumNegativeWidth < finalImageWidth || datumNegativeHeight < finalImageHeight)
        else
        {
            cv::Mat backgroundImageTemp;
            std::swap(backgroundImage, backgroundImageTemp);
            cv::resize(backgroundImageTemp, backgroundImage, cv::Size{finalImageWidth, finalImageHeight}, 0, 0, CV_INTER_CUBIC);
        }
    }

    // Convert format
    std::map<int, Joints> personIDJointsLast;
    cv::Mat imgAugPrev;
    AugmentSelection augmentSelection;
    for(int i=0; i<frames; i++){
        const Json::Value& json = vs.jsons[i];
        const cv::Mat& img = vs.images[i];
        const cv::Mat& mask = vs.masks[i];

        // Load metadata
        MetaData metaData;
        metaData.imageSize = vs.images[i].size();
        metaData.isValidation = false;
        metaData.numberOtherPeople = json["annorect"].size();
        metaData.writeNumber = json["image_id"].asInt();
        metaData.totalWriteNumber = 0; // WTF IS THIS IT USED BY THE DIAGONAL FUNCTION
        //metaData.objPos.x = 368/2; metaData.objPos.y = 368/2;
        if(bboxes.size()){
            metaData.objPos.x = bboxes[i].x + (bboxes[i].width/2);
            metaData.objPos.y = bboxes[i].y + (bboxes[i].height/2);
        }else
            metaData.objPos.x = metaData.imageSize.width/2.; metaData.objPos.y = metaData.imageSize.height/2; // Set to rotate around centre
        metaData.jointsOthers.resize(metaData.numberOtherPeople);
        metaData.scaleSelf = 1;

        // Handle prev case
        int j=0;
        metaData.jointsOthersPrev.resize(metaData.jointsOthers.size());
        for(Joints& joints : metaData.jointsOthers){
            int personID = json["annorect"][j]["track_id"].asInt();
            if(personIDJointsLast.count(personID)){
                metaData.jointsOthersPrev[j] = personIDJointsLast[personID].clone();
            }
            j++;
        }

        // Iterate each person
        j=0;
        personIDJointsLast.clear();
        for(Joints& joints : metaData.jointsOthers){
            // Can i just load it in direct HARDCODED TO 21
            joints.points.resize(21);
            joints.isVisible.resize(21);
            for(int m=0; m<21; m++){
                joints.points[m].x = json["annorect"][j]["keypoints"][m*3 + 0].asFloat();
                joints.points[m].y = json["annorect"][j]["keypoints"][m*3 + 1].asFloat();
                joints.isVisible[m] = json["annorect"][j]["keypoints"][m*3 + 2].asFloat();
                joints.isVisible[m] = ((int)joints.isVisible[m] + 2) % 3;
            }
            // Store Person IDS
            personIDJointsLast[json["annorect"][j]["track_id"].asInt()] = joints.clone();
            j++;
        }
        for (auto& joints : metaData.jointsOthers)
            lmdbJointsToOurModel(joints, mPoseModel);
        for (auto& joints : metaData.jointsOthersPrev){
            if(joints.points.size())
            lmdbJointsToOurModel(joints, mPoseModel);
        }

        // Augment here
        cv::Mat imgAug, maskAug, maskBgAug, bgImgAug;
        if(i==0) augmentSelection.scale = estimateScale(metaData, param_);
        applyScale(metaData, augmentSelection.scale, mPoseModel);
        if(i==0) augmentSelection.RotAndFinalSize = estimateRotation(
                    metaData,
                    cv::Size{(int)std::round(metaData.imageSize.width * augmentSelection.scale),
                             (int)std::round(metaData.imageSize.height * augmentSelection.scale)},
                    param_);
        applyRotation(metaData, augmentSelection.RotAndFinalSize.first, mPoseModel);
        if(i==0) augmentSelection.cropCenter = estimateCrop(metaData, param_);
        applyCrop(metaData, augmentSelection.cropCenter, finalCropSize, mPoseModel);
        if(i==0) augmentSelection.flip = estimateFlip(metaData, param_);
        applyFlip(metaData, augmentSelection.flip, finalImageHeight, param_, mPoseModel);
        applyAllAugmentation(imgAug, augmentSelection.RotAndFinalSize.first, augmentSelection.scale,
                             augmentSelection.flip, augmentSelection.cropCenter, finalCropSize, img, 0);
        applyAllAugmentation(maskAug, augmentSelection.RotAndFinalSize.first,
                             augmentSelection.scale, augmentSelection.flip, augmentSelection.cropCenter,
                             finalCropSize, mask, 255);
        applyAllAugmentation(maskBgAug, augmentSelection.RotAndFinalSize.first,
                             augmentSelection.scale, augmentSelection.flip, augmentSelection.cropCenter,
                             finalCropSize, maskBackgroundImage, 255);
        const cv::Point2i backgroundCropCenter{backgroundImage.cols/2, backgroundImage.rows/2};
        cv::Mat backgroundImageTemp;
        applyCrop(backgroundImageTemp, backgroundCropCenter, backgroundImage, 0, finalCropSize);
        applyFlip(bgImgAug, augmentSelection.flip, backgroundImageTemp);
        // Resize mask
        if (!maskAug.empty()){
            cv::Mat maskAugTemp;
            cv::resize(maskAug, maskAugTemp, cv::Size{gridX, gridY}, 0, 0, cv::INTER_AREA);
            maskAug = maskAugTemp;
        }
        // Final background image - elementwise multiplication
        if (!bgImgAug.empty() && !maskBgAug.empty())
        {
            // Apply mask to background image
            cv::Mat backgroundImageAugmentedTemp;
            bgImgAug.copyTo(backgroundImageAugmentedTemp, maskBgAug);
            // Add background image to image augmented
            cv::Mat imageAugmentedTemp;
            cv::addWeighted(imgAug, 1., backgroundImageAugmentedTemp, 1., 0., imageAugmentedTemp);
            imgAug = imageAugmentedTemp;
        }

        // Create Label for frame
        Dtype* labelmapTemp = new Dtype[getNumberChannels() * gridY * gridX];
        if(mStaf){
            if(mStaf == 1) generateLabelMapStaf(labelmapTemp, imgAug.size(), maskAug, metaData, imgAug, stride);
            else if(mStaf == 2) generateLabelMapStafWithPaf(labelmapTemp, imgAug.size(), maskAug, metaData, imgAug, stride);
        }else{
            generateLabelMap(labelmapTemp, imgAug.size(), maskAug, metaData, imgAug, stride);
        }
//        if(i == 3 && vid == 3){
//        vizDebug(imgAug, metaData, labelmapTemp, finalImageWidth, finalImageHeight, gridX, gridY, stride, mPoseModel, mModelString, getNumberChannels()/2);
//        exit(-1);
//        }
        //imgAugPrev = imgAug.clone();

        // Convert image to Caffe Format
        Dtype* imgaugTemp = new Dtype[imgAug.channels()*imgAug.size().width*imgAug.size().height];
        matToCaffe(imgaugTemp, imgAug);

        // Get pointers for all
        int dataOffset = imgAug.channels()*imgAug.size().width*imgAug.size().height;
        int labelOffset = getNumberChannels() * gridY * gridX;
        Dtype* transformedDataPtr = transformedData->mutable_cpu_data();
        Dtype* transformedLabelPtr = transformedLabel->mutable_cpu_data(); // Max 6,703,488

        // Copy label
        int totalVid = transformedLabel->shape()[0]/frames;
        std::copy(labelmapTemp, labelmapTemp + labelOffset, transformedLabelPtr + (i*totalVid*labelOffset + vid*labelOffset));
        delete labelmapTemp;

        // Copy data
        std::copy(imgaugTemp, imgaugTemp + dataOffset, transformedDataPtr + (i*totalVid*dataOffset + vid*dataOffset));
        delete imgaugTemp;
    }
}

template<typename Dtype>
void OPDataTransformer<Dtype>::Test(int frames, Blob<Dtype> *transformedData, Blob<Dtype> *transformedLabel)
{
    int totalVid = transformedLabel->shape()[0]/frames;
    int dataOffset = transformedData->shape()[1]*transformedData->shape()[2]*transformedData->shape()[3];
    int labelOffset = transformedLabel->shape()[1]*transformedLabel->shape()[2]*transformedLabel->shape()[3];
    Dtype* transformedDataPtr = transformedData->mutable_cpu_data();
    Dtype* transformedLabelPtr = transformedLabel->mutable_cpu_data(); // Max 6,703,488

    // Test Data
    for(int fid=0; fid<frames; fid++){
        for(int vid=0; vid<totalVid; vid++){
            Dtype* imgPtr = transformedDataPtr + totalVid*fid*dataOffset + vid*dataOffset;
            cv::Mat testImg;
            caffeToMat(testImg, imgPtr, cv::Size(transformedData->shape()[3], transformedData->shape()[2]));
            //int labelFrame = 2 * getNumberBodyBkgAndPAF(mPoseModel) - 1;

            int labelFrame = 174 + (2*5);
            int hmLabelFrame = 110 + 5;
            int maskLabelFrame = 132 + (2*5);

            // Need a way to visalize paf?
            cv::Mat hmLabel(cv::Size(transformedLabel->shape()[3], transformedLabel->shape()[2]), CV_32FC1);
            cv::Mat xLabel(cv::Size(transformedLabel->shape()[3], transformedLabel->shape()[2]), CV_32FC1);
            cv::Mat yLabel(cv::Size(transformedLabel->shape()[3], transformedLabel->shape()[2]), CV_32FC1);
            cv::Mat maskLabel(cv::Size(transformedLabel->shape()[3], transformedLabel->shape()[2]), CV_32FC1);
            Dtype* xLabelPtr = transformedLabelPtr + totalVid*fid*labelOffset + vid*labelOffset + (labelFrame)*transformedLabel->shape()[2]*transformedLabel->shape()[3];
            Dtype* yLabelPtr = transformedLabelPtr + totalVid*fid*labelOffset + vid*labelOffset + (labelFrame+1)*transformedLabel->shape()[2]*transformedLabel->shape()[3];
            Dtype* hmLabelPtr = transformedLabelPtr + totalVid*fid*labelOffset + vid*labelOffset + (hmLabelFrame)*transformedLabel->shape()[2]*transformedLabel->shape()[3];
            Dtype* maskLabelPtr = transformedLabelPtr + totalVid*fid*labelOffset + vid*labelOffset + (maskLabelFrame)*transformedLabel->shape()[2]*transformedLabel->shape()[3];
            std::copy(xLabelPtr, xLabelPtr + xLabel.size().width*xLabel.size().height, &xLabel.at<float>(0,0));
            std::copy(yLabelPtr, yLabelPtr + yLabel.size().width*yLabel.size().height, &yLabel.at<float>(0,0));
            std::copy(hmLabelPtr, hmLabelPtr + hmLabel.size().width*hmLabel.size().height, &hmLabel.at<float>(0,0));
            std::copy(maskLabelPtr, maskLabelPtr + maskLabel.size().width*maskLabel.size().height, &maskLabel.at<float>(0,0));
            cv::resize(xLabel, xLabel, cv::Size(xLabel.size().width*8,xLabel.size().height*8));
            cv::resize(yLabel, yLabel, cv::Size(yLabel.size().width*8,yLabel.size().height*8));
            cv::resize(hmLabel, hmLabel, cv::Size(hmLabel.size().width*8,hmLabel.size().height*8));
            cv::resize(maskLabel, maskLabel, cv::Size(maskLabel.size().width*8,maskLabel.size().height*8));
            for(int v=0; v<xLabel.size().height; v+=5){
                for(int u=0; u<xLabel.size().height; u+=5){
                    if(fabs(xLabel.at<float>(cv::Point(u,v))) > 0 || fabs(yLabel.at<float>(cv::Point(u,v))) > 0){
                        float scalar = 10;
                        cv::Point2f vector(xLabel.at<float>(cv::Point(u,v)), yLabel.at<float>(cv::Point(u,v)));
                        cv::Point2f p1(u,v);
                        cv::Point p2(u + scalar*vector.x, v + scalar*vector.y);
                        cv::line(testImg, p1, p2, cv::Scalar(255,0,0));
                    }
                }
            }
            hmLabel*=255;
            maskLabel*=255;
            hmLabel.convertTo(hmLabel, CV_8UC1);
            maskLabel.convertTo(maskLabel, CV_8UC1);
            cv::cvtColor(hmLabel, hmLabel, cv::COLOR_GRAY2BGR);
            cv::cvtColor(maskLabel, maskLabel, cv::COLOR_GRAY2BGR);
            testImg = testImg*0.4 + hmLabel*0.6;
            testImg = testImg*0.5 + maskLabel*0.5;
            //testImg = maskLabel;

            // 0,1,2,3,4
            //labelFrame = 131;

//            cv::Mat bgLabel(cv::Size(transformedLabel->shape()[3], transformedLabel->shape()[2]), CV_32FC1);
//            Dtype* labelPtr = transformedLabelPtr + totalVid*fid*labelOffset + vid*labelOffset + (labelFrame)*transformedLabel->shape()[2]*transformedLabel->shape()[3];
//            std::copy(labelPtr, labelPtr + bgLabel.size().width*bgLabel.size().height, &bgLabel.at<float>(0,0));
//            bgLabel*=255;
//            bgLabel.convertTo(bgLabel, CV_8UC1);
//            cv::bitwise_not(bgLabel, bgLabel);
//            cv::cvtColor(bgLabel, bgLabel, cv::COLOR_GRAY2BGR);
//            cv::resize(bgLabel, bgLabel, cv::Size(bgLabel.size().width*8,bgLabel.size().height*8));
//            testImg = testImg*0.2 + bgLabel*0.8;

            cv::imwrite("/home/raaj/visualize/"+std::to_string(vid)+"-"+std::to_string(fid)+".png",testImg);
        }
    }

    exit(-1);
}


// OpenPose: added
template<typename Dtype>
cv::Mat  OPDataTransformer<Dtype>::parseBackground(const Datum *datumNegative){
    const auto finalImageWidth = (int)param_.crop_size_x();
    const auto finalImageHeight = (int)param_.crop_size_y();
    const cv::Size finalCropSize{(int)param_.crop_size_x(), (int)param_.crop_size_y()};

    cv::Mat backgroundImage;
    if (datumNegative != nullptr)
    {
        const std::string& data = datumNegative->data();
        const int datumNegativeWidth = datumNegative->width();
        const int datumNegativeHeight = datumNegative->height();
        const auto datumNegativeArea = (int)(datumNegativeHeight * datumNegativeWidth);
        // OpenCV wrapping --> 1.7x speed up naive image.at<cv::Vec3b>, 1.25x speed up with smart speed up
        const cv::Mat b(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[0]);
        const cv::Mat g(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[datumNegativeArea]);
        const cv::Mat r(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[2*datumNegativeArea]);
        std::vector<cv::Mat> bgr{b,g,r};
        cv::merge(bgr, backgroundImage);
        // // Security checks
        // const auto datumNegativeArea2 = (int)(backgroundImage.rows * backgroundImage.cols);
        // CHECK_EQ(datumNegativeArea2, datumNegativeArea);
        // CHECK_EQ(cv::norm(backgroundImage-image2), 0);
        // Included data augmentation: cropping
        // Disable data augmentation --> minX = minY = 0
        // Data augmentation: cropping
        if (datumNegativeWidth > finalImageWidth && datumNegativeHeight > finalImageHeight)
        {
            const auto xDiff = datumNegativeWidth - finalImageWidth;
            const auto yDiff = datumNegativeHeight - finalImageHeight;
            const auto minX = (xDiff <= 0 ? 0 :
                                            (int)std::round(xDiff * float(std::rand()) / float(RAND_MAX)) // [0,1]
                                            );
            const auto minY = (xDiff <= 0 ? 0 :
                                            (int)std::round(yDiff * float(std::rand()) / float(RAND_MAX)) // [0,1]
                                            );
            cv::Mat backgroundImageTemp;
            std::swap(backgroundImage, backgroundImageTemp);
            const cv::Point2i backgroundCropCenter{minX + finalImageWidth/2, minY + finalImageHeight/2};
            applyCrop(backgroundImage, backgroundCropCenter, backgroundImageTemp, 0, finalCropSize);
        }
        // Resize (if smaller than final crop size)
        // if (datumNegativeWidth < finalImageWidth || datumNegativeHeight < finalImageHeight)
        else
        {
            cv::Mat backgroundImageTemp;
            std::swap(backgroundImage, backgroundImageTemp);
            cv::resize(backgroundImageTemp, backgroundImage, cv::Size{finalImageWidth, finalImageHeight}, 0, 0, CV_INTER_CUBIC);
        }
    }

    return backgroundImage;
}

// OpenPose: added
template<typename Dtype>
void OPDataTransformer<Dtype>::TransformVideoSF(int vid, int frames, VSeq& vs, Blob<Dtype>* transformedData, Blob<Dtype>* transformedLabel,
                                                  const Datum& datum, const Datum* datumNegative)
{
    // Parameters
    const std::string& data = datum.data();
    const int datumHeight = datum.height();
    const int datumWidth = datum.width();
    const auto datumArea = (int)(datumHeight * datumWidth);
    const cv::Size finalCropSize{(int)param_.crop_size_x(), (int)param_.crop_size_y()};
    const auto stride = (int)param_.stride();
    const auto finalImageWidth = (int)param_.crop_size_x();
    const auto finalImageHeight = (int)param_.crop_size_y();
    const auto gridX = finalImageWidth / stride;
    const auto gridY = finalImageHeight / stride;

    // Read meta data (LMDB channel 3)
    MetaData metaData;
    readMetaData<Dtype>(metaData, mCurrentEpoch, &data[3 * datumArea], datumWidth, mPoseCategory, mPoseModel);

    // Image
    cv::Mat image;
    const cv::Mat b(datumHeight, datumWidth, CV_8UC1, (unsigned char*)&data[0]);
    const cv::Mat g(datumHeight, datumWidth, CV_8UC1, (unsigned char*)&data[datumArea]);
    const cv::Mat r(datumHeight, datumWidth, CV_8UC1, (unsigned char*)&data[2*datumArea]);
    std::vector<cv::Mat> bgr = {b,g,r};
    cv::merge(bgr, image);
    const auto initImageWidth = (int)image.cols;
    const auto initImageHeight = (int)image.rows;

    // BG
    // Read background image
    cv::Mat backgroundImage;
    cv::Mat maskBackgroundImage = (datumNegative != nullptr
            ? cv::Mat(image.size().height, image.size().width, CV_8UC1, cv::Scalar{0}) : cv::Mat());
    if (datumNegative != nullptr)
    {
        const std::string& data = datumNegative->data();
        const int datumNegativeWidth = datumNegative->width();
        const int datumNegativeHeight = datumNegative->height();
        const auto datumNegativeArea = (int)(datumNegativeHeight * datumNegativeWidth);
        // OpenCV wrapping --> 1.7x speed up naive image.at<cv::Vec3b>, 1.25x speed up with smart speed up
        const cv::Mat b(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[0]);
        const cv::Mat g(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[datumNegativeArea]);
        const cv::Mat r(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[2*datumNegativeArea]);
        std::vector<cv::Mat> bgr = {b,g,r};
        cv::merge(bgr, backgroundImage);
        // // Security checks
        // const auto datumNegativeArea2 = (int)(backgroundImage.rows * backgroundImage.cols);
        // CHECK_EQ(datumNegativeArea2, datumNegativeArea);
        // CHECK_EQ(cv::norm(backgroundImage-image2), 0);
        // Included data augmentation: cropping
        // Disable data augmentation --> minX = minY = 0
        // Data augmentation: cropping
        if (datumNegativeWidth > finalImageWidth && datumNegativeHeight > finalImageHeight)
        {
            const auto xDiff = datumNegativeWidth - finalImageWidth;
            const auto yDiff = datumNegativeHeight - finalImageHeight;
            const auto minX = (xDiff <= 0 ? 0 :
                                            (int)std::round(xDiff * float(std::rand()) / float(RAND_MAX)) // [0,1]
                                            );
            const auto minY = (xDiff <= 0 ? 0 :
                                            (int)std::round(yDiff * float(std::rand()) / float(RAND_MAX)) // [0,1]
                                            );
            cv::Mat backgroundImageTemp;
            std::swap(backgroundImage, backgroundImageTemp);
            const cv::Point2i backgroundCropCenter{minX + finalImageWidth/2, minY + finalImageHeight/2};
            applyCrop(backgroundImage, backgroundCropCenter, backgroundImageTemp, 0, finalCropSize);
        }
        // Resize (if smaller than final crop size)
        // if (datumNegativeWidth < finalImageWidth || datumNegativeHeight < finalImageHeight)
        else
        {
            cv::Mat backgroundImageTemp;
            std::swap(backgroundImage, backgroundImageTemp);
            cv::resize(backgroundImageTemp, backgroundImage, cv::Size{finalImageWidth, finalImageHeight}, 0, 0, CV_INTER_CUBIC);
        }
    }

    // Mask
    cv::Mat maskMiss = ((mPoseCategory == PoseCategory::COCO || mPoseCategory == PoseCategory::MPII || mPoseCategory == PoseCategory::PT)
                              // COCO & MPII
                              ? cv::Mat(initImageHeight, initImageWidth, CV_8UC1, (unsigned char*)&data[4*datumArea])
            // DOME & MPII_hands
            : cv::Mat(initImageHeight, initImageWidth, CV_8UC1, cv::Scalar{255}));

    // Start Aug
    //metaData.objPos = cv::Point(image.size().width/2, image.size().height/2
    AugmentSelection startAug, endAug;
    startAug.scale = estimateScale(metaData, param_);
    startAug.rotation = getRotRand(param_);
    startAug.pointOffset = estimatePO(metaData, param_);

    endAug.scale = estimateScale(metaData, param_);
    endAug.rotation = getRotRand(param_);
    endAug.pointOffset = estimatePO(metaData, param_);

    std::vector<AugmentSelection> augVec(frames);
    MetaData metaDataPrev;
    for(int i=0; i<frames; i++){
        float scale = startAug.scale + (((endAug.scale - startAug.scale) / frames))*i;
        float rotation = startAug.rotation + (((endAug.rotation - startAug.rotation) / frames))*i;
        cv::Size ci = cv::Size(startAug.pointOffset.width + (((endAug.pointOffset.width - startAug.pointOffset.width) / frames))*i,
                                     startAug.pointOffset.height + (((endAug.pointOffset.height - startAug.pointOffset.height) / frames))*i);

        MetaData metaDataCopy = metaData;

        // Augment
        cv::Mat& img = image;
        const cv::Mat& mask = maskMiss;
        AugmentSelection augmentSelection;
        // Augment here
        cv::Mat imgAug, maskAug, maskBgAug, bgImgAug;
        augmentSelection.scale = scale;
        applyScale(metaDataCopy, augmentSelection.scale, mPoseModel);
        augmentSelection.RotAndFinalSize = estimateRotation(
                    metaDataCopy,
                    cv::Size{(int)std::round(metaDataCopy.imageSize.width * startAug.scale),
                             (int)std::round(metaDataCopy.imageSize.height * startAug.scale)},
                    rotation);
        applyRotation(metaDataCopy, augmentSelection.RotAndFinalSize.first, mPoseModel);

        augmentSelection.cropCenter = addPO(metaDataCopy, ci);
        //augmentSelection.cropCenter = ci;
        applyCrop(metaDataCopy, augmentSelection.cropCenter, finalCropSize, mPoseModel);
        //if(i==0) augmentSelection.flip = estimateFlip(metaData, param_);
        //applyFlip(metaData, augmentSelection.flip, finalImageHeight, param_, mPoseModel);
        applyAllAugmentation(imgAug, augmentSelection.RotAndFinalSize.first, augmentSelection.scale,
                             augmentSelection.flip, augmentSelection.cropCenter, finalCropSize, img, 0);
        applyAllAugmentation(maskAug, augmentSelection.RotAndFinalSize.first,
                             augmentSelection.scale, augmentSelection.flip, augmentSelection.cropCenter,
                             finalCropSize, mask, 255);
        applyAllAugmentation(maskBgAug, augmentSelection.RotAndFinalSize.first,
                             augmentSelection.scale, augmentSelection.flip, augmentSelection.cropCenter,
                             finalCropSize, maskBackgroundImage, 255);
        const cv::Point2i backgroundCropCenter{backgroundImage.cols/2, backgroundImage.rows/2};
        cv::Mat backgroundImageTemp;
        applyCrop(backgroundImageTemp, backgroundCropCenter, backgroundImage, 0, finalCropSize);
        applyFlip(bgImgAug, augmentSelection.flip, backgroundImageTemp);
        // Resize mask
        if (!maskAug.empty()){
            cv::Mat maskAugTemp;
            cv::resize(maskAug, maskAugTemp, cv::Size{gridX, gridY}, 0, 0, cv::INTER_AREA);
            maskAug = maskAugTemp;
        }
        // Final background image - elementwise multiplication
        if (!bgImgAug.empty() && !maskBgAug.empty())
        {
            // Apply mask to background image
            cv::Mat backgroundImageAugmentedTemp;
            bgImgAug.copyTo(backgroundImageAugmentedTemp, maskBgAug);
            // Add background image to image augmented
            cv::Mat imageAugmentedTemp;
            cv::addWeighted(imgAug, 1., backgroundImageAugmentedTemp, 1., 0., imageAugmentedTemp);
            imgAug = imageAugmentedTemp;
        }

        // Save Prev
        if(i != 0){
            metaDataCopy.jointsOthersPrev = metaDataPrev.jointsOthers;
            metaDataCopy.jointsSelfPrev = metaDataPrev.jointsSelf;
        }
        metaDataPrev = metaDataCopy;

        // Create Label for frame
        Dtype* labelmapTemp = new Dtype[getNumberChannels() * gridY * gridX];
        if(mStaf){
            if(mStaf == 1) generateLabelMapStaf(labelmapTemp, imgAug.size(), maskAug, metaData, imgAug, stride);
            else if(mStaf == 2) generateLabelMapStafWithPaf(labelmapTemp, imgAug.size(), maskAug, metaData, imgAug, stride);
        }else{
            generateLabelMap(labelmapTemp, imgAug.size(), maskAug, metaData, imgAug, stride);
        }
//        if(i == 3 &&  metaData.writeNumber == 1){
//        vizDebug(imgAug, metaDataCopy, labelmapTemp, finalImageWidth, finalImageHeight, gridX, gridY, stride, mPoseModel, mModelString, getNumberChannels()/2);
//        exit(-1);
//        }

        // Convert image to Caffe Format
        Dtype* imgaugTemp = new Dtype[imgAug.channels()*imgAug.size().width*imgAug.size().height];
        matToCaffe(imgaugTemp, imgAug);

        // Get pointers for all
        int dataOffset = imgAug.channels()*imgAug.size().width*imgAug.size().height;
        int labelOffset = getNumberChannels() * gridY * gridX;
        Dtype* transformedDataPtr = transformedData->mutable_cpu_data();
        Dtype* transformedLabelPtr = transformedLabel->mutable_cpu_data(); // Max 6,703,488

        // Copy label
        int totalVid = transformedLabel->shape()[0]/frames;
        std::copy(labelmapTemp, labelmapTemp + labelOffset, transformedLabelPtr + (i*totalVid*labelOffset + vid*labelOffset));
        delete labelmapTemp;

        // Copy data
        std::copy(imgaugTemp, imgaugTemp + dataOffset, transformedDataPtr + (i*totalVid*dataOffset + vid*dataOffset));
        delete imgaugTemp;
    }
}

// OpenPose: added
template<typename Dtype>
void OPDataTransformer<Dtype>::Transform(Blob<Dtype>* transformedData, Blob<Dtype>* transformedLabel,
                                         const Datum& datum, const Datum* datumNegative,
                                         Blob<Dtype> extra_transformed_labels[],
                                         std::vector<int> extra_strides,
                                         int extra_labels_count)
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
    generateDataAndLabel(transformedDataPtr, transformedLabelPtr, datum, datumNegative, extra_transformed_labels, extra_strides, extra_labels_count);
    VLOG(2) << "Transform: " << timer.MicroSeconds() / 1000.0  << " ms";
}

template <typename Dtype>
int OPDataTransformer<Dtype>::getNumberChannels() const
{
    int totalChannels = 0;
    if(mTpaf)
        totalChannels = 2 * getNumberBodyBkgAndPAF(mPoseModel) + getNumberBodyParts(mPoseModel) * 2 + getNumberBodyParts(mPoseModel) * 2;
    else if(mStaf){
        if(mStaf == 1){
            int totalStaf = getNumberBodyParts(mPoseModel) * 2 * mStafIDS.size();
            totalChannels = (totalStaf + (getNumberBodyParts(mPoseModel) + 1)) * 2;
        }else if(mStaf == 2){
            int totalStaf = getNumberBodyParts(mPoseModel) * 2 * mStafIDS.size();
            int totalPaf = (getNumberBodyParts(mPoseModel) + 1) * 2;
            totalChannels = (totalPaf + totalStaf + (getNumberBodyParts(mPoseModel) + 1)) * 2;
        }
    }else
        totalChannels = 2 * getNumberBodyBkgAndPAF(mPoseModel);

    return totalChannels;
    // // For Distance
    // return 2 * (getNumberBodyBkgAndPAF(mPoseModel) + getNumberPafChannels(mPoseModel)/2);
}
// OpenPose: end


// OpenPose: added
template<typename Dtype>
void OPDataTransformer<Dtype>::generateDataAndLabel(Dtype* transformedData, Dtype* transformedLabel,
                                                    const Datum& datum, const Datum* datumNegative,
                                                    Blob<Dtype> extra_transformed_labels[],
                                                    std::vector<int> extra_strides,
                                                    int extra_labels_count)
{
    // Parameters
    const std::string& data = datum.data();
    const int datumHeight = datum.height();
    const int datumWidth = datum.width();
    const auto datumArea = (int)(datumHeight * datumWidth);
    const cv::Size finalCropSize{(int)param_.crop_size_x(), (int)param_.crop_size_y()};
    const auto stride = (int)param_.stride();
    const auto finalImageWidth = (int)param_.crop_size_x();
    const auto finalImageHeight = (int)param_.crop_size_y();
    const auto gridX = finalImageWidth / stride;
    const auto gridY = finalImageHeight / stride;

    // Time measurement
    CPUTimer timer1;
    timer1.Start();

    // const bool hasUInt8 = data.size() > 0;
    CHECK(data.size() > 0);

    // Read meta data (LMDB channel 3)
    MetaData metaData;
    // DOME
    if (mPoseCategory == PoseCategory::DOME)
        readMetaData<Dtype>(metaData, mCurrentEpoch, data.c_str(), datumWidth, mPoseCategory, mPoseModel);
    // COCO & MPII_hands
    else
        readMetaData<Dtype>(metaData, mCurrentEpoch, &data[3 * datumArea], datumWidth, mPoseCategory, mPoseModel);
    const auto depthEnabled = metaData.depthEnabled;

    // Read image (LMDB channel 1)
    cv::Mat image;
    // DOME
    if (mPoseCategory == PoseCategory::DOME)
    {
        const auto imageFullPath = param_.media_directory() + metaData.imageSource;
        image = cv::imread(imageFullPath, CV_LOAD_IMAGE_COLOR);
        if (image.empty())
            throw std::runtime_error{"Empty image at " + imageFullPath + getLine(__LINE__, __FUNCTION__, __FILE__)};
    }
    // COCO & MPII_hands
    else
    {
        // // Naive copy
        // image = cv::Mat(datumHeight, datumWidth, CV_8UC3);
        // const auto initImageArea = (int)(image.rows * image.cols);
        // CHECK_EQ(initImageArea, datumArea);
        // for (auto y = 0; y < image.rows; y++)
        // {
        //     const auto yOffset = (int)(y*image.cols);
        //     for (auto x = 0; x < image.cols; x++)
        //     {
        //         const auto xyOffset = yOffset + x;
        //         cv::Vec3b& bgr = image.at<cv::Vec3b>(y, x);
        //         for (auto c = 0; c < 3; c++)
        //         {
        //             const auto dIndex = (int)(c*initImageArea + xyOffset);
        //             // if (hasUInt8)
        //                 bgr[c] = static_cast<Dtype>(static_cast<uint8_t>(data[dIndex]));
        //             // else
        //                 // bgr[c] = datum.float_data(dIndex);
        //         }
        //     }
        // }
        // // Naive copy (slightly optimized)
        // image = cv::Mat(datumHeight, datumWidth, CV_8UC3);
        // auto* uCharPtrCvMat = (unsigned char*)(image.data);
        // for (auto y = 0; y < image.rows; y++)
        // {
        //     const auto yOffset = (int)(y*image.cols);
        //     for (auto x = 0; x < image.cols; x++)
        //     {
        //         const auto xyOffset = yOffset + x;
        //         const auto baseIndex = 3*xyOffset;
        //         uCharPtrCvMat[baseIndex] = static_cast<Dtype>(static_cast<uint8_t>(data[xyOffset]));
        //         uCharPtrCvMat[baseIndex + 1] = static_cast<Dtype>(static_cast<uint8_t>(data[xyOffset + initImageArea]));
        //         uCharPtrCvMat[baseIndex + 2] = static_cast<Dtype>(static_cast<uint8_t>(data[xyOffset + 2*initImageArea]));
        //     }
        // }
        // // Security check - Assert
        // cv::Mat image2;
        // std::swap(image, image2);
        // OpenCV wrapping --> 1.7x speed up naive image.at<cv::Vec3b>, 1.25x speed up with smart speed up
        const cv::Mat b(datumHeight, datumWidth, CV_8UC1, (unsigned char*)&data[0]);
        const cv::Mat g(datumHeight, datumWidth, CV_8UC1, (unsigned char*)&data[datumArea]);
        const cv::Mat r(datumHeight, datumWidth, CV_8UC1, (unsigned char*)&data[2*datumArea]);
        std::vector<cv::Mat> bgr = {b,g,r};
        cv::merge(bgr, image);
        // // Security checks
        // const auto initImageArea = (int)(image.rows * image.cols);
        // CHECK_EQ(initImageArea, datumArea);
        // CHECK_EQ(cv::norm(image-image2), 0);
    }
    const auto initImageWidth = (int)image.cols;
    const auto initImageHeight = (int)image.rows;

    // Read background image
    cv::Mat backgroundImage;
    if (datumNegative != nullptr)
    {
        const std::string& data = datumNegative->data();
        const int datumNegativeWidth = datumNegative->width();
        const int datumNegativeHeight = datumNegative->height();
        const auto datumNegativeArea = (int)(datumNegativeHeight * datumNegativeWidth);
        // Background image
        // // Naive copy
        // backgroundImage = cv::Mat(datumNegativeHeight, datumNegativeWidth, CV_8UC3);
        // for (auto y = 0; y < datumNegativeHeight; y++)
        // {
        //     const auto yOffset = (int)(y*datumNegativeWidth);
        //     for (auto x = 0; x < datumNegativeWidth; x++)
        //     {
        //         const auto xyOffset = yOffset + x;
        //         cv::Vec3b& bgr = backgroundImage.at<cv::Vec3b>(y, x);
        //         for (auto c = 0; c < 3; c++)
        //         {
        //             const auto dIndex = (int)(c*datumNegativeArea + xyOffset);
        //             bgr[c] = static_cast<Dtype>(static_cast<uint8_t>(data[dIndex]));
        //         }
        //     }
        // }
        // // Naive copy (slightly optimized)
        // backgroundImage = cv::Mat(datumNegativeHeight, datumNegativeWidth, CV_8UC3);
        // auto* uCharPtrCvMat = (unsigned char*)(backgroundImage.data);
        // for (auto y = 0; y < datumNegativeHeight; y++)
        // {
        //     const auto yOffset = (int)(y*datumNegativeWidth);
        //     for (auto x = 0; x < datumNegativeWidth; x++)
        //     {
        //         const auto xyOffset = yOffset + x;
        //         const auto baseIndex = 3*xyOffset;
        //         uCharPtrCvMat[baseIndex] = static_cast<Dtype>(static_cast<uint8_t>(data[xyOffset]));
        //         uCharPtrCvMat[baseIndex + 1] = static_cast<Dtype>(static_cast<uint8_t>(data[xyOffset + datumNegativeArea]));
        //         uCharPtrCvMat[baseIndex + 2] = static_cast<Dtype>(static_cast<uint8_t>(data[xyOffset + 2*datumNegativeArea]));
        //     }
        // }
        // // Security check - Assert
        // cv::Mat image2;
        // std::swap(backgroundImage, image2);
        // OpenCV wrapping --> 1.7x speed up naive image.at<cv::Vec3b>, 1.25x speed up with smart speed up
        const cv::Mat b(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[0]);
        const cv::Mat g(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[datumNegativeArea]);
        const cv::Mat r(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[2*datumNegativeArea]);
        std::vector<cv::Mat> bgr = {b,g,r};
        cv::merge(bgr, backgroundImage);
        // // Security checks
        // const auto datumNegativeArea2 = (int)(backgroundImage.rows * backgroundImage.cols);
        // CHECK_EQ(datumNegativeArea2, datumNegativeArea);
        // CHECK_EQ(cv::norm(backgroundImage-image2), 0);
        // Included data augmentation: cropping
        // Disable data augmentation --> minX = minY = 0
        // Data augmentation: cropping
        if (datumNegativeWidth > finalImageWidth && datumNegativeHeight > finalImageHeight)
        {
            const auto xDiff = datumNegativeWidth - finalImageWidth;
            const auto yDiff = datumNegativeHeight - finalImageHeight;
            const auto minX = (xDiff <= 0 ? 0 :
                                            (int)std::round(xDiff * float(std::rand()) / float(RAND_MAX)) // [0,1]
                                            );
            const auto minY = (xDiff <= 0 ? 0 :
                                            (int)std::round(yDiff * float(std::rand()) / float(RAND_MAX)) // [0,1]
                                            );
            cv::Mat backgroundImageTemp;
            std::swap(backgroundImage, backgroundImageTemp);
            const cv::Point2i backgroundCropCenter{minX + finalImageWidth/2, minY + finalImageHeight/2};
            applyCrop(backgroundImage, backgroundCropCenter, backgroundImageTemp, 0, finalCropSize);
        }
        // Resize (if smaller than final crop size)
        // if (datumNegativeWidth < finalImageWidth || datumNegativeHeight < finalImageHeight)
        else
        {
            cv::Mat backgroundImageTemp;
            std::swap(backgroundImage, backgroundImageTemp);
            cv::resize(backgroundImageTemp, backgroundImage, cv::Size{finalImageWidth, finalImageHeight}, 0, 0, CV_INTER_CUBIC);
        }
    }

    // Read mask miss (LMDB channel 2)
    const cv::Mat maskMiss = ((mPoseCategory == PoseCategory::COCO || mPoseCategory == PoseCategory::MPII || mPoseCategory == PoseCategory::PT)
                              // COCO & MPII
                              ? cv::Mat(initImageHeight, initImageWidth, CV_8UC1, (unsigned char*)&data[4*datumArea])
            // DOME & MPII_hands
            : cv::Mat(initImageHeight, initImageWidth, CV_8UC1, cv::Scalar{255}));
    // // Naive copy
    // cv::Mat maskMiss2;
    // // COCO
    // if (mPoseCategory == PoseCategory::COCO)
    // {
    //     maskMiss2 = cv::Mat(initImageHeight, initImageWidth, CV_8UC1, cv::Scalar{0});
    //     for (auto y = 0; y < maskMiss2.rows; y++)
    //     {
    //         const auto yOffset = (int)(y*initImageWidth);
    //         for (auto x = 0; x < initImageWidth; x++)
    //         {
    //             const auto xyOffset = yOffset + x;
    //             const auto dIndex = (int)(4*datumArea + xyOffset);
    //             Dtype dElement;
    //             // if (hasUInt8)
    //                 dElement = static_cast<Dtype>(static_cast<uint8_t>(data[dIndex]));
    //             // else
    //                 // dElement = datum.float_data(dIndex);
    //             if (std::round(dElement/255)!=1 && std::round(dElement/255)!=0)
    //                 throw std::runtime_error{"Value out of {0,1}" + getLine(__LINE__, __FUNCTION__, __FILE__)};
    //             maskMiss2.at<uchar>(y, x) = dElement; //round(dElement/255);
    //         }
    //     }
    // }
    // // DOME & MPII_hands
    // else
    //     maskMiss2 = cv::Mat(initImageHeight, initImageWidth, CV_8UC1, cv::Scalar{255});
    // // Security checks
    // CHECK_EQ(cv::norm(maskMiss-maskMiss2), 0);

    // Time measurement
    VLOG(2) << "  bgr[:] = datum: " << timer1.MicroSeconds()*1e-3 << " ms";

    // Depth image
    cv::Mat depth;
    if (depthEnabled)
    {
        const auto depthFullPath = param_.media_directory() + metaData.depthSource;
        depth = cv::imread(depthFullPath, CV_LOAD_IMAGE_ANYDEPTH);
        if (image.empty())
            throw std::runtime_error{"Empty depth at " + depthFullPath + getLine(__LINE__, __FUNCTION__, __FILE__)};
    }

    // timer1.Start();
    // // Clahe
    // if (param_.do_clahe())
    //     clahe(image, param_.clahe_tile_size(), param_.clahe_clip_limit());
    // BGR --> Gray --> BGR
    // if image is grey
    // cv::cvtColor(image, image, CV_GRAY2BGR);
    // VLOG(2) << "  cvtColor and CLAHE: " << timer1.MicroSeconds()*1e-3 << " ms";
    timer1.Start();

    VLOG(2) << "  ReadMeta+MetaJoints: " << timer1.MicroSeconds()*1e-3 << " ms";

    // Data augmentation
    timer1.Start();
    AugmentSelection augmentSelection;
    // // Debug - Visualize original
    // debugVisualize(image, metaData, augmentSelection, mPoseModel, phase_, param_);
    // Augmentation
    cv::Mat imageAugmented;
    cv::Mat backgroundImageAugmented;
    cv::Mat maskMissAugmented;
    cv::Mat maskMissAugmentedOrig;
    cv::Mat depthAugmented;
    VLOG(2) << "   input size (" << initImageWidth << ", " << initImageHeight << ")";
    // We only do random transform augmentSelection augmentation when training.
    if (phase_ == TRAIN) // 80% time is spent here
    {
        // Mask for background image
        // Image size, not backgroundImage
        cv::Mat maskBackgroundImage = (datumNegative != nullptr
                ? cv::Mat(initImageHeight, initImageWidth, CV_8UC1, cv::Scalar{0}) : cv::Mat());
        cv::Mat maskBackgroundImageAugmented;
        // Swap center?
        swapCenterPoint(metaData, param_, mPoseModel);
        // Augmentation (scale, rotation, cropping, and flipping)
        // Order does matter, otherwise code will fail doing augmentation
        augmentSelection.scale = estimateScale(metaData, param_);
        applyScale(metaData, augmentSelection.scale, mPoseModel);
        augmentSelection.RotAndFinalSize = estimateRotation(
                    metaData,
                    cv::Size{(int)std::round(image.cols * augmentSelection.scale),
                             (int)std::round(image.rows * augmentSelection.scale)},
                    param_);
        applyRotation(metaData, augmentSelection.RotAndFinalSize.first, mPoseModel);
        augmentSelection.cropCenter = estimateCrop(metaData, param_);
        applyCrop(metaData, augmentSelection.cropCenter, finalCropSize, mPoseModel);
        augmentSelection.flip = estimateFlip(metaData, param_);
        applyFlip(metaData, augmentSelection.flip, finalImageHeight, param_, mPoseModel);
        // Aug on images - ~80% code time spent in the following `applyAllAugmentation` lines
        applyAllAugmentation(imageAugmented, augmentSelection.RotAndFinalSize.first, augmentSelection.scale,
                             augmentSelection.flip, augmentSelection.cropCenter, finalCropSize, image,
                             0);
        applyAllAugmentation(maskBackgroundImageAugmented, augmentSelection.RotAndFinalSize.first,
                             augmentSelection.scale, augmentSelection.flip, augmentSelection.cropCenter,
                             finalCropSize, maskBackgroundImage, 255);
        applyAllAugmentation(maskMissAugmented, augmentSelection.RotAndFinalSize.first,
                             augmentSelection.scale, augmentSelection.flip, augmentSelection.cropCenter,
                             finalCropSize, maskMiss, 255);
        applyAllAugmentation(depthAugmented, augmentSelection.RotAndFinalSize.first,
                             augmentSelection.scale, augmentSelection.flip, augmentSelection.cropCenter,
                             finalCropSize, depth, 0);
        // backgroundImage augmentation (no scale/rotation)
        const cv::Point2i backgroundCropCenter{backgroundImage.cols/2, backgroundImage.rows/2};
        cv::Mat backgroundImageTemp;
        applyCrop(backgroundImageTemp, backgroundCropCenter, backgroundImage, 0, finalCropSize);
        applyFlip(backgroundImageAugmented, augmentSelection.flip, backgroundImageTemp);
        // Introduce occlusions
        doOcclusions(imageAugmented, backgroundImageAugmented, metaData, param_.number_max_occlusions(),
                     mPoseModel);
        // Resize mask
        if (!maskMissAugmented.empty()){
            maskMissAugmentedOrig = maskMissAugmented.clone();
            cv::resize(maskMissAugmented, maskMissAugmented, cv::Size{gridX, gridY}, 0, 0, cv::INTER_AREA);
        }
        // Final background image - elementwise multiplication
        if (!backgroundImageAugmented.empty() && !maskBackgroundImageAugmented.empty())
        {
            // Apply mask to background image
            cv::Mat backgroundImageAugmentedTemp;
            backgroundImageAugmented.copyTo(backgroundImageAugmentedTemp, maskBackgroundImageAugmented);
            // Add background image to image augmented
            cv::Mat imageAugmentedTemp;
            addWeighted(imageAugmented, 1., backgroundImageAugmentedTemp, 1., 0., imageAugmentedTemp);
            imageAugmented = imageAugmentedTemp;
        }
        if (depthEnabled && !depthAugmented.empty())
            cv::resize(depthAugmented, depthAugmented, cv::Size{gridX, gridY}, 0, 0, cv::INTER_AREA);
    }
    // Test
    else
    {
        exit(-1);
        imageAugmented = image;
        maskMissAugmented = maskMiss;
        depthAugmented = depth;
        // Resize mask
        if (!maskMissAugmented.empty())
            cv::resize(maskMissAugmented, maskMissAugmented, cv::Size{gridX, gridY}, 0, 0, cv::INTER_AREA);
        if (depthEnabled)
            cv::resize(depthAugmented, depthAugmented, cv::Size{gridX, gridY}, 0, 0, cv::INTER_AREA);
    }
    // // Debug - Visualize final (augmented) image
    // debugVisualize(imageAugmented, metaData, augmentSelection, mPoseModel, phase_, param_);
    // Augmentation time
    VLOG(2) << "  Aug: " << timer1.MicroSeconds()*1e-3 << " ms";
    // Data copy
    timer1.Start();
    // Copy imageAugmented into transformedData + mean-subtraction
    const int imageAugmentedArea = imageAugmented.rows * imageAugmented.cols;
    auto* uCharPtrCvMat = (unsigned char*)(imageAugmented.data);
    // x/256 - 0.5
    if (param_.normalization() == 0)
    {
        for (auto y = 0; y < imageAugmented.rows; y++)
        {
            const auto yOffset = y*imageAugmented.cols;
            for (auto x = 0; x < imageAugmented.cols; x++)
            {
                const auto xyOffset = yOffset + x;
                // const cv::Vec3b& bgr = imageAugmented.at<cv::Vec3b>(y, x);
                auto* bgr = &uCharPtrCvMat[3*xyOffset];
                transformedData[xyOffset] = (bgr[0] - 128) / 256.0;
                transformedData[xyOffset + imageAugmentedArea] = (bgr[1] - 128) / 256.0;
                transformedData[xyOffset + 2*imageAugmentedArea] = (bgr[2] - 128) / 256.0;
            }
        }
    }
    // x - channel average
    else if (param_.normalization() == 1)
    {
        for (auto y = 0; y < imageAugmented.rows ; y++)
        {
            const auto yOffset = y*imageAugmented.cols;
            for (auto x = 0; x < imageAugmented.cols ; x++)
            {
                const auto xyOffset = yOffset + x;
                // const cv::Vec3b& bgr = imageAugmented.at<cv::Vec3b>(y, x);
                auto* bgr = &uCharPtrCvMat[3*xyOffset];
                transformedData[xyOffset] = bgr[0] - 102.9801;
                transformedData[xyOffset + imageAugmentedArea] = bgr[1] - 115.9465;
                transformedData[xyOffset + 2*imageAugmentedArea] = bgr[2] - 122.7717;
            }
        }
    }
    // Unknown
    else
        throw std::runtime_error{"Unknown normalization at " + getLine(__LINE__, __FUNCTION__, __FILE__)};

    if(!mStaf)
    {
        // Generate and copy label
        generateLabelMap(transformedLabel, imageAugmented.size(), maskMissAugmented, metaData, imageAugmented, stride);
        if (depthEnabled)
            generateDepthLabelMap(transformedLabel, depthAugmented);
        VLOG(2) << "  AddGaussian+CreateLabel: " << timer1.MicroSeconds()*1e-3 << " ms";

        // Multilabel case
        if(extra_labels_count){
            for(int i=0; i<extra_labels_count; i++){
                auto gridX_extra = finalImageWidth / extra_strides[i];
                auto gridY_extra = finalImageHeight / extra_strides[i];
                cv::Mat maskMissAugmented_extra;
                cv::resize(maskMissAugmentedOrig, maskMissAugmented_extra, cv::Size{gridX_extra, gridY_extra}, 0, 0, cv::INTER_AREA);
                generateLabelMap(extra_transformed_labels[i].mutable_cpu_data(), imageAugmented.size(), maskMissAugmented_extra, metaData, imageAugmented, extra_strides[i]);
            }
        }

//        if (metaData.writeNumber == 0 && mPoseModel == PoseModel::COCO_21){
//            vizDebug(imageAugmented, metaData, transformedLabel, finalImageWidth, finalImageHeight, gridX, gridY, stride, mPoseModel, mModelString, getNumberBodyBkgAndPAF(mPoseModel));
//            exit(-1);
//        }
    }
    else if (mStaf)
    {
        // Generate and copy label
        if(mStaf == 1)
            generateLabelMapStaf(transformedLabel, imageAugmented.size(), maskMissAugmented, metaData, imageAugmented, stride);
        else if(mStaf == 2){
            generateLabelMapStafWithPaf(transformedLabel, imageAugmented.size(), maskMissAugmented, metaData, imageAugmented, stride);
        }

//        if (metaData.writeNumber == 3 && mPoseModel == PoseModel::COCO_21){
//            vizDebug(imageAugmented, metaData, transformedLabel, finalImageWidth, finalImageHeight, gridX, gridY, stride, mPoseModel, mModelString, getNumberChannels()/2);
//            exit(-1);
//        }
    }
}

template<typename Dtype>
void OPDataTransformer<Dtype>::generateDepthLabelMap(Dtype* transformedLabel, const cv::Mat& depth) const
{
    const auto gridX = (int)depth.cols;
    const auto gridY = (int)depth.rows;
    const auto channelOffset = gridY * gridX;
    const auto numberBpPafChannels = getNumberBodyAndPafChannels(mPoseModel);
    // generate depth
    for (auto gY = 0; gY < gridY; gY++)
    {
        const auto yOffset = gY*gridX;
        for (auto gX = 0; gX < gridX; gX++)
        {
            const auto xyOffset = yOffset + gX;
            auto depth_val = depth.at<uint16_t>(gY, gX);
            transformedLabel[(2*numberBpPafChannels+2)*channelOffset + xyOffset] = (depth_val>0)?1.0:0.0;
            transformedLabel[(2*numberBpPafChannels+3)*channelOffset + xyOffset] = float(depth_val)/1000.0;
        }
    }
}

float getNorm(const cv::Point2f& pointA, const cv::Point2f& pointB)
{
    const auto difference = pointA - pointB;
    return std::sqrt(difference.x*difference.x + difference.y*difference.y);
}

void maskHands(cv::Mat& maskMiss, const std::vector<float>& isVisible, const std::vector<cv::Point2f>& points,
               const float stride, const float ratio)
{
    for (auto part = 0 ; part < 2 ; part++)
    {
        const auto shoulderIndex = (part == 0 ? 5:2);
        const auto elbowIndex = shoulderIndex+1;
        const auto wristIndex = elbowIndex+1;
        if (isVisible.at(shoulderIndex) != 2 && isVisible.at(elbowIndex) != 2 && isVisible.at(wristIndex) != 2)
        {
            const auto ratioStride = 1.f / stride;
            const auto wrist = ratioStride * points.at(wristIndex);
            const auto elbow = ratioStride * points.at(elbowIndex);
            const auto shoulder = ratioStride * points.at(shoulderIndex);

            const auto distance = (int)std::round(ratio*std::max(getNorm(wrist, elbow), getNorm(elbow, shoulder)));
            const cv::Point momentum = (wrist-elbow)*0.25f;
            cv::Rect roi{(int)std::round(wrist.x + momentum.x - distance /*- wrist.x/2.f*/),
                        (int)std::round(wrist.y + momentum.y - distance /*- wrist.y/2.f*/),
                        2*distance, 2*distance};
            // Apply ROI
            keepRoiInside(roi, maskMiss.size());
            if (roi.area() > 0)
                maskMiss(roi).setTo(0.f); // For debugging use 0.5f
        }
        // // If there is no visible desired keypoints, mask out the whole background
        // else
        //     maskMiss.setTo(0.f); // For debugging use 0.5f
    }
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
            const auto ratioStride = 1.f / stride;
            const auto knee = ratioStride * points.at(kneeIndex);
            const auto ankle = ratioStride * points.at(ankleIndex);
            const auto distance = (int)std::round(ratio*getNorm(knee, ankle));
            const cv::Point momentum = (ankle-knee)*0.15f;
            cv::Rect roi{(int)std::round(ankle.x + momentum.x)-distance,
                        (int)std::round(ankle.y + momentum.y)-distance,
                        2*distance, 2*distance};
            // Apply ROI
            keepRoiInside(roi, maskMiss.size());
            if (roi.area() > 0)
                maskMiss(roi).setTo(0.f); // For debugging use 0.5f
        }
        // // If there is no visible desired keypoints, mask out the whole background
        // else
        //     maskMiss.setTo(0.f); // For debugging use 0.5f
    }
}

template<typename Dtype>
void fillMaskChannels(Dtype* transformedLabel, const int gridX, const int gridY, const int numberTotalChannels,
                      const int channelOffset, const cv::Mat& maskMiss)
{
    // Initialize labels to [0, 1] (depending on maskMiss)
    // // Naive version (very slow)
    // for (auto gY = 0; gY < gridY; gY++)
    // {
    //     const auto yOffset = gY*gridX;
    //     for (auto gX = 0; gX < gridX; gX++)
    //     {
    //         const auto xyOffset = yOffset + gX;
    //         const float weight = float(maskMiss.at<uchar>(gY, gX)) / 255.f;
    //         // Body part & PAFs & background channel & distance
    //         for (auto part = 0; part < numberTotalChannels; part++)
    //         // // For Distance
    //         // for (auto part = 0; part < numberTotalChannels - numberPafChannels/2; part++)
    //             transformedLabel[part*channelOffset + xyOffset] = weight;
    //     }
    // }
    // OpenCV wrapper: ~10x speed up with baseline
    cv::Mat maskMissFloat;
    const auto type = getType(Dtype(0));
    maskMiss.convertTo(maskMissFloat, type);
    maskMissFloat /= Dtype(255.f);
    // // For Distance
    // for (auto part = 0; part < numberTotalChannels - numberPafChannels/2; part++)
    for (auto part = 0; part < numberTotalChannels; part++)
    {
        auto* pointer = &transformedLabel[part*channelOffset];
        cv::Mat transformedLabel(gridY, gridX, type, (unsigned char*)(pointer));
        // // Not exactly 0 for limited floating precission
        // CHECK_LT(std::abs(cv::norm(transformedLabel-maskMissFloat)), 1e-6);
        maskMissFloat.copyTo(transformedLabel);
    }
}

void drawPoints(cv::Mat& clone, const MetaData& metaData){
    int i=0;
    for(auto p : metaData.jointsSelf.points){
        int viz = metaData.jointsSelf.isVisible[i];
        if(viz == 2) continue;
        if(p.x >= 0 || p.x < clone.size().width || p.y >= 0 || p.y < clone.size().height){
            cv::circle(clone, p, 2, cv::Scalar(128*viz,128*viz,128*viz), CV_FILLED);
            cv::putText(clone, std::to_string(i),  p, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 200, 200), 1);
        }
        i++;
    }
    for(auto joint : metaData.jointsOthers){
        i=0;
        for(auto p : joint.points){
            int viz = joint.isVisible[i];
            if(viz == 2) continue;
            if(p.x >= 0 || p.x < clone.size().width || p.y >= 0 || p.y < clone.size().height){
                cv::circle(clone, p, 2, cv::Scalar(128*viz,128*viz,128*viz), CV_FILLED);
                cv::putText(clone, std::to_string(i),  p, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 200, 200), 1);
            }
            i++;
        }
    }
}

template<typename Dtype>
void OPDataTransformer<Dtype>::generateLabelMapStafWithPaf(Dtype* transformedLabel, const cv::Size& imageSize, const cv::Mat& maskMiss,
                                                const MetaData& metaData, const cv::Mat& img, const int stride) const
{
    // Label size = image size / stride
    const auto rezX = (int)imageSize.width;
    const auto rezY = (int)imageSize.height;
    const auto gridX = rezX / stride;
    const auto gridY = rezY / stride;
    const auto channelOffset = gridY * gridX;
    const auto numberBodyParts = getNumberBodyParts(mPoseModel); // #BP
    const int totalStaf = getNumberBodyParts(mPoseModel) * 2 * mStafIDS.size();
    const int totalPaf = (getNumberBodyParts(mPoseModel) + 1) * 2;
    const int totalPafStaf = totalPaf + totalStaf;
    const int numberTotalChannels = getNumberChannels() / 2;
    const int backgroundMaskIndex = numberTotalChannels-1;
    const int backgroundIndex = getNumberChannels() - 1;

//    std::cout << numberTotalChannels << std::endl;
//    exit(-1);

    // Labels to 0
    std::fill(transformedLabel, transformedLabel + getNumberChannels() * gridY * gridX, 0.f);

    // Initialize labels to [0, 1] (depending on maskMiss)
    fillMaskChannels(transformedLabel, gridX, gridY, numberTotalChannels, channelOffset, maskMiss);

    // Masking out channels - For COCO_YY_ZZ models (ZZ < YY)
    std::vector<int> missingChannels;
    std::vector<int> missingSID;
    if (numberBodyParts > getNumberBodyPartsLmdb(mPoseModel))
    {
        // Remove BP/PAF non-labeled channels
        const auto& lmdbToOpenPoseKeypoints = getLmdbToOpenPoseKeypoints(mPoseModel);
        for (auto i = 0u ; i < lmdbToOpenPoseKeypoints.size() ; i++){
            if (lmdbToOpenPoseKeypoints[i].empty()){
                missingChannels.emplace_back(totalPafStaf + i);
                for(int j=0; j<mStafIDS.size(); j++) if(i == mStafIDS[j]) missingSID.emplace_back(j);
            }
        }
        for(auto msid : missingSID){
            for(int i=0; i<numberBodyParts*2; i++)
                missingChannels.emplace_back(totalPaf + msid*(numberBodyParts*2) + i);
        }
        const auto missingPAFChannels = getMissingChannels(mPoseModel, std::vector<float>{}, false);
        for(const auto& index : missingPAFChannels){
            missingChannels.emplace_back(index);
        }


        for (auto i = 0u ; i < lmdbToOpenPoseKeypoints.size() ; i++){
            if (lmdbToOpenPoseKeypoints[i].empty()){
                for(int j=0; j<mStafIDS.size(); j++){
                    missingChannels.emplace_back(totalPaf + j*(numberBodyParts*2) + i*2);
                    missingChannels.emplace_back(totalPaf + j*(numberBodyParts*2) + i*2 + 1);
                }
            }
        }

        for (const auto& index : missingChannels){
            std::fill(&transformedLabel[index*channelOffset],
                    &transformedLabel[index*channelOffset + channelOffset], 0);
        }

        // Background
        const auto type = getType(Dtype(0));
        cv::Mat maskMissTemp(gridY, gridX, type, &transformedLabel[backgroundMaskIndex*channelOffset]);

        // Change BG for COCO
        if(mPoseModel == PoseModel::COCO_21){
            cv::Mat clone = img.clone();
            maskFaceCOCO(maskMissTemp, metaData.jointsSelf.isVisible, metaData.jointsSelf.points, clone);
            maskRealNeckCOCO(maskMissTemp, metaData.jointsSelf.isVisible, metaData.jointsSelf.points, clone);
            for(auto jointOther : metaData.jointsOthers){
                maskRealNeckCOCO(maskMissTemp, jointOther.isVisible, jointOther.points, clone);
                maskFaceCOCO(maskMissTemp, jointOther.isVisible, jointOther.points, clone);
            }
        }

        // Change BG for MPII
        if(mPoseModel == PoseModel::MPII_21){
            cv::Mat clone = img.clone();
            maskFaceMPII(maskMissTemp, metaData.jointsSelf.isVisible, metaData.jointsSelf.points, clone);
            for(auto jointOther : metaData.jointsOthers){
                maskFaceMPII(maskMissTemp, metaData.jointsSelf.isVisible, jointOther.points, clone);
            }
        }
    }

    // PAF
    //Dtype* pafPtr = transformedLabel + (numberTotalChannels * channelOffset);
    const auto& labelMapA = getPafIndexA(mPoseModel);
    const auto& labelMapB = getPafIndexB(mPoseModel);
    const auto threshold = 1;
    const auto diagonal = sqrt(gridX*gridX + gridY*gridY);
    const auto diagonalProportion = (mCurrentEpoch > 0 ? 1.f : metaData.writeNumber/(float)metaData.totalWriteNumber);
    for (auto i = 0 ; i < labelMapA.size() ; i++)
    {
        cv::Mat count = cv::Mat::zeros(gridY, gridX, CV_8UC1);
        // Self
        const auto& joints = metaData.jointsSelf;
        if(joints.isVisible.size()){
            if (joints.isVisible[labelMapA[i]] <= 1 && joints.isVisible[labelMapB[i]] <= 1)
            {
                putVectorMaps(transformedLabel + (numberTotalChannels + 2*i)*channelOffset,
                              transformedLabel + (numberTotalChannels + 2*i + 1)*channelOffset,
                              transformedLabel + 2*i*channelOffset,
                              transformedLabel + (2*i + 1)*channelOffset,
                              // // For Distance
                              // transformedLabel + (2*numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                              // transformedLabel + (numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                              count, joints.points[labelMapA[i]], joints.points[labelMapB[i]],
                        stride, gridX, gridY, param_.sigma(), threshold,
                        diagonal, diagonalProportion);
            }
        }

        // For every other person
        for (auto otherPerson = 0; otherPerson < metaData.numberOtherPeople; otherPerson++)
        {
            const auto& joints = metaData.jointsOthers[otherPerson];
            if (joints.isVisible[labelMapA[i]] <= 1 && joints.isVisible[labelMapB[i]] <= 1)
            {
                putVectorMaps(transformedLabel + (numberTotalChannels + 2*i)*channelOffset,
                              transformedLabel + (numberTotalChannels + 2*i + 1)*channelOffset,
                              transformedLabel + 2*i*channelOffset,
                              transformedLabel + (2*i + 1)*channelOffset,
                              // // For Distance
                              // transformedLabel + (2*numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                              // transformedLabel + (numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                              count, joints.points[labelMapA[i]], joints.points[labelMapB[i]],
                        stride, gridX, gridY, param_.sigma(), threshold,
                        diagonal, diagonalProportion);
            }
        }
    }

    // STAF
    Dtype* stafPtr = transformedLabel + (numberTotalChannels * channelOffset) + (totalPaf * channelOffset);
    for(int mid=0; mid<mStafIDS.size(); mid++){
        int mStafID = mStafIDS[mid];

        Dtype* stafPtrID = stafPtr + mid*numberBodyParts*2*channelOffset;

        const auto threshold = 1;
        for(int j=0; j<getNumberBodyParts(mPoseModel); j++){
            cv::Mat count = cv::Mat::zeros(gridY, gridX, CV_8UC1);

            // Self
            if(metaData.jointsSelfPrev.points.size()){
                if((metaData.jointsSelf.isVisible[j] <= 1 && metaData.jointsSelfPrev.isVisible[mStafID] <= 1)){
                    putVectorMaps(stafPtrID + 2*j*channelOffset,
                                  stafPtrID + ((2*j)+1)*channelOffset,
                                  nullptr,
                                  nullptr,
                                  count, metaData.jointsSelf.points[j], metaData.jointsSelfPrev.points[mStafID],
                                  stride, gridX, gridY, param_.sigma(), threshold, 0, 0, true, false, 0);
                }else if(metaData.jointsSelf.isVisible[j] <= 1 && metaData.jointsSelf.isVisible[mStafID] <= 1){
                    putVectorMaps(stafPtrID + 2*j*channelOffset,
                                  stafPtrID + ((2*j)+1)*channelOffset,
                                  nullptr,
                                  nullptr,
                                  count, metaData.jointsSelf.points[j], metaData.jointsSelf.points[mStafID],
                                  stride, gridX, gridY, param_.sigma(), threshold, 0, 0, true, false, 0);
                }
            }else if(metaData.jointsSelf.points.size()){
                if(metaData.jointsSelf.isVisible[j] <= 1 && metaData.jointsSelf.isVisible[mStafID] <= 1){
                    putVectorMaps(stafPtrID + 2*j*channelOffset,
                                  stafPtrID + ((2*j)+1)*channelOffset,
                                  nullptr,
                                  nullptr,
                                  count, metaData.jointsSelf.points[j], metaData.jointsSelf.points[mStafID],
                                  stride, gridX, gridY, param_.sigma(), threshold, 0, 0, true, false, 0);
                }
            }

            // Others
            for(int i=0; i<metaData.jointsOthers.size(); i++){
                if(metaData.jointsOthersPrev.size()){
                    if(metaData.jointsOthersPrev[i].points.size()){
                        if((metaData.jointsOthers[i].isVisible[j] <= 1 && metaData.jointsOthersPrev[i].isVisible[mStafID] <= 1)){
                            putVectorMaps(stafPtrID + 2*j*channelOffset,
                                          stafPtrID + ((2*j)+1)*channelOffset,
                                          nullptr,
                                          nullptr,
                                          count, metaData.jointsOthers[i].points[j], metaData.jointsOthersPrev[i].points[mStafID],
                                          stride, gridX, gridY, param_.sigma(), threshold, 0, 0, true, false, 0);
                        }else if(metaData.jointsOthers[i].isVisible[j] <= 1 && metaData.jointsOthers[i].isVisible[mStafID] <= 1){
                            putVectorMaps(stafPtrID + 2*j*channelOffset,
                                          stafPtrID + ((2*j)+1)*channelOffset,
                                          nullptr,
                                          nullptr,
                                          count, metaData.jointsOthers[i].points[j], metaData.jointsOthers[i].points[mStafID],
                                          stride, gridX, gridY, param_.sigma(), threshold, 0, 0, true, false, 0);
                        }
                    }else{
                        if(metaData.jointsOthers[i].isVisible[j] <= 1 && metaData.jointsOthers[i].isVisible[mStafID] <= 1){
                            putVectorMaps(stafPtrID + 2*j*channelOffset,
                                          stafPtrID + ((2*j)+1)*channelOffset,
                                          nullptr,
                                          nullptr,
                                          count, metaData.jointsOthers[i].points[j], metaData.jointsOthers[i].points[mStafID],
                                          stride, gridX, gridY, param_.sigma(), threshold, 0, 0, true, false, 0);
                        }
                    }
                }else{
                    if(metaData.jointsOthers[i].isVisible[j] <= 1 && metaData.jointsOthers[i].isVisible[mStafID] <= 1){
                        putVectorMaps(stafPtrID + 2*j*channelOffset,
                                      stafPtrID + ((2*j)+1)*channelOffset,
                                      nullptr,
                                      nullptr,
                                      count, metaData.jointsOthers[i].points[j], metaData.jointsOthers[i].points[mStafID],
                                      stride, gridX, gridY, param_.sigma(), threshold, 0, 0, true, false, 0);
                    }
                }
            }

        }
    }


    // Body parts
    Dtype* hmPtr = transformedLabel + (numberTotalChannels * channelOffset) + (totalPaf * channelOffset) + (totalStaf * channelOffset);
    for (auto part = 0; part < numberBodyParts; part++)
    {
        // Self
        if(metaData.jointsSelf.isVisible.size()){
            if (metaData.jointsSelf.isVisible[part] <= 1)
            {
                const auto& centerPoint = metaData.jointsSelf.points[part];
                putGaussianMaps(hmPtr + (part)*channelOffset,
                                centerPoint, stride, gridX, gridY, param_.sigma());
            }
        }
        // For every other person
        for (auto otherPerson = 0; otherPerson < metaData.numberOtherPeople; otherPerson++)
        {
            if (metaData.jointsOthers[otherPerson].isVisible[part] <= 1)
            {
                const auto& centerPoint = metaData.jointsOthers[otherPerson].points[part];
                putGaussianMaps(hmPtr + (part)*channelOffset,
                                centerPoint, stride, gridX, gridY, param_.sigma());
            }
        }
    }

    // Background channel
    // Naive implementation
    for (auto gY = 0; gY < gridY; gY++)
    {
        const auto yOffset = gY*gridX;
        for (auto gX = 0; gX < gridX; gX++)
        {
            const auto xyOffset = yOffset + gX;
            Dtype maximum = 0.;
            for (auto part = (numberTotalChannels) + (totalPaf) + (totalStaf) ; part < backgroundIndex ; part++)
            {
                const auto index = part * channelOffset + xyOffset;
                maximum = (maximum > transformedLabel[index]) ? maximum : transformedLabel[index];
            }
            transformedLabel[backgroundIndex*channelOffset + xyOffset] = std::max(Dtype(1.)-maximum, Dtype(0.));
        }
    }

}


template<typename Dtype>
void OPDataTransformer<Dtype>::generateLabelMapStaf(Dtype* transformedLabel, const cv::Size& imageSize, const cv::Mat& maskMiss,
                                                const MetaData& metaData, const cv::Mat& img, const int stride) const
{
    // Label size = image size / stride
    const auto rezX = (int)imageSize.width;
    const auto rezY = (int)imageSize.height;
    const auto gridX = rezX / stride;
    const auto gridY = rezY / stride;
    const auto channelOffset = gridY * gridX;
    const auto numberBodyParts = getNumberBodyParts(mPoseModel); // #BP
    const int totalStaf = getNumberBodyParts(mPoseModel) * 2 * mStafIDS.size();
    const int numberTotalChannels = getNumberChannels() / 2;
    const int backgroundMaskIndex = numberTotalChannels-1;
    const int backgroundIndex = getNumberChannels() - 1;

    // Labels to 0
    std::fill(transformedLabel, transformedLabel + getNumberChannels() * gridY * gridX, 0.f);

    // Initialize labels to [0, 1] (depending on maskMiss)
    fillMaskChannels(transformedLabel, gridX, gridY, numberTotalChannels, channelOffset, maskMiss);

    // Masking out channels - For COCO_YY_ZZ models (ZZ < YY)
    std::vector<int> missingChannels;
    std::vector<int> missingSID;
    if (numberBodyParts > getNumberBodyPartsLmdb(mPoseModel))
    {
        // Remove BP/PAF non-labeled channels
        const auto& lmdbToOpenPoseKeypoints = getLmdbToOpenPoseKeypoints(mPoseModel);
        for (auto i = 0u ; i < lmdbToOpenPoseKeypoints.size() ; i++)
            if (lmdbToOpenPoseKeypoints[i].empty()){
                missingChannels.emplace_back(totalStaf + i);

                for(int j=0; j<mStafIDS.size(); j++) if(i == mStafIDS[j]) missingSID.emplace_back(j);
            }
        for(auto msid : missingSID){
            for(int i=0; i<numberBodyParts*2; i++)
                missingChannels.emplace_back(msid*(numberBodyParts*2) + i);
        }
        for (const auto& index : missingChannels){
            std::fill(&transformedLabel[index*channelOffset],
                    &transformedLabel[index*channelOffset + channelOffset], 0);
        }

        // Background
        const auto type = getType(Dtype(0));
        cv::Mat maskMissTemp(gridY, gridX, type, &transformedLabel[backgroundMaskIndex*channelOffset]);

        // Change BG for COCO
        if(mPoseModel == PoseModel::COCO_21){
            cv::Mat clone = img.clone();
            maskFaceCOCO(maskMissTemp, metaData.jointsSelf.isVisible, metaData.jointsSelf.points, clone);
            maskRealNeckCOCO(maskMissTemp, metaData.jointsSelf.isVisible, metaData.jointsSelf.points, clone);
            for(auto jointOther : metaData.jointsOthers){
                maskRealNeckCOCO(maskMissTemp, jointOther.isVisible, jointOther.points, clone);
                maskFaceCOCO(maskMissTemp, jointOther.isVisible, jointOther.points, clone);
            }
        }

        // Change BG for MPII
        if(mPoseModel == PoseModel::MPII_21){
            cv::Mat clone = img.clone();
            maskFaceMPII(maskMissTemp, metaData.jointsSelf.isVisible, metaData.jointsSelf.points, clone);
            for(auto jointOther : metaData.jointsOthers){
                maskFaceMPII(maskMissTemp, metaData.jointsSelf.isVisible, jointOther.points, clone);
            }
        }
    }

    // STAF
    Dtype* stafPtr = transformedLabel + (numberTotalChannels * channelOffset);
    for(int mid=0; mid<mStafIDS.size(); mid++){
        int mStafID = mStafIDS[mid];

        Dtype* stafPtrID = stafPtr + mid*numberBodyParts*2*channelOffset;

        const auto threshold = 1;
        for(int j=0; j<getNumberBodyParts(mPoseModel); j++){
            cv::Mat count = cv::Mat::zeros(gridY, gridX, CV_8UC1);

            // Self
            if(metaData.jointsSelfPrev.points.size()){
                if((metaData.jointsSelf.isVisible[j] <= 1 && metaData.jointsSelfPrev.isVisible[mStafID] <= 1)){
                    putVectorMaps(stafPtrID + 2*j*channelOffset,
                                  stafPtrID + ((2*j)+1)*channelOffset,
                                  nullptr,
                                  nullptr,
                                  count, metaData.jointsSelf.points[j], metaData.jointsSelfPrev.points[mStafID],
                                  stride, gridX, gridY, param_.sigma(), threshold, 0, 0, true, false, 0);
                }else if(metaData.jointsSelf.isVisible[j] <= 1 && metaData.jointsSelf.isVisible[mStafID] <= 1){
                    putVectorMaps(stafPtrID + 2*j*channelOffset,
                                  stafPtrID + ((2*j)+1)*channelOffset,
                                  nullptr,
                                  nullptr,
                                  count, metaData.jointsSelf.points[j], metaData.jointsSelf.points[mStafID],
                                  stride, gridX, gridY, param_.sigma(), threshold, 0, 0, true, false, 0);
                }
            }else if(metaData.jointsSelf.points.size()){
                if(metaData.jointsSelf.isVisible[j] <= 1 && metaData.jointsSelf.isVisible[mStafID] <= 1){
                    putVectorMaps(stafPtrID + 2*j*channelOffset,
                                  stafPtrID + ((2*j)+1)*channelOffset,
                                  nullptr,
                                  nullptr,
                                  count, metaData.jointsSelf.points[j], metaData.jointsSelf.points[mStafID],
                                  stride, gridX, gridY, param_.sigma(), threshold, 0, 0, true, false, 0);
                }
            }

            // Others
            for(int i=0; i<metaData.jointsOthers.size(); i++){
                if(metaData.jointsOthersPrev.size()){
                    if(metaData.jointsOthersPrev[i].points.size()){
                        if((metaData.jointsOthers[i].isVisible[j] <= 1 && metaData.jointsOthersPrev[i].isVisible[mStafID] <= 1)){
                            putVectorMaps(stafPtrID + 2*j*channelOffset,
                                          stafPtrID + ((2*j)+1)*channelOffset,
                                          nullptr,
                                          nullptr,
                                          count, metaData.jointsOthers[i].points[j], metaData.jointsOthersPrev[i].points[mStafID],
                                          stride, gridX, gridY, param_.sigma(), threshold, 0, 0, true, false, 0);
                        }else if(metaData.jointsOthers[i].isVisible[j] <= 1 && metaData.jointsOthers[i].isVisible[mStafID] <= 1){
                            putVectorMaps(stafPtrID + 2*j*channelOffset,
                                          stafPtrID + ((2*j)+1)*channelOffset,
                                          nullptr,
                                          nullptr,
                                          count, metaData.jointsOthers[i].points[j], metaData.jointsOthers[i].points[mStafID],
                                          stride, gridX, gridY, param_.sigma(), threshold, 0, 0, true, false, 0);
                        }
                    }else{
                        if(metaData.jointsOthers[i].isVisible[j] <= 1 && metaData.jointsOthers[i].isVisible[mStafID] <= 1){
                            putVectorMaps(stafPtrID + 2*j*channelOffset,
                                          stafPtrID + ((2*j)+1)*channelOffset,
                                          nullptr,
                                          nullptr,
                                          count, metaData.jointsOthers[i].points[j], metaData.jointsOthers[i].points[mStafID],
                                          stride, gridX, gridY, param_.sigma(), threshold, 0, 0, true, false, 0);
                        }
                    }
                }else{
                    if(metaData.jointsOthers[i].isVisible[j] <= 1 && metaData.jointsOthers[i].isVisible[mStafID] <= 1){
                        putVectorMaps(stafPtrID + 2*j*channelOffset,
                                      stafPtrID + ((2*j)+1)*channelOffset,
                                      nullptr,
                                      nullptr,
                                      count, metaData.jointsOthers[i].points[j], metaData.jointsOthers[i].points[mStafID],
                                      stride, gridX, gridY, param_.sigma(), threshold, 0, 0, true, false, 0);
                    }
                }
            }

        }
    }




    // Body parts
    Dtype* hmPtr = transformedLabel + (numberTotalChannels * channelOffset) + (totalStaf * channelOffset);
    for (auto part = 0; part < numberBodyParts; part++)
    {
        // Self
        if(metaData.jointsSelf.isVisible.size()){
            if (metaData.jointsSelf.isVisible[part] <= 1)
            {
                const auto& centerPoint = metaData.jointsSelf.points[part];
                putGaussianMaps(hmPtr + (part)*channelOffset,
                                centerPoint, stride, gridX, gridY, param_.sigma());
            }
        }
        // For every other person
        for (auto otherPerson = 0; otherPerson < metaData.numberOtherPeople; otherPerson++)
        {
            if (metaData.jointsOthers[otherPerson].isVisible[part] <= 1)
            {
                const auto& centerPoint = metaData.jointsOthers[otherPerson].points[part];
                putGaussianMaps(hmPtr + (part)*channelOffset,
                                centerPoint, stride, gridX, gridY, param_.sigma());
            }
        }
    }

    // Background channel
    // Naive implementation
    for (auto gY = 0; gY < gridY; gY++)
    {
        const auto yOffset = gY*gridX;
        for (auto gX = 0; gX < gridX; gX++)
        {
            const auto xyOffset = yOffset + gX;
            Dtype maximum = 0.;
            for (auto part = (numberTotalChannels) + (totalStaf) ; part < backgroundIndex ; part++)
            {
                const auto index = part * channelOffset + xyOffset;
                maximum = (maximum > transformedLabel[index]) ? maximum : transformedLabel[index];
            }
            transformedLabel[backgroundIndex*channelOffset + xyOffset] = std::max(Dtype(1.)-maximum, Dtype(0.));
        }
    }

}

template<typename Dtype>
void OPDataTransformer<Dtype>::generateLabelMap(Dtype* transformedLabel, const cv::Size& imageSize, const cv::Mat& maskMiss,
                                                const MetaData& metaData, const cv::Mat& img, const int stride) const
{
    // Label size = image size / stride
    const auto rezX = (int)imageSize.width;
    const auto rezY = (int)imageSize.height;
    const auto gridX = rezX / stride;
    const auto gridY = rezY / stride;
    const auto channelOffset = gridY * gridX;
    const auto numberBodyParts = getNumberBodyParts(mPoseModel); // #BP
    const auto numberPafChannels = getNumberPafChannels(mPoseModel); // 2 x #PAF
    const auto numberTotalChannels = getNumberBodyBkgAndPAF(mPoseModel); // numberBodyParts + numberPafChannels + 1
    // // For Distance
    // const auto numberTotalChannels = getNumberBodyBkgAndPAF(mPoseModel) + (numberPafChannels / 2); // numberBodyParts + numberPafChannels + 1

    // Labels to 0
    if(mTpaf)
        std::fill(transformedLabel, transformedLabel + getNumberChannels() * gridY * gridX, 0.f);
    else
        std::fill(transformedLabel, transformedLabel + 2*numberTotalChannels * gridY * gridX, 0.f);

    // Initialize labels to [0, 1] (depending on maskMiss)
    fillMaskChannels(transformedLabel, gridX, gridY, numberTotalChannels, channelOffset, maskMiss);

    // Masking out channels - For COCO_YY_ZZ models (ZZ < YY)
    if (numberBodyParts > getNumberBodyPartsLmdb(mPoseModel) || mPoseModel == PoseModel::MPII_hands_59)
    {
        // Remove BP/PAF non-labeled channels
        const auto missingChannels = getMissingChannels(mPoseModel, (mPoseModel == PoseModel::MPII_hands_59
                                                                     ? metaData.jointsSelf.isVisible
                                                                     : std::vector<float>{}));
        for (const auto& index : missingChannels)
            std::fill(&transformedLabel[index*channelOffset],
                    &transformedLabel[index*channelOffset + channelOffset], 0);
        // Background
        const auto type = getType(Dtype(0));
        const auto backgroundIndex = numberPafChannels + numberBodyParts;
        cv::Mat maskMissTemp(gridY, gridX, type, &transformedLabel[backgroundIndex*channelOffset]);
        // If hands
        if (numberBodyParts == 59 && mPoseModel != PoseModel::MPII_hands_59)
        {
            maskHands(maskMissTemp, metaData.jointsSelf.isVisible, metaData.jointsSelf.points, stride, 0.6f);
            for (const auto& jointsOther : metaData.jointsOthers)
                maskHands(maskMissTemp, jointsOther.isVisible, jointsOther.points, stride, 0.6f);
        }
        // If foot
        if (numberBodyParts == 23)
        {
            maskFeet(maskMissTemp, metaData.jointsSelf.isVisible, metaData.jointsSelf.points, stride, 0.6f);
            for (const auto& jointsOther : metaData.jointsOthers)
                maskFeet(maskMissTemp, jointsOther.isVisible, jointsOther.points, stride, 0.6f);
        }

        // Change BG for COCO
        if(mPoseModel == PoseModel::COCO_21){
            cv::Mat clone = img.clone();
            maskFaceCOCO(maskMissTemp, metaData.jointsSelf.isVisible, metaData.jointsSelf.points, clone);
            maskRealNeckCOCO(maskMissTemp, metaData.jointsSelf.isVisible, metaData.jointsSelf.points, clone);
            for(auto jointOther : metaData.jointsOthers){
                maskRealNeckCOCO(maskMissTemp, jointOther.isVisible, jointOther.points, clone);
                maskFaceCOCO(maskMissTemp, jointOther.isVisible, jointOther.points, clone);
            }

            //            if(metaData.writeNumber == 8){
            //                drawPoints(clone, metaData);
            //                cv::imwrite("/home/ryaadhav/test.png", clone);
            //                cv::imwrite("/home/ryaadhav/mask.png", maskMissTemp*255);
            //                exit(-1);
            //            }
        }

        // Change BG for MPII
        if(mPoseModel == PoseModel::MPII_21){
            cv::Mat clone = img.clone();
            maskFaceMPII(maskMissTemp, metaData.jointsSelf.isVisible, metaData.jointsSelf.points, clone);
            for(auto jointOther : metaData.jointsOthers){
                maskFaceMPII(maskMissTemp, metaData.jointsSelf.isVisible, jointOther.points, clone);
            }

            //            if(metaData.writeNumber == 4){
            //                drawPoints(clone, metaData);
            //                cv::imwrite("/home/ryaadhav/test.png", clone);
            //                cv::imwrite("/home/ryaadhav/mask.png", maskMissTemp*255);
            //            }

        }

        // Change BG for PT
        if(mPoseModel == PoseModel::PT_21){
            cv::Mat clone = img.clone();
            // NOT DONE
        }
    }

    // TODO: Remove, temporary hack to get foot data, do nicely for 6-keypoint foot
    // Remove if required RBigToe, RSmallToe, LBigToe, LSmallToe, and Background
    if (mPoseModel == PoseModel::COCO_23 || mPoseModel == PoseModel::DOME_23_19 || mPoseModel == PoseModel::COCO_23_17)
    {
        std::vector<int> indexesToRemove;
        // PAFs
        for (const auto& index : {11, 12, 15, 16})
        {
            const auto indexBase = 2*index;
            indexesToRemove.emplace_back(indexBase);
            indexesToRemove.emplace_back(indexBase+1);
        }
        // Body parts
        for (const auto& index : {11, 12, 16, 17})
        {
            const auto indexBase = numberPafChannels + index;
            indexesToRemove.emplace_back(indexBase);
        }
        // Included in code 10-30 lines above...
        // // Dome data: Exclude (unlabeled) foot keypoints
        // if (mPoseModel == PoseModel::DOME_23_19 || mPoseModel == PoseModel::COCO_23_17)
        // {
        //     // Remove those channels
        //     for (const auto& index : indexesToRemove)
        //     {
        //         std::fill(&transformedLabel[index*channelOffset],
        //                   &transformedLabel[index*channelOffset + channelOffset], 0);
        //     }
        // }
        // // Background
        // if (mPoseModel == PoseModel::DOME_23_19 || mPoseModel == PoseModel::COCO_23_17)
        // {
        //     const auto backgroundIndex = numberPafChannels + numberBodyParts;
        //     int type;
        //     const auto type = getType(Dtype(0));
        //     cv::Mat maskMiss(gridY, gridX, type, &transformedLabel[backgroundIndex*channelOffset]);
        //     maskFeet(maskMiss, metaData.jointsSelf.isVisible, metaData.jointsSelf.points, stride, 0.6f);
        //     for (const auto& jointsOther : metaData.jointsOthers)
        //         maskFeet(maskMiss, jointsOther.isVisible, jointsOther.points, stride, 0.6f);
        // }
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
                            const auto type = getType(Dtype(0));
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
            //     for (auto part = 0; part < 2*numberTotalChannels; part++)
            //     {
            //         // Reduce #images saved (ideally images from 0 to numberTotalChannels should be the same)
            //         // if (part >= 11*2)
            //         if (part >= 22 && part <= numberTotalChannels)
            //         // if (part < 3 || part >= numberTotalChannels - 3)
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

    // Fill masks for tpaf if valid
    if(mTpaf){
        // HARDCODED TO 21
        int hmWeightStart = 2 * (getNumberBodyParts(PoseModel::COCO_21) + 1);
        int tpafWeightStart = 2 * getNumberBodyBkgAndPAF(PoseModel::COCO_21);
        for(int i=0; i<getNumberBodyParts(PoseModel::COCO_21); i++){
            memcpy(transformedLabel + tpafWeightStart*gridY*gridX + 2*i*gridY*gridX,
                   transformedLabel + hmWeightStart*gridY*gridX + i*gridY*gridX, sizeof(Dtype)*gridY*gridX);
            memcpy(transformedLabel + tpafWeightStart*gridY*gridX + (2*i+1)*gridY*gridX,
                   transformedLabel + hmWeightStart*gridY*gridX + i*gridY*gridX, sizeof(Dtype)*gridY*gridX);
        }

        // TPAFs
        // Fill from data of other person metaData (and self metaData)
        Dtype* tpafMaskStartPointer = transformedLabel + (2 * getNumberBodyBkgAndPAF(mPoseModel) * channelOffset);
        Dtype* tpafStartPointer = transformedLabel + ((2 * getNumberBodyBkgAndPAF(mPoseModel) + getNumberBodyParts(mPoseModel) * 2) * channelOffset);

        // ASK!! Resolution too low for direction? perhaps distance between too small to encode?
        // What about meaning of count? Iterate joint, then person, or person then joint
        // Delta value is too noisy for no change?
        // Encode angles-scalar?
        // Encode x,y change directly?

        const auto threshold = 1;
        for(int j=0; j<getNumberBodyParts(mPoseModel); j++){
            cv::Mat count = cv::Mat::zeros(gridY, gridX, CV_8UC1);

            // Self
            if(metaData.jointsSelfPrev.points.size()){
                if((metaData.jointsSelf.isVisible[j] <= 1 && metaData.jointsSelfPrev.isVisible[j] <= 1)){
                    putVectorMaps(tpafStartPointer + 2*j*channelOffset,
                                  tpafStartPointer + ((2*j)+1)*channelOffset,
                                  tpafMaskStartPointer + 2*j*channelOffset,
                                  tpafMaskStartPointer + ((2*j)+1)*channelOffset,
                                  count, metaData.jointsSelf.points[j], metaData.jointsSelfPrev.points[j],
                                  stride, gridX, gridY, param_.sigma(), threshold, 0, 0, true, false, 0);
                }
            }

            // Others
            for(int i=0; i<metaData.jointsOthers.size(); i++){
                if(!metaData.jointsOthersPrev.size()) continue;
                if(!metaData.jointsOthersPrev[i].points.size()) continue;
                if(!(metaData.jointsOthers[i].isVisible[j] <= 1 && metaData.jointsOthersPrev[i].isVisible[j] <= 1)) continue;
                putVectorMaps(tpafStartPointer + 2*j*channelOffset,
                              tpafStartPointer + ((2*j)+1)*channelOffset,
                              tpafMaskStartPointer + 2*j*channelOffset,
                              tpafMaskStartPointer + ((2*j)+1)*channelOffset,
                              count, metaData.jointsOthers[i].points[j], metaData.jointsOthersPrev[i].points[j],
                              stride, gridX, gridY, param_.sigma(), threshold, 0, 0, true, false, 0);
            }

        }

//        for(int i=0; i<metaData.jointsOthers.size(); i++){
//            Joints& currentJoints = metaData.jointsOthers[i];
//            Joints& prevJoints = metaData.jointsOthersPrev[i];
//            if(!prevJoints.points.size() || !currentJoints.points.size()) continue;
//            for(int j=0; j<prevJoints.points.size(); j++){
//                if(!(currentJoints.isVisible[j] <= 1 && prevJoints.isVisible[j] <= 1)) continue;


//                putVectorMaps(tpafStartPointer + 2*j*channelOffset,
//                              tpafStartPointer + ((2*j)+1)*channelOffset,
//                              tpafMaskStartPointer + 2*j*channelOffset,
//                              tpafMaskStartPointer + ((2*j)+1)*channelOffset,);
//            }
//        }

    }

    // PAFs
    const auto& labelMapA = getPafIndexA(mPoseModel);
    const auto& labelMapB = getPafIndexB(mPoseModel);
    const auto threshold = 1;
    const auto diagonal = sqrt(gridX*gridX + gridY*gridY);
    const auto diagonalProportion = (mCurrentEpoch > 0 ? 1.f : metaData.writeNumber/(float)metaData.totalWriteNumber);
    for (auto i = 0 ; i < labelMapA.size() ; i++)
    {
        cv::Mat count = cv::Mat::zeros(gridY, gridX, CV_8UC1);
        // Self
        const auto& joints = metaData.jointsSelf;
        if(joints.isVisible.size()){
            if (joints.isVisible[labelMapA[i]] <= 1 && joints.isVisible[labelMapB[i]] <= 1)
            {
                putVectorMaps(transformedLabel + (numberTotalChannels + 2*i)*channelOffset,
                              transformedLabel + (numberTotalChannels + 2*i + 1)*channelOffset,
                              transformedLabel + 2*i*channelOffset,
                              transformedLabel + (2*i + 1)*channelOffset,
                              // // For Distance
                              // transformedLabel + (2*numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                              // transformedLabel + (numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                              count, joints.points[labelMapA[i]], joints.points[labelMapB[i]],
                        stride, gridX, gridY, param_.sigma(), threshold,
                        diagonal, diagonalProportion);
            }
        }

        // For every other person
        for (auto otherPerson = 0; otherPerson < metaData.numberOtherPeople; otherPerson++)
        {
            const auto& joints = metaData.jointsOthers[otherPerson];
            if (joints.isVisible[labelMapA[i]] <= 1 && joints.isVisible[labelMapB[i]] <= 1)
            {
                putVectorMaps(transformedLabel + (numberTotalChannels + 2*i)*channelOffset,
                              transformedLabel + (numberTotalChannels + 2*i + 1)*channelOffset,
                              transformedLabel + 2*i*channelOffset,
                              transformedLabel + (2*i + 1)*channelOffset,
                              // // For Distance
                              // transformedLabel + (2*numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                              // transformedLabel + (numberTotalChannels - numberPafChannels/2 + i)*channelOffset,
                              count, joints.points[labelMapA[i]], joints.points[labelMapB[i]],
                        stride, gridX, gridY, param_.sigma(), threshold,
                        diagonal, diagonalProportion);
            }
        }
    }
    // // Re-normalize masks (otherwise PAF explodes)
    // const auto finalImageArea = gridX*gridY;
    // for (auto i = 0 ; i < labelMapA.size() ; i++)
    // {
    //     auto* initPoint = &transformedLabel[2*i*channelOffset];
    //     const auto accumulation = std::accumulate(initPoint, initPoint+channelOffset, 0);
    //     const auto ratio = finalImageArea / (float)accumulation;
    //     if (ratio > 1.01 || ratio < 0.99)
    //         std::transform(initPoint, initPoint + 2*channelOffset, initPoint, std::bind1st(std::multiplies<Dtype>(), ratio)) ;
    // }

    // Body parts
    for (auto part = 0; part < numberBodyParts; part++)
    {
        // Self
        if(metaData.jointsSelf.isVisible.size()){
            if (metaData.jointsSelf.isVisible[part] <= 1)
            {
                const auto& centerPoint = metaData.jointsSelf.points[part];
                putGaussianMaps(transformedLabel + (part+numberTotalChannels+numberPafChannels)*channelOffset,
                                centerPoint, stride, gridX, gridY, param_.sigma());
            }
        }
        // For every other person
        for (auto otherPerson = 0; otherPerson < metaData.numberOtherPeople; otherPerson++)
        {
            if (metaData.jointsOthers[otherPerson].isVisible[part] <= 1)
            {
                const auto& centerPoint = metaData.jointsOthers[otherPerson].points[part];
                putGaussianMaps(transformedLabel + (part+numberTotalChannels+numberPafChannels)*channelOffset,
                                centerPoint, stride, gridX, gridY, param_.sigma());
            }
        }
    }

    // Background channel
    // Naive implementation
    for (auto gY = 0; gY < gridY; gY++)
    {
        const auto yOffset = gY*gridX;
        for (auto gX = 0; gX < gridX; gX++)
        {
            const auto xyOffset = yOffset + gX;
            Dtype maximum = 0.;
            const auto backgroundIndex = numberTotalChannels+numberPafChannels+numberBodyParts;
            for (auto part = numberTotalChannels+numberPafChannels ; part < backgroundIndex ; part++)
            {
                const auto index = part * channelOffset + xyOffset;
                maximum = (maximum > transformedLabel[index]) ? maximum : transformedLabel[index];
            }
            transformedLabel[backgroundIndex*channelOffset + xyOffset] = std::max(Dtype(1.)-maximum, Dtype(0.));
        }
    }
}

template<typename Dtype>
void OPDataTransformer<Dtype>::putGaussianMaps(Dtype* entry, const cv::Point2f& centerPoint, const int stride,
                                               const int gridX, const int gridY, const float sigma) const
{
    //LOG(INFO) << "putGaussianMaps here we start for " << centerPoint.x << " " << centerPoint.y;
    const Dtype start = stride/2.f - 0.5f; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
    const auto multiplier = 2.0 * sigma * sigma;
    for (auto gY = 0; gY < gridY; gY++)
    {
        const auto yOffset = gY*gridX;
        const Dtype y = start + gY * stride;
        const auto yMenosCenterPointSquared = (y-centerPoint.y)*(y-centerPoint.y);
        for (auto gX = 0; gX < gridX; gX++)
        {
            const Dtype x = start + gX * stride;
            const Dtype d2 = (x-centerPoint.x)*(x-centerPoint.x) + yMenosCenterPointSquared;
            const Dtype exponent = d2 / multiplier;
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
void OPDataTransformer<Dtype>::putVectorMaps(Dtype* entryX, Dtype* entryY, Dtype* maskX, Dtype* maskY,
                                             cv::Mat& count, const cv::Point2f& centerA,
                                             const cv::Point2f& centerB, const int stride, const int gridX,
                                             const int gridY, const float sigma, const int threshold,
                                             const int diagonal, const float diagonalProportion,
                                             const bool normalize, const bool demask, const float tanval) const
// void OPDataTransformer<Dtype>::putVectorMaps(Dtype* entryX, Dtype* entryY, Dtype* entryD, Dtype* entryDMask,
//                                              cv::Mat& count, const cv::Point2f& centerA,
//                                              const cv::Point2f& centerB, const int stride, const int gridX,
//                                              const int gridY, const float sigma, const int threshold) const
{
    const auto scaleLabel = Dtype(1)/Dtype(stride);
    const auto centerALabelScale = scaleLabel * centerA;
    const auto centerBLabelScale = scaleLabel * centerB;
    cv::Point2f directionAB = centerBLabelScale - centerALabelScale;
    const auto distanceAB = std::sqrt(directionAB.x*directionAB.x + directionAB.y*directionAB.y);
    if(normalize) directionAB *= (Dtype(1) / distanceAB);

    if(tanval){
        cv::Point2f directionABOrig = centerB - centerA;
        const auto distanceABOrig = std::sqrt(directionABOrig.x*directionABOrig.x + directionABOrig.y*directionABOrig.y);
        directionAB *= tanh(tanval * distanceABOrig);
    }

    // // For Distance
    // const auto dMin = Dtype(0);
    // const auto dMax = Dtype(std::sqrt(gridX*gridX + gridY*gridY));
    // const auto dRange = dMax - dMin;
    // const auto entryDValue = 2*(distanceAB - dMin)/dRange - 1; // Main range: [-1, 1], -1 is 0px-distance, 1 is 368 / stride x sqrt(2) px of distance

    // If PAF is not 0 or NaN (e.g. if PAF perpendicular to image plane)
    if (!isnan(directionAB.x) && !isnan(directionAB.y))
    {
        int minX = std::max(0,
                                  int(std::round(std::min(centerALabelScale.x, centerBLabelScale.x) - threshold)));
        int maxX = std::min(gridX,
                                  int(std::round(std::max(centerALabelScale.x, centerBLabelScale.x) + threshold)));
        int minY = std::max(0,
                                  int(std::round(std::min(centerALabelScale.y, centerBLabelScale.y) - threshold)));
        int maxY = std::min(gridY,
                                  int(std::round(std::max(centerALabelScale.y, centerBLabelScale.y) + threshold)));
        (void)diagonalProportion;
        (void)diagonal;
        (void)entryX;
        (void)entryY;

        //minX = 0; maxX = gridX; minY = 0; maxY = gridY;

        // const auto weight = (1-diagonalProportion) + diagonalProportion * diagonal/distanceAB; // alpha*1 + (1-alpha)*realProportion
        for (auto gY = minY; gY < maxY; gY++)
        {
            const auto yOffset = gY*gridX;
            const auto gYMenosCenterALabelScale = gY - centerALabelScale.y;
            for (auto gX = minX; gX < maxX; gX++)
            {
                const auto xyOffset = yOffset + gX;
                const cv::Point2f ba{gX - centerALabelScale.x, gYMenosCenterALabelScale};
                const float distance = std::abs(ba.x*directionAB.y - ba.y*directionAB.x);
                if (distance <= threshold)
                {
                    auto& counter = count.at<uchar>(gY, gX);
                    if (counter == 0)
                    {
                        entryX[xyOffset] = directionAB.x;
                        entryY[xyOffset] = directionAB.y;
                        // Weight makes small PAFs as important as big PAFs
                        // maskX[xyOffset] *= weight;
                        // maskY[xyOffset] *= weight;
                        // // For Distance
                        // entryD[xyOffset] = entryDValue;
                        // entryDMask[xyOffset] = Dtype(1);
                    }
                    else
                    {
                        entryX[xyOffset] = (entryX[xyOffset]*counter + directionAB.x) / (counter + 1);
                        entryY[xyOffset] = (entryY[xyOffset]*counter + directionAB.y) / (counter + 1);
                        // // For Distance
                        // entryD[xyOffset] = (entryD[xyOffset]*counter + entryDValue) / (counter + 1);
                    }
                    counter++;
                }
            }
        }

        if(demask){
            for (auto gY = 0; gY < gridY; gY++)
            {
                for (auto gX = 0; gX < gridX; gX++)
                {
                    const auto xyOffset = gY*gridX + gX;
                     maskX[xyOffset] *= 0;
                     maskY[xyOffset] *= 0;
                }
            }
        }
    }
}
// OpenPose: added end

INSTANTIATE_CLASS(OPDataTransformer);

}  // namespace caffe
