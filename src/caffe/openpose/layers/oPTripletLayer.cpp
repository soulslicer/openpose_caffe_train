#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"
// OpenPose: added
#include <chrono>
#include <stdexcept>
#include "caffe/util/io.hpp" // DecodeDatum, DecodeDatumNative
#include "caffe/openpose/getLine.hpp"
#include "caffe/openpose/layers/oPTripletLayer.hpp"
#include <opencv2/opencv.hpp>
// OpenPose: added end

#include <iostream>
using namespace std;

namespace caffe {

template <typename Dtype>
OPTripletLayer<Dtype>::OPTripletLayer(const LayerParameter& param) :
    BasePrefetchingDataLayer<Dtype>(param),
    offset_(),
    offsetSecond(), // OpenPose: added
    op_transform_param_(param.op_transform_param()) // OpenPose: added
{
    // LOAD THE TEXT FILE HERE
//    db_.reset(db::GetDB(param.data_param().backend()));
//    db_->Open(param.data_param().source(), db::READ);
//    cursor_.reset(db_->NewCursor());

    // OpenPose: added
    mOnes = 0;
    mTwos = 0;
    // Set up secondary DB
    if (!param.op_transform_param().source_secondary().empty())
    {
        secondDb = true;
        secondProbability = param.op_transform_param().prob_secondary();
        CHECK_GE(secondProbability, 0.f);
        CHECK_LE(secondProbability, 1.f);
        dbSecond.reset(db::GetDB(DataParameter_DB::DataParameter_DB_LMDB));
        dbSecond->Open(param.op_transform_param().source_secondary(), db::READ);
        cursorSecond.reset(dbSecond->NewCursor());
    }
    else
    {
        secondDb = false;
        secondProbability = 0.f;
    }
    // Set up negatives DB
    if (!param.op_transform_param().source_background().empty())
    {
        backgroundDb = true;
        dbBackground.reset(db::GetDB(DataParameter_DB::DataParameter_DB_LMDB));
        dbBackground->Open(param.op_transform_param().source_background(), db::READ);
        cursorBackground.reset(dbBackground->NewCursor());
    }
    else
        backgroundDb = false;
    // Timer
    mDuration = 0;
    mCounter = 0;
    // OpenPose: added end
}

template <typename Dtype>
OPTripletLayer<Dtype>::~OPTripletLayer()
{
    this->StopInternalThread();
}


template <typename Dtype>
void OPTripletLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    std::string train_source = this->layer_param().data_param().source();
    std::ifstream file(train_source + "train_info.txt");
    std::string str;
    while (std::getline(file, str))
    {
        std::vector<std::string> splitString;
        boost::split(splitString,str,boost::is_any_of(" "));
        int person_id = std::stoi(splitString[0]);
        std::string image_path = splitString[1];
        if(!reidData.count(person_id)) reidData[person_id] = std::vector<std::string>();
        else reidData[person_id].emplace_back(train_source + image_path);
    }
    file.close();
    for (auto& kv : reidData) {
        reidKeys.emplace_back(kv.first);
    }


    mOPDataTransformer.reset(new OPDataTransformer<Dtype>(op_transform_param_));

    const int batch_size = this->layer_param_.data_param().batch_size();
    const int num_people_image = this->layer_param_.op_transform_param().num_people_image();

    // Multi Image shape (Data layer is ([frame*batch * 3 * 368 * 38])) - Set Data size
    const int width = this->layer_param_.op_transform_param().crop_size_x();
    const int height = this->layer_param_.op_transform_param().crop_size_y();
    std::vector<int> topShape{batch_size * triplet_size, 3, height, width};
    top[0]->Reshape(topShape);

    // Set output and prefetch size
    this->transformed_data_.Reshape(topShape[0], topShape[1], topShape[2], topShape[3]);
    for (int i = 0; i < this->prefetch_.size(); ++i)
        this->prefetch_[i]->data_.Reshape(topShape);
    LOG(INFO) << "Image shape: " << topShape[0] << ", " << topShape[1] << ", " << topShape[2] << ", " << topShape[3];

    // Labels
    if (this->output_labels_)
    {
        std::vector<int> labelShape{batch_size * triplet_size * num_people_image, 5};
        top[1]->Reshape(labelShape);
        for (int i = 0; i < this->prefetch_.size(); ++i)
            this->prefetch_[i]->label_.Reshape(labelShape);
        for (int i = 0; i < this->prefetch_.size(); ++i)
            for (int j = 0; j < Batch<float>::extra_labels_count; ++j)
                this->prefetch_[i]->extra_labels_[j].Reshape(labelShape);
        this->transformed_label_.Reshape(labelShape);
        LOG(INFO) << "Label shape: " << labelShape[0] << ", " << labelShape[1] << ", ";
    }
    else
        throw std::runtime_error{"output_labels_ must be set to true" + getLine(__LINE__, __FUNCTION__, __FILE__)};

    cout << "\t\t****Data Layer successfully initialized****" << endl;
}

template <typename Dtype>
bool OPTripletLayer<Dtype>::Skip()
{
    int size = Caffe::solver_count();
    int rank = Caffe::solver_rank();
    bool keep = (offset_ % size) == rank ||
                  // In test mode, only rank 0 runs, so avoid skipping
                  this->layer_param_.phase() == TEST;
    return !keep;
}

template<typename Dtype>
void OPTripletLayer<Dtype>::Next()
{
    cursor_->Next();
    if (!cursor_->valid())
    {
        LOG_IF(INFO, Caffe::root_solver())
                << "Restarting data prefetching from start.";
        cursor_->SeekToFirst();
    }
    offset_++;
}

// OpenPose: added
template <typename Dtype>
bool OPTripletLayer<Dtype>::SkipSecond()
{
    int size = Caffe::solver_count();
    int rank = Caffe::solver_rank();
    bool keep = (offsetSecond % size) == rank ||
                  // In test mode, only rank 0 runs, so avoid skipping
                  this->layer_param_.phase() == TEST;
    return !keep;
}

template<typename Dtype>
void OPTripletLayer<Dtype>::NextBackground()
{
    if (backgroundDb)
    {
        cursorBackground->Next();
        if (!cursorBackground->valid())
        {
            LOG_IF(INFO, Caffe::root_solver())
                    << "Restarting negatives data prefetching from start.";
            cursorBackground->SeekToFirst();
        }
    }
}

template<typename Dtype>
void OPTripletLayer<Dtype>::NextSecond()
{
    cursorSecond->Next();
    if (!cursorSecond->valid())
    {
        LOG_IF(INFO, Caffe::root_solver())
                << "Restarting second data prefetching from start.";
        cursorSecond->SeekToFirst();
    }
    offsetSecond++;
}
// OpenPose: added ended

//def area(a, b):  # returns None if rectangles don't intersect
//    dx = min(a.x2, b.x2) - max(a.x1, b.x1)
//    dy = min(a.y2, b.y2) - max(a.y1, b.y1)
//    if (dx>=0) and (dy>=0):
//        return dx*dy

float intersectionPercentage(cv::Rect a, cv::Rect b){
    float dx = min(a.br().x, b.br().x) - max(a.tl().x, b.tl().x);
    float dy = min(a.br().y, b.br().y) - max(a.tl().y, b.tl().y);
    float intersect_area = 0;
    if (dx >= 0 && dy >= 0) intersect_area = dx*dy;
    return max(intersect_area/a.area(), intersect_area/b.area());
}

//def rotate_bound(image, angle):
//    # grab the dimensions of the image and then determine the
//    # center
//    (h, w) = image.shape[:2]
//    (cX, cY) = (w // 2, h // 2)

//    # grab the rotation matrix (applying the negative of the
//    # angle to rotate clockwise), then grab the sine and cosine
//    # (i.e., the rotation components of the matrix)
//    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
//    cos = np.abs(M[0, 0])
//    sin = np.abs(M[0, 1])

//    # compute the new bounding dimensions of the image
//    nW = int((h * sin) + (w * cos))
//    nH = int((h * cos) + (w * sin))

//    # adjust the rotation matrix to take into account translation
//    M[0, 2] += (nW / 2) - cX
//    M[1, 2] += (nH / 2) - cY

//    # perform the actual rotation and return the image
//    return cv2.warpAffine(image, M, (nW, nH))

cv::Size rotateBoundSize(cv::Size currSize, float angle){
    int h = currSize.height;
    int w = currSize.width;
    int cx = w/2;
    int cy = h/2;

    cv::Mat M = cv::getRotationMatrix2D(cv::Point(cx,cy), -angle, 1.0);
    float cos = M.at<double>(0,0);
    float sin = M.at<double>(1,0);
    int nW = int((h * sin) + (w * cos));
    int nH = int((h * cos) + (w * sin));

    M.at<double>(0,2) += (nW / 2) - cx;
    M.at<double>(1,2) += (nH / 2) - cy;

    return cv::Size(nW, nH);
}

cv::Mat rotateBound(const cv::Mat& image, float angle){
    int h = image.size().height;
    int w = image.size().width;
    int cx = w/2;
    int cy = h/2;

    cv::Mat M = cv::getRotationMatrix2D(cv::Point(cx,cy), -angle, 1.0);
    float cos = M.at<double>(0,0);
    float sin = M.at<double>(1,0);
    int nW = int((h * sin) + (w * cos));
    int nH = int((h * cos) + (w * sin));

    M.at<double>(0,2) += (nW / 2) - cx;
    M.at<double>(1,2) += (nH / 2) - cy;

    cv::Mat finalImage;
    cv::warpAffine(image, finalImage, M, cv::Size(nW, nH), cv::INTER_CUBIC, // CUBIC to consider rotations
                   cv::BORDER_CONSTANT, cv::Scalar{0,0,0});
    return finalImage;
}

void generateImage(const cv::Mat& backgroundImage, const std::vector<cv::Mat>& personImages, cv::Mat& finalImage, std::vector<cv::Rect>& rectangles){
/*
 * ? A mechanism to crop image, flip, scale rotate
 *
 * I take the images I have, 1st image, randomly sample some start index and scale
 * 2nd one do the same, keep sampling until no overlap
 * 3rd one do the same for both
 */
    float size_scale = 0.7;
    float intersect_ratio = 0.2;
    float image_size_ratio = 1.1;
    float rotate_angle = 10;

    std::vector<cv::Rect> hold_rectangles;
    rectangles.clear();
    finalImage = backgroundImage.clone();
    for(int i=0; i<personImages.size(); i++){
        const cv::Mat& personImage = personImages[i];

        // Do the crop or rotation here!!


        int counter = 0;
        int x, y, w, h;
        float rot;
        while(1){
            // NEED A BETTER WAY TO HANDLE SCALE
            if(counter > 500){
                size_scale = 0.5;
                std::cout << "warning: reducing scale" << std::endl;
            }else if(counter > 1000){
                size_scale = 0.4;
                std::cout << "warning: reducing scale" << std::endl;
            }

            counter++;
            w = getRand(personImage.size().width*size_scale, personImage.size().width*(1./size_scale));
            h = w*(personImage.size().height/personImage.size().width);
            x = getRand(0, fabs(finalImage.size().width - w));
            y = getRand(0, fabs(finalImage.size().height - h));
            cv::Rect hold_rect(x,y,w,h);

            // Rot
            rot = getRand(-rotate_angle,rotate_angle);
            cv::Size newPossibleSize = rotateBoundSize(cv::Size(w,h), rot);
            x += (newPossibleSize.width - w) / 2;
            y += (newPossibleSize.height - h) / 2;
            w = newPossibleSize.width;
            h = newPossibleSize.height;

            if(w >= finalImage.size().width/image_size_ratio || h >= finalImage.size().height/image_size_ratio) continue;
            if(x < 0 || x >= finalImage.size().width || y < 0 || y >= finalImage.size().height) continue;
            if(x+w < 0 || x+w >= finalImage.size().width || y+h < 0 || y+h >= finalImage.size().height) continue;
            cv::Rect currentRect = cv::Rect(x,y,w,h);

            bool intersectFail = false;
            for(cv::Rect& otherRect : rectangles){
                if(intersectionPercentage(currentRect, otherRect) > intersect_ratio){
                    intersectFail = true;
                    break;
                }
            }
            if(intersectFail) continue;

            rectangles.emplace_back(currentRect);
            hold_rectangles.emplace_back(hold_rect);
            break;
        }

        cv::Mat newPersonImage;
        cv::Rect rect = rectangles.back();
        cv::Rect hold_rect = hold_rectangles.back();
        cv::resize(personImage, newPersonImage,cv::Size(hold_rect.width, hold_rect.height));

        cv::rectangle(finalImage, rect, cv::Scalar(255,0,0));

        cv::Mat mask = cv::Mat(newPersonImage.size(), CV_8UC3,cv::Scalar(255,255,255));
        mask = rotateBound(mask, rot);
        newPersonImage = rotateBound(newPersonImage, rot);

        //cv::Mat mask = newPersonImage.clone();
        newPersonImage.copyTo(finalImage(rect), mask);
    }

    cv::imshow("win", finalImage);
    cv::waitKey(1000);
}

// This function is called on prefetch thread
template<typename Dtype>
void OPTripletLayer<Dtype>::load_batch(Batch<Dtype>* batch)
{
//    CPUTimer batch_timer;
//    batch_timer.Start();
//    double read_time = 0;
//    double trans_time = 0;
//    CPUTimer timer;
//    CHECK(batch->data_.count());
//    CHECK(this->transformed_data_.count());
    const int batch_size = this->layer_param_.data_param().batch_size();
    const int total_images = batch_size * triplet_size;
    const int num_people_image = this->layer_param_.op_transform_param().num_people_image();

    // Get Label pointer [Label shape: 20, 132, 46, 46]
    auto* topLabel = batch->label_.mutable_cpu_data();
    for(int i=0; i<Batch<float>::extra_labels_count; i++)
        batch->extra_labels_[i].mutable_cpu_data();

    auto* topData = batch->data_.mutable_cpu_data();
    auto* labelData = batch->label_.mutable_cpu_data();

    std::cout << batch->data_.shape_string() << std::endl; // 9, 3, 368, 368
    std::cout << batch->label_.shape_string() << std::endl; // 27, 5

    /*
     * 0. Store Path of Train Folder
     *      1. Save all the background paths into some dict[pid] = array(cv::Mat)
     * 1. Load 9 Backgrounds into CVMat with the LMDB
     * 2. Load 3(Batchsize * NumPeople) completely unique people / ids
     *      For each unique person, load one positive and load one negative (Comes to 9*3 = 27)
     *      [1 2 3 4 5 6 7 8 9] Ref
     *      [1 2 3 4 5 6 7 8 9] +
     *      [1 2 3 4 5 6 7 8 9] -
     * 3. Loop through 3 then 3 background
     *      For each set, spread 3 people equally on every image, diff size/rotation etc.
     *      Store the bounding box
     * 4. Setup the table
     */

    // Load background images
    std::vector<cv::Mat> backgroundImages;
    Datum datumBackground;
    for (int item_id = 0; item_id < total_images; ++item_id) {
        if (backgroundDb)
        {
            NextBackground();
            datumBackground.ParseFromString(cursorBackground->value());
            backgroundImages.emplace_back(mOPDataTransformer->parseBackground(&datumBackground));
        }
    }

    // Load Unique People IDS
    for(int i=0; i<batch_size; i++){

        std::vector< std::pair<int, std::vector<std::string>>> positive_ids, negative_ids; // 3 each
        for(int j=0; j<num_people_image; j++){
            positive_ids.emplace_back(*select_randomly(reidData.begin(), reidData.end()));
            negative_ids.emplace_back(*select_randomly(reidData.begin(), reidData.end()));
        }

        for(int j=0; j<triplet_size; j++){
            int image_id = i*triplet_size + j;
            cv::Mat backgroundImage = backgroundImages[image_id];
            std::vector<cv::Mat> personImages;

            // J=0 Is for Reference Image
            if(j==0){
                for(auto& pos_id : positive_ids){
                    cv::Mat pos_id_image = cv::imread(pos_id.second[getRand(0, pos_id.second.size()-1)]);
                    personImages.emplace_back(pos_id_image);
                }
            }
            else if(j==1){
                for(auto& pos_id : positive_ids){
                    cv::Mat pos_id_image = cv::imread(pos_id.second[getRand(0, pos_id.second.size()-1)]);
                    personImages.emplace_back(pos_id_image);
                }
            }
            else if(j==2){
                for(auto& neg_id : negative_ids){
                    cv::Mat neg_id_image = cv::imread(neg_id.second[getRand(0, neg_id.second.size()-1)]);
                    personImages.emplace_back(neg_id_image);
                }
            }

            cv::Mat finalImage; std::vector<cv::Rect> rects;
            generateImage(backgroundImage, personImages, finalImage, rects);
//            int xx = 0;
//            for(cv::Mat& personImage : personImages){
//                cv::imwrite("visualize/"+std::to_string(image_id)+"_"+std::to_string(xx)+".jpg", personImage);
//                xx++;
//            }
            std::cout << image_id << std::endl;

        }

    }
    exit(-1);

    std::cout << "---" << std::endl;



//    // Testing Optional
////    if(vCounter == 2){
////    auto oPDataTransformerPtr = this->mOPDataTransformer;
////    oPDataTransformerPtr->Test(frame_size, &(this->transformed_data_), &(this->transformed_label_));
////    }
//    //boost::this_thread::sleep_for(boost::chrono::milliseconds(1000));
//    //std::cout << "Loaded Data" << std::endl;

//    // Timer (every 20 iterations x batch size)
//    mCounter++;
//    vCounter++;
//    const auto repeatEveryXVisualizations = 2;
//    if (mCounter == 20*repeatEveryXVisualizations)
//    {
//        std::cout << "Time: " << mDuration/repeatEveryXVisualizations * 1e-9 << "s\t"
//                  << "Ratio: " << mOnes/float(mOnes+mTwos) << std::endl;
//        mDuration = 0;
//        mCounter = 0;
//    }
//    timer.Stop();
//    batch_timer.Stop();
//    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
//    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
//    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(OPTripletLayer);
REGISTER_LAYER_CLASS(OPTriplet);

}  // namespace caffe
