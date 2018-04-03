// File based in `data_layer.hpp`, extracted from Caffe GitHub on Sep 7th, 2017
// https://github.com/BVLC/caffe/

#ifndef CAFFE_OPENPOSE_OP_VIDEO_LAYER_HPP
#define CAFFE_OPENPOSE_OP_VIDEO_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
// OpenPose: added
#include "caffe/openpose/oPDataTransformer.hpp"
#include <boost/thread/thread.hpp>
// OpenPose: added end

namespace caffe {

template <typename Dtype>
class OPVideoLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit OPVideoLayer(const LayerParameter& param);
  virtual ~OPVideoLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "OPVideo"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  void Next();
  bool Skip();
  virtual void load_batch(Batch<Dtype>* batch);

  shared_ptr<db::DB> db_;
  shared_ptr<db::Cursor> cursor_;
  uint64_t offset_;

  // OpenPose: added
  bool SkipSecond();
  void NextBackground();
  void NextSecond();
  // Secondary lmdb
  uint64_t offsetSecond;
  bool secondDb;
  float secondProbability;
  shared_ptr<db::DB> dbSecond;
  shared_ptr<db::Cursor> cursorSecond;
  shared_ptr<OPDataTransformer<Dtype> > mOPDataTransformerSecondary;
  // Background lmdb
  bool backgroundDb;
  shared_ptr<db::DB> dbBackground;
  shared_ptr<db::Cursor> cursorBackground;
  // New label
  Blob<Dtype> transformed_label_;
  // Data augmentation parameters
  OPTransformationParameter op_transform_param_;
  // Data augmentation class
  shared_ptr<OPDataTransformer<Dtype> > mOPDataTransformer;
  // Timer
  unsigned long long mOnes;
  unsigned long long mTwos;
  int mCounter;
  double mDuration;

  const int frame_size = 6;
  // OpenPose: added end
};

}  // namespace caffe

#endif  // CAFFE_OPENPOSE_OP_DATA_LAYER_HPP
