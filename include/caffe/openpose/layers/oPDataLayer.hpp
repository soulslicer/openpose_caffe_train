// File based in `data_layer.hpp`, extracted from Caffe GitHub on Sep 7th, 2017
// https://github.com/BVLC/caffe/

#ifndef CAFFE_OPENPOSE_OP_DATA_LAYER_HPP
#define CAFFE_OPENPOSE_OP_DATA_LAYER_HPP

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
// OpenPose: added end

namespace caffe {

template <typename Dtype>
class OPDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit OPDataLayer(const LayerParameter& param);
  virtual ~OPDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "OPData"; }
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
  Blob<Dtype> transformed_label_;
  OPTransformationParameter op_transform_param_;
  shared_ptr<OPDataTransformer<Dtype> > mOPDataTransformer;
  // OpenPose: added end
};

}  // namespace caffe

#endif  // CAFFE_OPENPOSE_OP_DATA_LAYER_HPP
