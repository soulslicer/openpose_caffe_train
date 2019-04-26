#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <caffe/openpose/layers/oPDataLayer.hpp>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
namespace py = pybind11;

class OPCaffe{
public:
    std::shared_ptr<OPDataLayer<float>> dataLayer;
    std::vector<Blob<float>*> bottom, top;

    OPCaffe(py::dict d){
        LayerParameter param;
        param.mutable_data_param()->set_batch_size(py::int_(d["batch_size"]));
        param.mutable_data_param()->set_backend(DataParameter::LMDB);
        param.mutable_op_transform_param()->set_stride(py::int_(d["stride"]));
        param.mutable_op_transform_param()->set_max_degree_rotations(py::str(d["max_degree_rotations"]));
        param.mutable_op_transform_param()->set_crop_size_x(py::int_(d["crop_size_x"]));
        param.mutable_op_transform_param()->set_crop_size_y(py::int_(d["crop_size_y"]));
        param.mutable_op_transform_param()->set_center_perterb_max(py::float_(d["center_perterb_max"]));
        param.mutable_op_transform_param()->set_center_swap_prob(py::float_(d["center_swap_prob"]));
        param.mutable_op_transform_param()->set_scale_prob(py::float_(d["scale_prob"]));
        param.mutable_op_transform_param()->set_scale_mins(py::str(d["scale_mins"]));
        param.mutable_op_transform_param()->set_scale_maxs(py::str(d["scale_maxs"]));
        param.mutable_op_transform_param()->set_target_dist(py::float_(d["target_dist"]));
        param.mutable_op_transform_param()->set_number_max_occlusions(py::str(d["number_max_occlusions"]));
        param.mutable_op_transform_param()->set_sigmas(py::str(d["sigmas"]));
        param.mutable_op_transform_param()->set_models(py::str(d["models"]));
        param.mutable_op_transform_param()->set_sources(py::str(d["sources"]));
        param.mutable_op_transform_param()->set_probabilities(py::str(d["probabilities"]));
        param.mutable_op_transform_param()->set_source_background(py::str(d["source_background"]));
        param.mutable_op_transform_param()->set_normalization(py::int_(d["normalization"]));
        param.mutable_op_transform_param()->set_add_distance(py::int_(d["add_distance"]));
        int size = -1; int rank = -1;
        if(d.contains("msize") && d.contains("mrank")){
            size = py::int_(d["msize"]);
            rank = py::int_(d["mrank"]);
        }
        dataLayer = std::shared_ptr<OPDataLayer<float>>(new OPDataLayer<float>(param, size, rank));

        bottom = {new Blob<float>{1,1,1,1}};
        top = {new Blob<float>{1,1,1,1}, new Blob<float>{1,1,1,1}};
        dataLayer->DataLayerSetUp(bottom, top);

        std::cout << "Initialized" << std::endl;
    }

    void load(Batch<float>& batch){
        batch.data_.Reshape(top[0]->shape());
        batch.label_.Reshape(top[1]->shape());
        dataLayer->load_batch(&batch);
    }
};

class OPTransformer{
public:
    std::shared_ptr<OPDataTransformer<float>> dataTransformer;
    std::vector<Blob<float>*> bottom, top;

    uint64_t offsetBackground;
    boost::shared_ptr<db::DB> dbBackground;
    boost::shared_ptr<db::Cursor> cursorBackground;

    void NextBackground()
    {
        cursorBackground->Next();
        if (!cursorBackground->valid())
        {
            LOG_IF(INFO, Caffe::root_solver())
                    << "Restarting negatives data prefetching from start.";
            cursorBackground->SeekToFirst();
        }
        offsetBackground++;
    }

    OPTransformer(py::dict d){
        LayerParameter param;
        param.mutable_op_transform_param()->set_stride(py::int_(d["stride"]));
        param.mutable_op_transform_param()->set_max_degree_rotations(py::str(d["max_degree_rotations"]));
        param.mutable_op_transform_param()->set_crop_size_x(py::int_(d["crop_size_x"]));
        param.mutable_op_transform_param()->set_crop_size_y(py::int_(d["crop_size_y"]));
        param.mutable_op_transform_param()->set_center_perterb_max(py::float_(d["center_perterb_max"]));
        param.mutable_op_transform_param()->set_center_swap_prob(py::float_(d["center_swap_prob"]));
        param.mutable_op_transform_param()->set_scale_prob(py::float_(d["scale_prob"]));
        param.mutable_op_transform_param()->set_scale_mins(py::str(d["scale_mins"]));
        param.mutable_op_transform_param()->set_scale_maxs(py::str(d["scale_maxs"]));
        param.mutable_op_transform_param()->set_target_dist(py::float_(d["target_dist"]));
        param.mutable_op_transform_param()->set_number_max_occlusions(py::str(d["number_max_occlusions"]));
        param.mutable_op_transform_param()->set_sigmas(py::str(d["sigmas"]));
        param.mutable_op_transform_param()->set_source_background(py::str(d["source_background"]));

        OPTransformationParameter op_transform_param_ = param.op_transform_param();

        dataTransformer.reset(
            new OPDataTransformer<float>(op_transform_param_, Phase::TRAIN, py::str(d["model"])));

        dbBackground.reset(db::GetDB(DataParameter_DB::DataParameter_DB_LMDB));
        dbBackground->Open(op_transform_param_.source_background(), db::READ);
        cursorBackground.reset(dbBackground->NewCursor());

        std::cout << "Initialized" << std::endl;
    }

    void load(cv::Mat& img, MetaData& metaData, Batch<float>& batch){
        // Background
        Datum datumBackground;
        NextBackground();
        datumBackground.ParseFromString(cursorBackground->value());
        cv::Mat backgroundImage;
        const std::string& data = datumBackground.data();
        const int datumNegativeWidth = datumBackground.width();
        const int datumNegativeHeight = datumBackground.height();
        const auto datumNegativeArea = (int)(datumNegativeHeight * datumNegativeWidth);
        const cv::Mat b(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[0]);
        const cv::Mat g(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[datumNegativeArea]);
        const cv::Mat r(datumNegativeHeight, datumNegativeWidth, CV_8UC1, (uchar*)&data[2*datumNegativeArea]);
        std::vector<cv::Mat> bgr = {b,g,r};
        cv::merge(bgr, backgroundImage);

        dataTransformer->TransformPOF(img, backgroundImage, metaData, batch);
    }
};

PYBIND11_MODULE(opcaffe, m){
    py::class_<OPCaffe>(m, "OPCaffe")
        .def(py::init<py::dict>())
        .def("load", &OPCaffe::load)
        ;

    py::class_<MetaData>(m, "MetaData")
        .def(py::init<>())
        .def_readwrite("jointsSelf", &MetaData::jointsSelf)
        .def_readwrite("objPos", &MetaData::objPos)
        .def_readwrite("imageSize", &MetaData::imageSize)
        .def_readwrite("numberOtherPeople", &MetaData::numberOtherPeople)
        .def_readwrite("scaleSelf", &MetaData::scaleSelf)
        ;

    py::class_<OPTransformer>(m, "OPTransformer")
        .def(py::init<py::dict>())
        .def("load", &OPTransformer::load)
        ;

    py::class_<caffe::Batch<float>, std::shared_ptr<caffe::Batch<float>>>(m, "Batch")
        .def(py::init<>())
        .def_readonly("data", &caffe::Batch<float>::data_)
        .def_readonly("label", &caffe::Batch<float>::label_)
        .def_readonly("other", &caffe::Batch<float>::other_)
        ;

    // Point
    py::class_<cv::Size>(m, "Size")
        .def(py::init<>())
        .def(py::init<int, int>())
        .def_readwrite("width", &cv::Size::width)
        .def_readwrite("height", &cv::Size::width)
        ;

    // Point
    py::class_<cv::Point2f>(m, "Point2f")
        .def("__repr__", [](cv::Point2f &a) { return "[" + std::to_string(a.x) + " " + std::to_string(a.y) + "]"; })
        .def(py::init<>())
        .def(py::init<float, float>())
        .def_readwrite("x", &cv::Point2f::x)
        .def_readwrite("y", &cv::Point2f::y)
        ;

    // Point
    py::class_<cv::Point3f>(m, "Point3f")
        .def("__repr__", [](cv::Point3f &a) { return "[" + std::to_string(a.x) + " " + std::to_string(a.y) + " " + std::to_string(a.z) + "]"; })
        .def(py::init<>())
        .def(py::init<float, float, float>())
        .def_readwrite("x", &cv::Point3f::x)
        .def_readwrite("y", &cv::Point3f::y)
        .def_readwrite("z", &cv::Point3f::z)
        ;

    // Point
    py::class_<Joints>(m, "Joints")
        .def(py::init<>())
        .def_readwrite("points", &Joints::points)
        .def_readwrite("points3D", &Joints::points3D)
        .def_readwrite("isVisible", &Joints::isVisible)
        ;

    #ifdef VERSION_INFO
        m.attr("__version__") = VERSION_INFO;
    #else
        m.attr("__version__") = "dev";
    #endif
}

// Numpy - caffe::Blob<float> interop
namespace pybind11 { namespace detail {

template <> struct type_caster<caffe::Blob<float>> {
    public:

        PYBIND11_TYPE_CASTER(caffe::Blob<float>, _("numpy.ndarray"));

        // Cast numpy to op::Array<float>
        bool load(handle src, bool imp)
        {
            throw std::runtime_error("Not implemented");
        }

        // Cast op::Array<float> to numpy
        static handle cast(const caffe::Blob<float> &m, return_value_policy, handle defval)
        {
            std::string format = format_descriptor<float>::format();
            return array(buffer_info(
                m.psuedo_cpu_data(),    /* Pointer to buffer */
                sizeof(float),          /* Size of one scalar */
                format,                 /* Python struct-style format descriptor */
                m.shape().size(),       /* Number of dimensions */
                m.shape(),              /* Buffer dimensions */
                m.stride()              /* Strides (in bytes) for each index */
                )).release();
        }

    };
}} // namespace pybind11::detail

// Numpy - cv::Mat interop
namespace pybind11 { namespace detail {

template <> struct type_caster<cv::Mat> {
    public:

        PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

        // Cast numpy to cv::Mat
        bool load(handle src, bool)
        {
            /* Try a default converting into a Python */
            //array b(src, true);
            array b = reinterpret_borrow<array>(src);
            buffer_info info = b.request();

            int ndims = info.ndim;

            decltype(CV_32F) dtype;
            size_t elemsize;
            if (info.format == format_descriptor<float>::format()) {
                if (ndims == 3) {
                    dtype = CV_32FC3;
                } else {
                    dtype = CV_32FC1;
                }
                elemsize = sizeof(float);
            } else if (info.format == format_descriptor<double>::format()) {
                if (ndims == 3) {
                    dtype = CV_64FC3;
                } else {
                    dtype = CV_64FC1;
                }
                elemsize = sizeof(double);
            } else if (info.format == format_descriptor<unsigned char>::format()) {
                if (ndims == 3) {
                    dtype = CV_8UC3;
                } else {
                    dtype = CV_8UC1;
                }
                elemsize = sizeof(unsigned char);
            } else {
                throw std::logic_error("Unsupported type");
                return false;
            }

            std::vector<int> shape = {(int)info.shape[0], (int)info.shape[1]};

            value = cv::Mat(cv::Size(shape[1], shape[0]), dtype, info.ptr, cv::Mat::AUTO_STEP);
            return true;
        }

        // Cast cv::Mat to numpy
        static handle cast(const cv::Mat &m, return_value_policy, handle defval)
        {
            std::string format = format_descriptor<unsigned char>::format();
            size_t elemsize = sizeof(unsigned char);
            int dim;
            switch(m.type()) {
                case CV_8U:
                    format = format_descriptor<unsigned char>::format();
                    elemsize = sizeof(unsigned char);
                    dim = 2;
                    break;
                case CV_8UC3:
                    format = format_descriptor<unsigned char>::format();
                    elemsize = sizeof(unsigned char);
                    dim = 3;
                    break;
                case CV_32F:
                    format = format_descriptor<float>::format();
                    elemsize = sizeof(float);
                    dim = 2;
                    break;
                case CV_64F:
                    format = format_descriptor<double>::format();
                    elemsize = sizeof(double);
                    dim = 2;
                    break;
                default:
                    throw std::logic_error("Unsupported type");
            }

            std::vector<size_t> bufferdim;
            std::vector<size_t> strides;
            if (dim == 2) {
                bufferdim = {(size_t) m.rows, (size_t) m.cols};
                strides = {elemsize * (size_t) m.cols, elemsize};
            } else if (dim == 3) {
                bufferdim = {(size_t) m.rows, (size_t) m.cols, (size_t) 3};
                strides = {(size_t) elemsize * m.cols * 3, (size_t) elemsize * 3, (size_t) elemsize};
            }
            return array(buffer_info(
                m.data,         /* Pointer to buffer */
                elemsize,       /* Size of one scalar */
                format,         /* Python struct-style format descriptor */
                dim,            /* Number of dimensions */
                bufferdim,      /* Buffer dimensions */
                strides         /* Strides (in bytes) for each index */
                )).release();
        }

    };
}} // namespace pybind11::detail


#endif  // USE_OPENCV
