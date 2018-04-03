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

#include <caffe/openpose/oPDataTransformer.hpp>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using namespace std;

class OPDataTransformerExtended : OPDataTransformer<float>{
public:
    OPDataTransformerExtended(std::string modelString) : OPDataTransformer(modelString) {

    }
};

int main(int argc, char** argv) {
    cout << "a" << endl;

    OPDataTransformerExtended op("PT_21");
    //op.p
}

#endif  // USE_OPENCV
