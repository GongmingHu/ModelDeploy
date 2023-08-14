#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include "utils.h"

using namespace std;

class ResNet18Ort
{
private:
    cv::Mat _src;
    string _imagePath;
    string _classPath;
    string _modelPath;
    vector<string> _classNames;

    Ort::Env _env;
    Ort::SessionOptions _session_options;
    unique_ptr<Ort::Session> _session;

    vector<int64_t> _inputShape, _outputShape;
    Ort::AllocatorWithDefaultOptions _allocator;

    Ort::MemoryInfo _memory_info_handler = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

    vector<string> inputNames;
    vector<string> outputNames;
    
    void InitOrtEnv();

public:
    explicit ResNet18Ort(const string & imagePath, const string & classPath,const string & modelPath);
    ~ResNet18Ort();
    void classify(util::ClassificationResults & result, int topk = 1);
};

