#include "resnet18ort.h"

void ResNet18Ort::InitOrtEnv()
{
    _env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Classification");
    _session_options = Ort::SessionOptions();
    _session_options.SetInterOpNumThreads(4);
    _session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
    const wchar_t *model_path = multi_Byte_To_Wide_Char(_modelPath);
#else
    const char *model_path = _modelPath.c_str();
#endif

    _session = std::make_unique<Ort::Session>(_env, model_path, _session_options);

    size_t inputCount = _session->GetInputCount();
    size_t outputCount = _session->GetOutputCount();
    inputNames.reserve(inputCount);
    outputNames.reserve(outputCount);
    cout << "Number of Input Nodes: " << inputCount << endl;
    cout << "Number of Output Nodes: " << outputCount << endl;

    _inputShape = _session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();   //[1, 3, 224, 224]
    _outputShape = _session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape(); //[1, 1000]

    // inputNames.push_back(_session->GetInputName(0, _allocator));  //onnxruntime 1.13 GetInputName deprecated
    // outputNames.push_back(_session->GetOutputName(0, _allocator));

    auto inputName = _session->GetInputNameAllocated(0, _allocator);
    inputNames.push_back(inputName.get());
    auto outputName = _session->GetOutputNameAllocated(0, _allocator);
    outputNames.push_back(outputName.get());

    // delete []model_path;
}

ResNet18Ort::ResNet18Ort(const string &imagePath, const string &classPath, const string &modelPath)
{
    _imagePath = imagePath;
    _classPath = classPath;
    _modelPath = modelPath;
    ifstream ifs(_classPath.c_str());
    if (!ifs.is_open())
    {
        CV_Error(cv::Error::StsError, "File " + _classPath + " not found");
    }
    string line;
    while (getline(ifs, line))
    {
        _classNames.push_back(line);
    }
    cout << "read label over!" << endl;
    InitOrtEnv();
}

ResNet18Ort::~ResNet18Ort()
{
}

void ResNet18Ort::classify(util::ClassificationResults & result, int topk)
{
    _src = cv::imread(_imagePath);
    cv::Mat processImage = _src.clone();
    // 预处理
    cv::resize(processImage, processImage,
               cv::Size(_inputShape.at(3), _inputShape.at(2)),
               cv::InterpolationFlags::INTER_CUBIC);

    cv::cvtColor(processImage, processImage,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);
    processImage.convertTo(processImage, CV_32F, 1.0 / 255);

    cv::Mat channels[3];
    cv::split(processImage, channels);

    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    cv::merge(channels, 3, processImage);
    cv::dnn::blobFromImage(processImage, processImage);

    // size_t inputTensorSize = vectorProduct(_inputShape);        // 1*3*640*640
    size_t inputTensorSize = _inputShape[0] * _inputShape[1] * _inputShape[2] * _inputShape[3];
    std::vector<float> inputTensorValue(inputTensorSize);
    inputTensorValue.assign(processImage.begin<float>(),
                            processImage.end<float>());

    // size_t outputTensorSize = vectorProduct(_outputShape);      // 1*1000
    size_t outputTensorSize = _outputShape[0] * _outputShape[1];
    assert(("Output tensor size should equal to the label set size.",
            _classNames.size() == outputTensorSize));
    std::vector<float> outputTensorValues(outputTensorSize);

    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        _memory_info_handler, inputTensorValue.data(), inputTensorSize, _inputShape.data(),
        _inputShape.size()));

    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        _memory_info_handler, outputTensorValues.data(), outputTensorSize,
        _outputShape.data(), _outputShape.size()));

    // const char *inputNamesPre[] = {"input"}, *outputNamesPre[] = {"output"};

    const std::array<const char *, 1> inputNamesPre = {inputNames[0].c_str()};
    const std::array<const char *, 1> outputNamesPre = {outputNames[0].c_str()};

    _session->Run(Ort::RunOptions{nullptr},
                  inputNamesPre.data(),
                  inputTensors.data(),
                  1,
                  outputNamesPre.data(),
                  outputTensors.data(),
                  1);
    // the result restore in vector<float> outputTensorValues;
    unsigned int maxId;
    vector<float> scores = util::softmax(outputTensorValues, maxId);
    vector<unsigned int> indices = util::argsort(scores);
    
    //cout << "predict class: " << _classNames[maxId] << endl;
    cout << "predict class: " << _classNames[indices[0]] << endl;
    cout << "confidence after softmax: " << scores[indices[0]] << endl;
    result.score.clear();
    result.label.clear();
    for(int i = 0; i < topk; i++){
        result.label.push_back(_classNames[indices[i]]);
        result.score.push_back(scores[indices[i]]);      
    }

}
