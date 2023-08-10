#include "resnet18.h"
#include "utils.h"

ResNet18::ResNet18(const string &imagePath, const string &classPath, const string &modelPath)
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
    _net = cv::dnn::readNet(_modelPath);
    _net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    _net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    _src = cv::imread(_imagePath);
}

void ResNet18::transform(cv::Mat &outputImage)
{
    cv::Mat image = _src.clone();
    image.convertTo(image, CV_32FC3, 1. / 255.);
    cv::subtract(image, cv::Scalar(0.406, 0.456, 0.485), image);
    cv::divide(image, cv::Scalar(0.225, 0.224, 0.229), image);
    cv::dnn::blobFromImage(image, outputImage, 1, cv::Size(224, 224),
                           cv::Scalar(), true, false);
}

void ResNet18::classify(ClassificationResults & result, int topk)
{
    cv::Mat blob;
    transform(blob);
    _net.setInput(blob);
    cv::Mat prob = _net.forward();
    
    
    // cv::Point classIdPoint;
    // double confidence;
    // float maxProb = 0.0;
    // float sum = 0.0;
    // cv::Mat softmaxProb;
    
    // maxProb = *std::max_element(prob.begin<float>(), prob.end<float>());
    // cv::exp(prob-maxProb, softmaxProb);
    // sum = (float)cv::sum(softmaxProb)[0];
    // softmaxProb /= sum;

    // minMaxLoc(softmaxProb.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    // cout << "predict class: " << _classNames[classIdPoint.x] << endl;
    // cout << "confidence after softmax: " << confidence << endl;

    const float * logits = (float *)prob.data;
    unsigned int maxId;
    vector<float> scores = softmax(logits, _classNames.size(), maxId);
    vector<unsigned int> indices = argsort(scores);

    cout << "predict class: " << _classNames[maxId] << endl;
    //cout << "predict class: " << _classNames[indices[0]] << endl;
    cout << "confidence after softmax: " << scores[indices[0]] << endl;
    result.score.clear();
    result.label.clear();
    for(int i = 0; i < topk; i++){
        result.label.push_back(_classNames[indices[i]]);
        result.score.push_back(scores[indices[i]]);      
    }
}

ResNet18::~ResNet18()
{
}
