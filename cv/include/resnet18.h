#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;

class ResNet18
{
private:
    /* data */
    cv::Mat _src;
    string _imagePath;
    string _classPath;
    string _modelPath;
    vector<string> _classNames;
    cv::dnn::Net _net;

public:
    struct ClassificationResults{
        vector<float> score;
        vector<std::string>label;
    };
    ResNet18(const string & imagePath, const string & classPath, const string & modelPath);
    void transform(cv::Mat & outputImage);
    void classify(ClassificationResults & result, int topk = 1);
    ~ResNet18();
};

