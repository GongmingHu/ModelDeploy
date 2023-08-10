#include "resnet18ort.h"

void ResNet18Ort::InitOrtEnv()
{
}

ResNet18Ort::ResNet18Ort(const string & imagePath, const string & classPath,const string & modelPath)
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

void ResNet18Ort::classify()
{

}
