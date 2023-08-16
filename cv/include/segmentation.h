#ifndef SEGMENTATION_H
#define SEGMENTATION_H
#include <opencv2/opencv.hpp>
#include <iostream>

class Segmentation
{
public:
    explicit Segmentation(const std::string &imagePath,const std::string &modelPath);
    ~Segmentation();

    void segment(cv::Mat & output);

private:
    std::string _imagePath;
    std::string _modelPath;
    cv::dnn::Net _net;
    cv::Mat _src;

    std::vector<std::string> _classNames = {
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"
    };
    void transform(cv::Mat &blob);
    void process(cv::Mat & blob, cv::Mat & outputs);
    void colorizeSegmentation(const cv::Mat &score, cv::Mat &segm);
    std::vector<cv::Vec3b> _colors;
};
#endif