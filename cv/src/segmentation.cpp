#include "segmentation.h"

Segmentation::Segmentation(const std::string &imagePath, const std::string &modelPath)
{
    _imagePath = imagePath;
    _modelPath = modelPath;
    //CV_Assert(!_modelPath.empty());
    _net = cv::dnn::readNet(_modelPath);
    _net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    _net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cout << "Inference device: CPU" << std::endl;
}

Segmentation::~Segmentation()
{
}

void Segmentation::segment(cv::Mat & output)
{
    cv::Mat blob;
    //std::vector<cv::Mat> output;
    //cv::Mat output;
    transform(blob);
    process(blob, output);
    cv::Mat segm;
    colorizeSegmentation(output, segm);
    cv::resize(segm, segm, _src.size(), 0, 0, cv::INTER_NEAREST);

    cv::addWeighted(_src, 0.4, segm, 0.6, 0, output);
    cv::imshow("Output", output);
    cv::waitKey(0);
}

void Segmentation::transform(cv::Mat &blob)
{
    _src = cv::imread(_imagePath);
    cv::Mat image = _src.clone();
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    resize(image, image, cv::Size(512, 512));
    image.convertTo(image,  CV_32FC3, 1.f / 255.f, 0.f);
    cv::subtract(image, cv::Scalar(0.406, 0.456, 0.485), image);
    cv::divide(image, cv::Scalar(0.225, 0.224, 0.229), image);
    cv::dnn::blobFromImage(image, blob, 1, cv::Size(image.cols, image.rows), cv::Scalar(), true, false);
}

void Segmentation::process(cv::Mat &blob, cv::Mat &outputs)
{
    _net.setInput(blob);
    std::cout << _net.getUnconnectedOutLayersNames().size() <<std::endl;
    std::cout << _net.getUnconnectedOutLayersNames()[0] <<std::endl;
    _net.forward(outputs);
}

void Segmentation::colorizeSegmentation(const cv::Mat &score, cv::Mat &segm)
{
    const int rows = score.size[2];
    const int cols = score.size[3];
    const int chns = score.size[1];

    if (_colors.empty())
    {
        // Generate colors.
        _colors.push_back(cv::Vec3b());
        for (int i = 1; i < chns; ++i)
        {
            cv::Vec3b color;
            for (int j = 0; j < 3; ++j)
                color[j] = (_colors[i - 1][j] + rand() % 256) / 2;
            _colors.push_back(color);
        }
    }
    else if (chns != (int)_colors.size())
    {
        CV_Error(cv::Error::StsError, cv::format("Number of output classes does not match "
                                         "number of colors (%d != %zu)", chns, _colors.size()));
    }

    cv::Mat maxCl = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat maxVal(rows, cols, CV_32FC1, score.data);
    for (int ch = 1; ch < chns; ch++)
    {
        for (int row = 0; row < rows; row++)
        {
            const float *ptrScore = score.ptr<float>(0, ch, row);
            uint8_t *ptrMaxCl = maxCl.ptr<uint8_t>(row);
            float *ptrMaxVal = maxVal.ptr<float>(row);
            for (int col = 0; col < cols; col++)
            {
                if (ptrScore[col] > ptrMaxVal[col])
                {
                    ptrMaxVal[col] = ptrScore[col];
                    ptrMaxCl[col] = (uchar)ch;
                }
            }
        }
    }

    segm.create(rows, cols, CV_8UC3);
    for (int row = 0; row < rows; row++)
    {
        const uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
        cv::Vec3b *ptrSegm = segm.ptr<cv::Vec3b>(row);
        for (int col = 0; col < cols; col++)
        {
            ptrSegm[col] = _colors[ptrMaxCl[col]];
        }
    }
}
