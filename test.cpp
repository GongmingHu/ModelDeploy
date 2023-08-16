#include <iostream>
#include "resnet18.h"
#include "resnet18ort.h"
#include "utils.h"
#include "segmentation.h"

using namespace std;

int main(){

    // ResNet18 resnet18("../assets/samoye.jpg", "../assets/imagenet1k.txt", "../assets/resnet18.onnx");
    // util::ClassificationResults result;
    // resnet18.classify(result, 3);
    // for(int i = 0; i < result.label.size(); i++ ){
    //     cout.width(50);  
    //     cout << "predict class is:" << result.label[i] << "score:" << result.score[i] << endl;
    //     //cout << "score:" << result.score[i] << endl;
    // }

    // ResNet18Ort resnet18ort("../assets/samoye.jpg", "../assets/imagenet1k.txt", "../assets/resnet18.onnx");
    // util::ClassificationResults result1;
    // resnet18ort.classify(result1, 3);
    // for(int i = 0; i < result1.label.size(); i++ ){  
    //     cout.width(50);   
    //     cout << "predict class is:" << result1.label[i] << "score:" << result1.score[i] << endl;
    // }

    Segmentation model("../assets/COCO_test2014_000000000016.jpg",
    "../assets/lraspp_mobilenet_v3_large.onnx");

    cv::Mat output;
    model.segment(output);
    cv::imwrite("output.jpg", output);
    std::cout << "ok" << std::endl;
    return 0;
}