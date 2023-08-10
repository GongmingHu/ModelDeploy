#include <iostream>
#include <opencv2/opencv.hpp>
#include "resnet18.h"

using namespace std;
using namespace cv;
#include "utils.h"
int main(){
    ResNet18 resnet18("../assets/samoye.jpg", "../assets/imagenet1k.txt", "../assets/resnet18.onnx");
    ResNet18::ClassificationResults result;
    resnet18.classify(result, 5);
    cout << endl;
    for(int i = 0; i < result.label.size(); i++ ){   
        cout << "predict class is:" << result.label[i] << endl;
        cout << "score:" << result.score[i] << endl;
    }

    return 0;
}