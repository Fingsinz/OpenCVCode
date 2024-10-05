#include "ImgProcessor.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"

int main(int argc, char const *argv[]) {
    cv::Mat img = cv::imread("../../res/lena.jpg");

    ImgProcessor::GrayInversion(img, img);
    
    cv::imshow("img", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}