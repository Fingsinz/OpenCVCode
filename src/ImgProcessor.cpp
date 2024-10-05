#include "ImgProcessor.hpp"
#include "opencv2/imgproc.hpp"

void ImgProcessor::GrayInversion(cv::Mat const &src, cv::Mat &dst) {
    if(src.channels() == 3) {
        cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
    } 
    
    if(src.channels() == 1) {
        for(int i = 0; i < src.rows; ++i) {
            for(int j = 0; j < src.cols; ++j) {
                dst.at<uchar>(i, j) = 255 - src.at<uchar>(i, j);
            }
        }
    }
}