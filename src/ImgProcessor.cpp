#include "ImgProcessor.hpp"
#include "opencv2/imgproc.hpp"

void ImgProcessor::GrayInversion(cv::Mat const &src, cv::Mat &dst) {
    if (src.channels() == 3) {
        cv::Mat tmp;
        cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);

        for (int i = 0; i < tmp.rows; ++i) {
            for (int j = 0; j < tmp.cols; ++j) {
                tmp.at<uchar>(i, j) = 255 - tmp.at<uchar>(i, j);
            }
        }
        dst = tmp;
    }

    if (src.channels() == 1) {
        for (int i = 0; i < src.rows; ++i) {
            for (int j = 0; j < src.cols; ++j) {
                dst.at<uchar>(i, j) = 255 - src.at<uchar>(i, j);
            }
        }
    }
}