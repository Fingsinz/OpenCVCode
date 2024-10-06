#include "ImgProcessor.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

void ImgProcessor::GrayInversion(cv::Mat const &src, cv::Mat &dst) {
    cv::Mat tmp;
    if (src.channels() == 3) {
        cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 1) {
        tmp = src;
    }

    dst = cv::Mat::zeros(tmp.size(), CV_8UC1);
    for (int i = 0; i < tmp.rows; ++i) {
        for (int j = 0; j < tmp.cols; ++j) {
            dst.at<uchar>(i, j) = 255 - tmp.at<uchar>(i, j);
        }
    }
}

void ImgProcessor::GrayLogTrans(const cv::Mat &src, cv::Mat &dst, double c /*= 1.0*/) {
    cv::Mat tmp;
    if (src.channels() == 3) {
        cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 1) {
        tmp = src;
    }

    dst = cv::Mat::zeros(tmp.size(), CV_64FC1);
    for (int i = 0; i < tmp.rows; ++i) {
        for (int j = 0; j < tmp.cols; ++j) {
            // y = c * ln(1 + x)
            dst.at<double>(i, j) = c * std::log(1.0 + tmp.at<uchar>(i, j));
        }
    }

    // 将像素从double类型归一化到灰度图
    cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
}