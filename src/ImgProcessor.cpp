#include "ImgProcessor.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

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

    dst = tmp.clone();
    for (int i = 0; i < tmp.rows; ++i) {
        for (int j = 0; j < tmp.cols; ++j) {
            // y = c * ln(1 + x)
            dst.at<uchar>(i, j) = cv::saturate_cast<uchar>(c * log(1.0 + tmp.at<uchar>(i, j)));
        }
    }

    // 图像归一化
    cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
}

void ImgProcessor::GrayGammaTrans(const cv::Mat &src, cv::Mat &dst, double c /*= 1.0*/, double gamma /*= 1.0*/) {
    cv::Mat tmp;
    if (src.channels() == 3) {
        cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 1) {
        tmp = src;
    }

    dst = tmp.clone();
    for (int i = 0; i < tmp.rows; ++i) {
        for (int j = 0; j < tmp.cols; ++j) {
            // y = c * x ^ gamma
            dst.at<uchar>(i, j) = cv::saturate_cast<uchar>(c * pow(tmp.at<uchar>(i, j), gamma));
        }
    }

    // 图像归一化
    cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
}

void ImgProcessor::GetHistogram(cv::Mat const &src, cv::Mat &dst) {
    cv::Mat tmp;
    if (src.channels() == 3) {
        cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 1) {
        tmp = src;
    }

    int hSize = 256;
    float ranges[] = {0, 256};
    float const *hRanges = {ranges};

    // 计算直方图
    cv::Mat hist;
    cv::calcHist(&tmp, 1, 0, cv::Mat(), hist, 1, &hSize, &hRanges, true, false);

    // 确定直方图长宽
    int histH = 300, histW = 512;
    int biW = histW / hSize;
    dst = cv::Mat::ones(histH, histW, CV_8UC1);

    // 直方图输出值归一化到0~255
    cv::normalize(hist, hist, 0, histH, cv::NORM_MINMAX, CV_8UC1, cv::Mat());

    // 绘制直方图
    for (int i = 1; i < hSize; ++i) {
        cv::line(
            dst,
            cv::Point(biW * (i - 1), histH - hist.at<uchar>(i - 1)),
            cv::Point(biW * i, histH - hist.at<uchar>(i)),
            cv::Scalar(255));
    }
}

void ImgProcessor::HistEqualization(const cv::Mat &src, cv::Mat &dst) {
    cv::Mat tmp;
    if (src.channels() == 3) {
        cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 1) {
        tmp = src;
    }

    // 直方图均衡化
    cv::equalizeHist(tmp, dst);
}