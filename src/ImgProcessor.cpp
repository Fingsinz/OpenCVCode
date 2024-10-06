#include "ImgProcessor.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/matx.hpp"
#include "opencv2/core/saturate.hpp"
#include "opencv2/imgproc.hpp"
#include <random>

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

void ImgProcessor::GrayLogTrans(cv::Mat const &src, cv::Mat &dst, double c /*= 1.0*/) {
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

void ImgProcessor::GrayGammaTrans(cv::Mat const &src, cv::Mat &dst, double c /*= 1.0*/, double gamma /*= 1.0*/) {
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

void ImgProcessor::HistEqualization(cv::Mat const &src, cv::Mat &dst) {
    cv::Mat tmp;
    if (src.channels() == 3) {
        cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 1) {
        tmp = src;
    }

    // 直方图均衡化
    cv::equalizeHist(tmp, dst);
}

void ImgProcessor::HistMatch(cv::Mat const &src, cv::Mat const &pattern, cv::Mat &dst) {
    cv::Mat tmp;
    if (src.channels() == 3) {
        cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 1) {
        tmp = src;
    }

    cv::Mat _pattern;
    if (pattern.channels() == 3) {
        cv::cvtColor(pattern, _pattern, cv::COLOR_BGR2GRAY);
    } else if (pattern.channels() == 1) {
        _pattern = pattern;
    }

    cv::Mat equalizeHist1, equalizeHist2;
    // 图像1和图像2进行均衡化
    cv::equalizeHist(tmp, equalizeHist1);
    cv::equalizeHist(_pattern, equalizeHist2);

    // 求图像1和图像2均衡化后的直方图
    cv::Mat hist1, hist2;
    int hSize = 256;
    float ranges[] = {0, 256};
    const float *hranges = {ranges};
    cv::calcHist(&equalizeHist1, 1, 0, cv::Mat(), hist1, 1, &hSize, &hranges, true, false);
    cv::calcHist(&equalizeHist2, 1, 0, cv::Mat(), hist2, 1, &hSize, &hranges, true, false);

    // 计算两个均衡化图像直方图的累积概率
    float hist1Rate[256] = {hist1.at<float>(0)};
    float hist2Rate[256] = {hist2.at<float>(0)};
    for (int i = 1; i < 256; i++) {
        hist1Rate[i] = hist1Rate[i - 1] + hist1.at<float>(i);
        hist2Rate[i] = hist2Rate[i - 1] + hist2.at<float>(i);
    }

    for (int i = 0; i < 256; i++) {
        hist1Rate[i] /= (equalizeHist1.rows * equalizeHist1.cols);
        hist2Rate[i] /= (equalizeHist2.rows * equalizeHist2.cols);
    }

    // 两个累计概率之间的差值，用于找到最接近的点
    float diff[256][256];
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            diff[i][j] = fabs(hist1Rate[i] - hist2Rate[j]);
        }
    }

    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; i++) {
        // 查找源灰度级为i的映射灰度和i的累积概率差最小(灰度接近)的规定化灰度
        float min = diff[i][0];
        int idx = 0;
        for (int j = 0; j < 256; j++) {
            if (min > diff[i][j]) {
                min = diff[i][j];
                idx = j;
            }
        }
        lut.at<uchar>(i) = idx;
    }

    cv::LUT(equalizeHist1, lut, dst);
}

void ImgProcessor::AddSaltNoice(cv::Mat &iSrc, int iNum) {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> randomRow(0, iSrc.rows - 1);
    std::uniform_int_distribution<int> randomCol(0, iSrc.cols - 1);

    for (int k = 0; k < iNum; ++k) {
        int i = randomRow(generator);
        int j = randomCol(generator);
        if (iSrc.channels() == 3) {
            iSrc.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
        } else if (iSrc.channels() == 1) {
            iSrc.at<uchar>(i, j) = 255;
        }
    }
}

void ImgProcessor::MeanFilter(cv::Mat const &iSrc, cv::Mat &oDst, int iFilterSize) {
    oDst = iSrc.clone();
    int k = (iFilterSize - 1) / 2;
    cv::copyMakeBorder(iSrc, oDst, k, k, k, k, cv::BORDER_REFLECT);

    if (iSrc.channels() == 3) {
        for (int i = k; i < oDst.rows - k; i++) {
            for (int j = k; j < oDst.cols - k; j++) {
                cv::Vec3i sum{0, 0, 0};
                // 卷积过程
                for (int x = -k; x <= k; x++) {
                    for (int y = -k; y <= k; y++) {
                        sum += oDst.at<cv::Vec3b>(i + x, j + y);
                    }
                }
                oDst.at<cv::Vec3b>(i, j) = cv::Vec3b(
                    cv::saturate_cast<uchar>(sum[0] / (iFilterSize * iFilterSize)),
                    cv::saturate_cast<uchar>(sum[1] / (iFilterSize * iFilterSize)),
                    cv::saturate_cast<uchar>(sum[2] / (iFilterSize * iFilterSize)));
            }
        }
    } else if (iSrc.channels() == 1) {
        for (int i = k; i < oDst.rows - k; i++) {
            for (int j = k; j < oDst.cols - k; j++) {
                int sum = 0;
                for (int x = -k; x <= k; x++) {
                    for (int y = -k; y <= k; y++) {
                        sum += oDst.at<uchar>(i + x, j + y);
                    }
                }
                oDst.at<uchar>(i, j) = cv::saturate_cast<uchar>(sum / (iFilterSize * iFilterSize));
            }
        }
    }
    
    oDst = oDst(cv::Rect(k, k, iSrc.cols, iSrc.rows));
}