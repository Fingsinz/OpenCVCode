#include "ImgProcessor.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/matx.hpp"
#include "opencv2/core/saturate.hpp"
#include "opencv2/imgproc.hpp"
#include <vector>

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
    for (int i = 1; i < 256; ++i) {
        hist1Rate[i] = hist1Rate[i - 1] + hist1.at<float>(i);
        hist2Rate[i] = hist2Rate[i - 1] + hist2.at<float>(i);
    }

    for (int i = 0; i < 256; ++i) {
        hist1Rate[i] /= (equalizeHist1.rows * equalizeHist1.cols);
        hist2Rate[i] /= (equalizeHist2.rows * equalizeHist2.cols);
    }

    // 两个累计概率之间的差值，用于找到最接近的点
    float diff[256][256];
    for (int i = 0; i < 256; ++i) {
        for (int j = 0; j < 256; ++j) {
            diff[i][j] = fabs(hist1Rate[i] - hist2Rate[j]);
        }
    }

    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i) {
        // 查找源灰度级为i的映射灰度和i的累积概率差最小(灰度接近)的规定化灰度
        float min = diff[i][0];
        int idx = 0;
        for (int j = 0; j < 256; ++j) {
            if (min > diff[i][j]) {
                min = diff[i][j];
                idx = j;
            }
        }
        lut.at<uchar>(i) = idx;
    }

    cv::LUT(equalizeHist1, lut, dst);
}

void ImgProcessor::MeanFilter(cv::Mat const &iSrc, cv::Mat &oDst, int iFilterSize) {
    int k = (iFilterSize - 1) / 2;
    cv::copyMakeBorder(iSrc, oDst, k, k, k, k, cv::BORDER_REFLECT);
    cv::Mat tmpSrc = oDst.clone();

    if (iSrc.channels() == 3) {
        for (int i = k; i < oDst.rows - k; ++i) {
            for (int j = k; j < oDst.cols - k; ++j) {
                cv::Vec3i sum{0, 0, 0};
                // 卷积过程
                for (int x = -k; x <= k; x++) {
                    for (int y = -k; y <= k; y++) {
                        sum += tmpSrc.at<cv::Vec3b>(i + x, j + y);
                    }
                }
                oDst.at<cv::Vec3b>(i, j) = cv::Vec3b(
                    cv::saturate_cast<uchar>(sum[0] / (iFilterSize * iFilterSize)),
                    cv::saturate_cast<uchar>(sum[1] / (iFilterSize * iFilterSize)),
                    cv::saturate_cast<uchar>(sum[2] / (iFilterSize * iFilterSize)));
            }
        }
    } else if (iSrc.channels() == 1) {
        for (int i = k; i < oDst.rows - k; ++i) {
            for (int j = k; j < oDst.cols - k; ++j) {
                int sum = 0;
                for (int x = -k; x <= k; x++) {
                    for (int y = -k; y <= k; y++) {
                        sum += tmpSrc.at<uchar>(i + x, j + y);
                    }
                }
                oDst.at<uchar>(i, j) = cv::saturate_cast<uchar>(sum / (iFilterSize * iFilterSize));
            }
        }
    }

    oDst = oDst(cv::Rect(k, k, iSrc.cols, iSrc.rows));
}

void ImgProcessor::GaussianFilter(
    cv::Mat const &iSrc,
    cv::Mat &oDst,
    int iFilterSize,
    double iSigmaX,
    double iSigmaY /*= 0.0*/) {

    int k = (iFilterSize - 1) / 2;
    cv::copyMakeBorder(iSrc, oDst, k, k, k, k, cv::BORDER_REFLECT);
    cv::Mat tmpSrc = oDst.clone();

    if (iSrc.channels() == 3) {
        for (int i = k; i < oDst.rows - k; ++i) {
            for (int j = k; j < oDst.cols - k; ++j) {
                cv::Vec3d sum{0.0, 0.0, 0.0};
                double g;
                double sumG = 0.0;
                // 卷积过程
                for (int x = -k; x <= k; x++) {
                    for (int y = -k; y <= k; y++) {
                        g = exp(-(x * x + y * y) / (2 * iSigmaX * iSigmaX));
                        sumG += g;
                        sum += tmpSrc.at<cv::Vec3b>(i + x, j + y) * g;
                    }
                }
                oDst.at<cv::Vec3b>(i, j) = cv::Vec3b(
                    cv::saturate_cast<uchar>(sum[0] / sumG),
                    cv::saturate_cast<uchar>(sum[1] / sumG),
                    cv::saturate_cast<uchar>(sum[2] / sumG));
            }
        }
    } else if (iSrc.channels() == 1) {
        for (int i = k; i < oDst.rows - k; ++i) {
            for (int j = k; j < oDst.cols - k; ++j) {
                double sum = 0.0;
                double g;
                double sumG = 0.0;
                for (int x = -k; x <= k; x++) {
                    for (int y = -k; y <= k; y++) {
                        g = exp(-(x * x + y * y) / (2 * iSigmaX * iSigmaX));
                        sumG += g;
                        sum += tmpSrc.at<uchar>(i + x, j + y) * g;
                    }
                }
                oDst.at<uchar>(i, j) = cv::saturate_cast<uchar>(sum / sumG);
            }
        }
    }

    oDst = oDst(cv::Rect(k, k, iSrc.cols, iSrc.rows));
}

void ImgProcessor::MedianFilter(cv::Mat const &iSrc, cv::Mat &oDst, int iFilterSize) {
    int k = (iFilterSize - 1) / 2;
    cv::copyMakeBorder(iSrc, oDst, k, k, k, k, cv::BORDER_REFLECT);
    cv::Mat tmpSrc = oDst.clone();

    if (iSrc.channels() == 3) {
        std::vector<cv::Vec3b> tmp(iFilterSize * iFilterSize);

        for (int i = k; i < oDst.rows - k; ++i) {
            for (int j = k; j < oDst.cols - k; ++j) {
                tmp.clear();
                for (int x = -k; x <= k; x++) {
                    for (int y = -k; y <= k; y++) {
                        tmp.push_back(tmpSrc.at<cv::Vec3b>(i + x, j + y));
                    }
                }
                std::sort(tmp.begin(), tmp.end(), [&](cv::Vec3b a, cv::Vec3b b) {
                    return a[0] + a[1] + a[2] < b[0] + b[1] + b[2];
                });
                oDst.at<cv::Vec3b>(i, j) = tmp[tmp.size() / 2];
            }
        }
    } else if (iSrc.channels() == 1) {
        std::vector<uchar> tmp(iFilterSize * iFilterSize);

        for (int i = k; i < oDst.rows - k; ++i) {
            for (int j = k; j < oDst.cols - k; ++j) {
                tmp.clear();
                for (int x = -k; x <= k; x++) {
                    for (int y = -k; y <= k; y++) {
                        tmp.push_back(tmpSrc.at<uchar>(i + x, j + y));
                    }
                }
                std::sort(tmp.begin(), tmp.end());
                oDst.at<uchar>(i, j) = tmp[tmp.size() / 2];
            }
        }
    }

    oDst = oDst(cv::Rect(k, k, iSrc.cols, iSrc.rows));
}

void ImgProcessor::LaplacianFilter(cv::Mat const &iSrc, cv::Mat &oDst, bool ibAll /*= false*/) {
    auto filter4 = [](cv::Mat const &iSrc, cv::Mat &oDst) {
        cv::copyMakeBorder(iSrc, oDst, 1, 1, 1, 1, cv::BORDER_REFLECT);
        cv::Mat tmp = oDst.clone();

        if (iSrc.channels() == 3) {
            cv::Vec3i la{0, 0, 0};
            for (int i = 1; i < oDst.rows - 1; ++i) {
                for (int j = 1; j < oDst.cols - 1; ++j) {
                    // 拉普拉斯核
                    // 0  1  0
                    // 1 -4  1
                    // 0  1  0
                    la[0] = tmp.at<cv::Vec3b>(i + 1, j)[0] + tmp.at<cv::Vec3b>(i - 1, j)[0] +
                        tmp.at<cv::Vec3b>(i, j + 1)[0] + tmp.at<cv::Vec3b>(i, j - 1)[0] -
                        4 * tmp.at<cv::Vec3b>(i, j)[0];
                    la[1] = tmp.at<cv::Vec3b>(i + 1, j)[1] + tmp.at<cv::Vec3b>(i - 1, j)[1] +
                        tmp.at<cv::Vec3b>(i, j + 1)[1] + tmp.at<cv::Vec3b>(i, j - 1)[1] -
                        4 * tmp.at<cv::Vec3b>(i, j)[1];
                    la[2] = tmp.at<cv::Vec3b>(i + 1, j)[2] + tmp.at<cv::Vec3b>(i - 1, j)[2] +
                        tmp.at<cv::Vec3b>(i, j + 1)[2] + tmp.at<cv::Vec3b>(i, j - 1)[2] -
                        4 * tmp.at<cv::Vec3b>(i, j)[2];

                    oDst.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(oDst.at<cv::Vec3b>(i, j)[0] - la[0]);
                    oDst.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(oDst.at<cv::Vec3b>(i, j)[1] - la[1]);
                    oDst.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(oDst.at<cv::Vec3b>(i, j)[2] - la[2]);
                }
            }
        } else if (iSrc.channels() == 1) {
            for (int i = 1; i < oDst.rows - 1; ++i) {
                for (int j = 1; j < oDst.cols - 1; ++j) {
                    // 拉普拉斯核
                    // 0  -1  0
                    // -1  4  -1
                    // 0  -1  0
                    int la = 4 * tmp.at<uchar>(i, j) - tmp.at<uchar>(i + 1, j) - tmp.at<uchar>(i - 1, j) -
                        tmp.at<uchar>(i, j + 1) - tmp.at<uchar>(i, j - 1);

                    oDst.at<uchar>(i, j) = cv::saturate_cast<uchar>(oDst.at<uchar>(i, j) + la);
                }
            }
        }

        oDst = oDst(cv::Rect(1, 1, iSrc.cols, iSrc.rows));
    };

    auto filter8 = [](cv::Mat const &iSrc, cv::Mat &oDst) {
        cv::copyMakeBorder(iSrc, oDst, 1, 1, 1, 1, cv::BORDER_REFLECT);
        cv::Mat tmp = oDst.clone();

        if (iSrc.channels() == 3) {
            cv::Vec3i la{0, 0, 0};
            for (int i = 1; i < oDst.rows - 1; ++i) {
                for (int j = 1; j < oDst.cols - 1; ++j) {
                    // 拉普拉斯核
                    // -1 -1 -1
                    // -1  8 -1
                    // -1 -1 -1
                    la[0] = 8 * tmp.at<cv::Vec3b>(i, j)[0] - tmp.at<cv::Vec3b>(i - 1, j - 1)[0] -
                        tmp.at<cv::Vec3b>(i - 1, j)[0] - tmp.at<cv::Vec3b>(i - 1, j + 1)[0] -
                        tmp.at<cv::Vec3b>(i, j - 1)[0] - tmp.at<cv::Vec3b>(i, j + 1)[0] -
                        tmp.at<cv::Vec3b>(i + 1, j - 1)[0] - tmp.at<cv::Vec3b>(i + 1, j)[0] -
                        tmp.at<cv::Vec3b>(i + 1, j + 1)[0];
                    la[1] = 8 * tmp.at<cv::Vec3b>(i, j)[1] - tmp.at<cv::Vec3b>(i - 1, j - 1)[1] -
                        tmp.at<cv::Vec3b>(i - 1, j)[1] - tmp.at<cv::Vec3b>(i - 1, j + 1)[1] -
                        tmp.at<cv::Vec3b>(i, j - 1)[1] - tmp.at<cv::Vec3b>(i, j + 1)[1] -
                        tmp.at<cv::Vec3b>(i + 1, j - 1)[1] - tmp.at<cv::Vec3b>(i + 1, j)[1] -
                        tmp.at<cv::Vec3b>(i + 1, j + 1)[1];
                    la[2] = 8 * tmp.at<cv::Vec3b>(i, j)[2] - tmp.at<cv::Vec3b>(i - 1, j - 1)[2] -
                        tmp.at<cv::Vec3b>(i - 1, j)[2] - tmp.at<cv::Vec3b>(i - 1, j + 1)[2] -
                        tmp.at<cv::Vec3b>(i, j - 1)[2] - tmp.at<cv::Vec3b>(i, j + 1)[2] -
                        tmp.at<cv::Vec3b>(i + 1, j - 1)[2] - tmp.at<cv::Vec3b>(i + 1, j)[2] -
                        tmp.at<cv::Vec3b>(i + 1, j + 1)[2];

                    oDst.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(oDst.at<cv::Vec3b>(i, j)[0] + la[0]);
                    oDst.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(oDst.at<cv::Vec3b>(i, j)[1] + la[1]);
                    oDst.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(oDst.at<cv::Vec3b>(i, j)[2] + la[2]);
                }
            }
        } else if (iSrc.channels() == 1) {
            for (int i = 1; i < oDst.rows - 1; ++i) {
                for (int j = 1; j < oDst.cols - 1; ++j) {
                    // 拉普拉斯核
                    // -1 -1 -1
                    // -1  8 -1
                    // -1 -1 -1
                    int la = 8 * oDst.at<uchar>(i, j) - tmp.at<uchar>(i - 1, j - 1) - tmp.at<uchar>(i - 1, j) -
                        tmp.at<uchar>(i - 1, j + 1) - tmp.at<uchar>(i, j - 1) - tmp.at<uchar>(i, j + 1) -
                        tmp.at<uchar>(i + 1, j - 1) - tmp.at<uchar>(i + 1, j) - tmp.at<uchar>(i + 1, j + 1);

                    oDst.at<uchar>(i, j) = cv::saturate_cast<uchar>(oDst.at<uchar>(i, j) + la);
                }
            }
        }

        oDst = oDst(cv::Rect(1, 1, iSrc.cols, iSrc.rows));
    };

    if (ibAll) {
        filter8(iSrc, oDst);
    } else {
        filter4(iSrc, oDst);
    }
}

auto filter4 = [](cv::Mat const &iSrc, cv::Mat &oDst) {
    cv::copyMakeBorder(iSrc, oDst, 1, 1, 1, 1, cv::BORDER_REFLECT);
    cv::Mat tmp = oDst.clone();

    if (iSrc.channels() == 3) {
        cv::Vec3i la{0, 0, 0};
        for (int i = 1; i < oDst.rows - 1; ++i) {
            for (int j = 1; j < oDst.cols - 1; ++j) {
                // 拉普拉斯核
                // 0  1  0
                // 1 -4  1
                // 0  1  0
                la[0] = tmp.at<cv::Vec3b>(i + 1, j)[0] + tmp.at<cv::Vec3b>(i - 1, j)[0] +
                    tmp.at<cv::Vec3b>(i, j + 1)[0] + tmp.at<cv::Vec3b>(i, j - 1)[0] - 4 * tmp.at<cv::Vec3b>(i, j)[0];
                la[1] = tmp.at<cv::Vec3b>(i + 1, j)[1] + tmp.at<cv::Vec3b>(i - 1, j)[1] +
                    tmp.at<cv::Vec3b>(i, j + 1)[1] + tmp.at<cv::Vec3b>(i, j - 1)[1] - 4 * tmp.at<cv::Vec3b>(i, j)[1];
                la[2] = tmp.at<cv::Vec3b>(i + 1, j)[2] + tmp.at<cv::Vec3b>(i - 1, j)[2] +
                    tmp.at<cv::Vec3b>(i, j + 1)[2] + tmp.at<cv::Vec3b>(i, j - 1)[2] - 4 * tmp.at<cv::Vec3b>(i, j)[2];

                oDst.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(oDst.at<cv::Vec3b>(i, j)[0] - la[0]);
                oDst.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(oDst.at<cv::Vec3b>(i, j)[1] - la[1]);
                oDst.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(oDst.at<cv::Vec3b>(i, j)[2] - la[2]);
            }
        }
    } else if (iSrc.channels() == 1) {
        for (int i = 1; i < oDst.rows - 1; ++i) {
            for (int j = 1; j < oDst.cols - 1; ++j) {
                // 拉普拉斯核
                // 0  -1  0
                // -1  4  -1
                // 0  -1  0
                int la = 4 * tmp.at<uchar>(i, j) - tmp.at<uchar>(i + 1, j) - tmp.at<uchar>(i - 1, j) -
                    tmp.at<uchar>(i, j + 1) - tmp.at<uchar>(i, j - 1);

                oDst.at<uchar>(i, j) = cv::saturate_cast<uchar>(oDst.at<uchar>(i, j) + la);
            }
        }
    }

    oDst = oDst(cv::Rect(1, 1, iSrc.cols, iSrc.rows));
};

void ImgProcessor::ArithmeticMeanFilter(cv::Mat const &iSrc, cv::Mat &oDst, cv::Size iFilterSize) {
    int m = (iFilterSize.height - 1) / 2;
    int n = (iFilterSize.width - 1) / 2;
    int area = iFilterSize.area();
    cv::copyMakeBorder(iSrc, oDst, m, m, n, n, cv::BORDER_REFLECT);
    cv::Mat tmpSrc = oDst.clone();

    if (iSrc.channels() == 3) {
        for (int i = m; i < oDst.rows - m; ++i) {
            for (int j = n; j < oDst.cols - n; ++j) {
                cv::Vec3i sum{0, 0, 0};
                for (int x = -m; x <= m; x++) {
                    for (int y = -n; y <= n; y++) {
                        sum += tmpSrc.at<cv::Vec3b>(i + x, j + y);
                    }
                }
                oDst.at<cv::Vec3b>(i, j) = cv::Vec3b(
                    cv::saturate_cast<uchar>(sum[0] / area),
                    cv::saturate_cast<uchar>(sum[1] / area),
                    cv::saturate_cast<uchar>(sum[2] / area));
            }
        }
    } else if (iSrc.channels() == 1) {
        for (int i = m; i < oDst.rows - m; ++i) {
            for (int j = n; j < oDst.cols - n; ++j) {
                int sum = 0;
                for (int x = -m; x <= m; x++) {
                    for (int y = -n; y <= n; y++) {
                        sum += tmpSrc.at<uchar>(i + x, j + y);
                    }
                }
                oDst.at<uchar>(i, j) = cv::saturate_cast<uchar>(sum / area);
            }
        }
    }

    oDst = oDst(cv::Rect(m, n, iSrc.cols, iSrc.rows));
}

void ImgProcessor::GeometricMeanFilter(cv::Mat const &iSrc, cv::Mat &oDst, cv::Size iFilterSize) {
    int m = (iFilterSize.height - 1) / 2;
    int n = (iFilterSize.width - 1) / 2;
    int area = iFilterSize.area();
    cv::copyMakeBorder(iSrc, oDst, m, m, n, n, cv::BORDER_REFLECT);
    cv::Mat tmpSrc = oDst.clone();

    if (iSrc.channels() == 3) {
        for (int i = m; i < oDst.rows - m; ++i) {
            for (int j = n; j < oDst.cols - n; ++j) {
                cv::Vec3d sum{0.0, 0.0, 0.0};
                for (int x = -m; x <= m; x++) {
                    for (int y = -n; y <= n; y++) {
                        // 对数运算，避免数值连乘过大
                        sum[0] += log10(0.1 + tmpSrc.at<cv::Vec3b>(i + x, j + y)[0]);
                        sum[1] += log10(0.1 + tmpSrc.at<cv::Vec3b>(i + x, j + y)[1]);
                        sum[2] += log10(0.1 + tmpSrc.at<cv::Vec3b>(i + x, j + y)[2]);
                    }
                }
                sum /= area;
                oDst.at<cv::Vec3b>(i, j) = cv::Vec3b(
                    cv::saturate_cast<uchar>(pow(10, sum[0])),
                    cv::saturate_cast<uchar>(pow(10, sum[1])),
                    cv::saturate_cast<uchar>(pow(10, sum[2])));
            }
        }
    } else if (iSrc.channels() == 1) {
        for (int i = m; i < oDst.rows - m; ++i) {
            for (int j = n; j < oDst.cols - n; ++j) {
                double sum = 0;
                for (int x = -m; x <= m; x++) {
                    for (int y = -n; y <= n; y++) {
                        sum += log10(0.1 + tmpSrc.at<uchar>(i + x, j + y));
                    }
                }
                sum /= area;
                oDst.at<uchar>(i, j) = cv::saturate_cast<uchar>(pow(10, sum));
            }
        }
    }

    oDst = oDst(cv::Rect(m, n, iSrc.cols, iSrc.rows));
}

void ImgProcessor::HarmonicMeanFilter(cv::Mat const &iSrc, cv::Mat &oDst, cv::Size iFilterSize) {
    int m = (iFilterSize.height - 1) / 2;
    int n = (iFilterSize.width - 1) / 2;
    int area = iFilterSize.area();
    cv::copyMakeBorder(iSrc, oDst, m, m, n, n, cv::BORDER_REFLECT);
    cv::Mat tmpSrc = oDst.clone();

    if (iSrc.channels() == 3) {
        for (int i = m; i < oDst.rows - m; ++i) {
            for (int j = n; j < oDst.cols - n; ++j) {
                cv::Vec3d sum{0.0, 0.0, 0.0};
                for (int x = -m; x <= m; x++) {
                    for (int y = -n; y <= n; y++) {
                        sum[0] += 1.0 / (0.1 + tmpSrc.at<cv::Vec3b>(i + x, j + y)[0]);
                        sum[1] += 1.0 / (0.1 + tmpSrc.at<cv::Vec3b>(i + x, j + y)[1]);
                        sum[2] += 1.0 / (0.1 + tmpSrc.at<cv::Vec3b>(i + x, j + y)[2]);
                    }
                }
                sum[0] = static_cast<double>(area) / sum[0];
                sum[1] = static_cast<double>(area) / sum[1];
                sum[2] = static_cast<double>(area) / sum[2];
                oDst.at<cv::Vec3b>(i, j) = sum;
            }
        }
    } else if (iSrc.channels() == 1) {
        for (int i = m; i < oDst.rows - m; ++i) {
            for (int j = n; j < oDst.cols - n; ++j) {
                double sum = 0;
                for (int x = -m; x <= m; x++) {
                    for (int y = -n; y <= n; y++) {
                        sum += 1.0 / (0.1 + tmpSrc.at<uchar>(i + x, j + y));
                    }
                }
                sum = static_cast<double>(area) / sum;
                oDst.at<uchar>(i, j) = cv::saturate_cast<uchar>(sum);
            }
        }
    }

    oDst = oDst(cv::Rect(m, n, iSrc.cols, iSrc.rows));
}

void ImgProcessor::AntiHarmonicMeanFilter(cv::Mat const &iSrc, cv::Mat &oDst, cv::Size iFilterSize, double q /*= 0*/) {
    int m = (iFilterSize.height - 1) / 2;
    int n = (iFilterSize.width - 1) / 2;
    int area = iFilterSize.area();
    cv::copyMakeBorder(iSrc, oDst, m, m, n, n, cv::BORDER_REFLECT);
    cv::Mat tmpSrc = oDst.clone();

    if (iSrc.channels() == 3) {
        for (int i = m; i < oDst.rows - m; ++i) {
            for (int j = n; j < oDst.cols - n; ++j) {
                cv::Vec3d sum1{0.0, 0.0, 0.0}, sum2{0.0, 0.0, 0.0};
                for (int x = -m; x <= m; x++) {
                    for (int y = -n; y <= n; y++) {
                        sum1[0] += pow(tmpSrc.at<cv::Vec3b>(i + x, j + y)[0], q + 1);
                        sum1[1] += pow(tmpSrc.at<cv::Vec3b>(i + x, j + y)[1], q + 1);
                        sum1[2] += pow(tmpSrc.at<cv::Vec3b>(i + x, j + y)[2], q + 1);
                        sum2[0] += pow(tmpSrc.at<cv::Vec3b>(i + x, j + y)[0], q);
                        sum2[1] += pow(tmpSrc.at<cv::Vec3b>(i + x, j + y)[1], q);
                        sum2[2] += pow(tmpSrc.at<cv::Vec3b>(i + x, j + y)[2], q);
                    }
                }
                cv::Vec3b sum{
                    cv::saturate_cast<uchar>(sum1[0] / sum2[0]),
                    cv::saturate_cast<uchar>(sum1[1] / sum2[1]),
                    cv::saturate_cast<uchar>(sum1[2] / sum2[2])};
                oDst.at<cv::Vec3b>(i, j) = sum;
            }
        }
    } else if (iSrc.channels() == 1) {
        for (int i = m; i < oDst.rows - m; ++i) {
            for (int j = n; j < oDst.cols - n; ++j) {
                double sum1 = 0.0, sum2 = 0.0;
                for (int x = -m; x <= m; x++) {
                    for (int y = -n; y <= n; y++) {
                        sum1 += pow(tmpSrc.at<uchar>(i + x, j + y), q + 1);
                        sum2 += pow(tmpSrc.at<uchar>(i + x, j + y), q);
                    }
                }
                double sum = sum1 / sum2;
                oDst.at<uchar>(i, j) = cv::saturate_cast<uchar>(sum);
            }
        }
    }

    oDst = oDst(cv::Rect(m, n, iSrc.cols, iSrc.rows));
}

void ImgProcessor::MedianFilter(cv::Mat const &iSrc, cv::Mat &oDst, cv::Size iFilterSize) {
    int m = (iFilterSize.height - 1) / 2;
    int n = (iFilterSize.width - 1) / 2;
    int area = iFilterSize.area();
    cv::copyMakeBorder(iSrc, oDst, m, m, n, n, cv::BORDER_REFLECT);
    cv::Mat tmpSrc = oDst.clone();

    if (iSrc.channels() == 3) {
        std::vector<cv::Vec3b> tmp(area);

        for (int i = m; i < oDst.rows - m; ++i) {
            for (int j = n; j < oDst.cols - n; ++j) {
                tmp.clear();
                for (int x = -m; x <= m; x++) {
                    for (int y = -n; y <= n; y++) {
                        tmp.push_back(tmpSrc.at<cv::Vec3b>(i + x, j + y));
                    }
                }
                std::sort(tmp.begin(), tmp.end(), [&](cv::Vec3b a, cv::Vec3b b) {
                    return a[0] + a[1] + a[2] < b[0] + b[1] + b[2];
                });
                oDst.at<cv::Vec3b>(i, j) = tmp[tmp.size() / 2];
            }
        }
    } else if (iSrc.channels() == 1) {
        std::vector<uchar> tmp(area);

        for (int i = m; i < oDst.rows - m; ++i) {
            for (int j = n; j < oDst.cols - n; ++j) {
                tmp.clear();
                for (int x = -m; x <= m; x++) {
                    for (int y = -n; y <= n; y++) {
                        tmp.push_back(tmpSrc.at<uchar>(i + x, j + y));
                    }
                }
                std::sort(tmp.begin(), tmp.end());
                oDst.at<uchar>(i, j) = tmp[tmp.size() / 2];
            }
        }
    }

    oDst = oDst(cv::Rect(m, n, iSrc.cols, iSrc.rows));
}

void ImgProcessor::MinMaxFilter(cv::Mat const &iSrc, cv::Mat &oDst, cv::Size iFilterSize, bool ibMax /*= true*/) {
    int m = (iFilterSize.height - 1) / 2;
    int n = (iFilterSize.width - 1) / 2;
    int area = iFilterSize.area();
    cv::copyMakeBorder(iSrc, oDst, m, m, n, n, cv::BORDER_REFLECT);

    auto minFilter = [&area, &m, &n](cv::Mat const &iSrc, cv::Mat &oDst) {
        cv::Mat tmpSrc = oDst.clone();
        if (iSrc.channels() == 3) {
            std::vector<cv::Vec3b> tmp(area);

            for (int i = m; i < oDst.rows - m; ++i) {
                for (int j = n; j < oDst.cols - n; ++j) {
                    tmp.clear();
                    for (int x = -m; x <= m; x++) {
                        for (int y = -n; y <= n; y++) {
                            tmp.push_back(tmpSrc.at<cv::Vec3b>(i + x, j + y));
                        }
                    }
                    std::sort(tmp.begin(), tmp.end(), [&](cv::Vec3b a, cv::Vec3b b) {
                        return a[0] + a[1] + a[2] < b[0] + b[1] + b[2];
                    });
                    oDst.at<cv::Vec3b>(i, j) = tmp[0];
                }
            }
        } else if (iSrc.channels() == 1) {
            std::vector<uchar> tmp(area);

            for (int i = m; i < oDst.rows - m; ++i) {
                for (int j = n; j < oDst.cols - n; ++j) {
                    tmp.clear();
                    for (int x = -m; x <= m; x++) {
                        for (int y = -n; y <= n; y++) {
                            tmp.push_back(tmpSrc.at<uchar>(i + x, j + y));
                        }
                    }
                    std::sort(tmp.begin(), tmp.end());
                    oDst.at<uchar>(i, j) = tmp[0];
                }
            }
        }
    };

    auto maxFilter = [&area, &m, &n](cv::Mat const &iSrc, cv::Mat &oDst) {
        cv::Mat tmpSrc = oDst.clone();
        if (iSrc.channels() == 3) {
            std::vector<cv::Vec3b> tmp(area);

            for (int i = m; i < oDst.rows - m; ++i) {
                for (int j = n; j < oDst.cols - n; ++j) {
                    tmp.clear();
                    for (int x = -m; x <= m; x++) {
                        for (int y = -n; y <= n; y++) {
                            tmp.push_back(tmpSrc.at<cv::Vec3b>(i + x, j + y));
                        }
                    }
                    std::sort(tmp.begin(), tmp.end(), [&](cv::Vec3b a, cv::Vec3b b) {
                        return a[0] + a[1] + a[2] < b[0] + b[1] + b[2];
                    });
                    oDst.at<cv::Vec3b>(i, j) = tmp[tmp.size() - 1];
                }
            }
        } else if (iSrc.channels() == 1) {
            std::vector<uchar> tmp(area);

            for (int i = m; i < oDst.rows - m; ++i) {
                for (int j = n; j < oDst.cols - n; ++j) {
                    tmp.clear();
                    for (int x = -m; x <= m; x++) {
                        for (int y = -n; y <= n; y++) {
                            tmp.push_back(tmpSrc.at<uchar>(i + x, j + y));
                        }
                    }
                    std::sort(tmp.begin(), tmp.end());
                    oDst.at<uchar>(i, j) = tmp[tmp.size() - 1];
                }
            }
        }
    };

    if (ibMax) {
        maxFilter(iSrc, oDst);
    } else {
        minFilter(iSrc, oDst);
    }

    oDst = oDst(cv::Rect(m, n, iSrc.cols, iSrc.rows));
}