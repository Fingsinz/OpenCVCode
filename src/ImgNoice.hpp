#pragma once

#include "opencv2/core/mat.hpp"

class ImgNoice {
public:
    static void AddSaltNoice(cv::Mat &iSrc, int iNum);
    static void AddGaussianNoice(cv::Mat &iSrc, double mean = 10, double sigma = 50);
    static void AddUniformNoice(cv::Mat &iSrc, double low = 10, double high = 50);
};