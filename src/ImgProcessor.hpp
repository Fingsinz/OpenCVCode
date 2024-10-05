#pragma once

#include "opencv2/core/mat.hpp"

class ImgProcessor {
public:
    static void GrayInversion(cv::Mat const &src, cv::Mat &dst);
};