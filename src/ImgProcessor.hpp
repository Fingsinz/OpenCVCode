#pragma once

#include "opencv2/core/mat.hpp"

class ImgProcessor {
public:
    /**
     * @brief 灰度反转，将图像亮暗对调，可以增强图像中的暗色区域细节。
     * @details y = 255 - x
     * @param[in] src 输入图像
     * @param[out] dst 输出图像
     */
    static void GrayInversion(cv::Mat const &src, cv::Mat &dst);

    /**
     * @brief 灰度对数变换，扩展图像中的暗像素值，压缩高灰度值
     * @details y = c * ln(1 + x)
     * @param[in] src 输入图像
     * @param[out] dst 输出图像
     */
    static void GrayLogTrans(cv::Mat const &src, cv::Mat &dst, double c = 1.0);
};