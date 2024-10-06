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
     * @param[in] c 对数变换因子
     */
    static void GrayLogTrans(cv::Mat const &src, cv::Mat &dst, double c = 1.0);

    /**
     * @brief 灰度 gamma 变换，扩展图像中的暗像素值，压缩高灰度值
     * @details y = c * x ^ gamma
     * @param[in] src 输入图像
     * @param[out] dst 输出图像
     * @param[in] c 因子
     * @param[in] gamma gamma 值
     */
    static void GrayGammaTrans(cv::Mat const &src, cv::Mat &dst, double c = 1.0, double gamma = 1.0);

    /**
     * @brief 获取src的直方图
     * @details
     * @param[in] src 输入图像
     * @param[out] dst 直方图
     */
    static void GetHistogram(cv::Mat const &src, cv::Mat &dst);

    /**
     * @brief 直方图均衡化
     * @details
     * @param[in] src 输入图像
     * @param[out] dst均衡化后的图像
     */
    static void HistEqualization(cv::Mat const &src, cv::Mat &dst);

    /**
     * @brief 直方图匹配
     * @details
     * @param[in] src 输入图像
     * @param[in] pattern 模板图像
     * @param[out] dst 匹配后的图像
     */
    static void HistMatch(cv::Mat const &src, cv::Mat const &pattern, cv::Mat &dst);

    /**
     * @brief 添加椒盐噪声
     * @details
     * @param[in,out] iSrc 输入图像
     * @param[in] iNum 椒盐噪声的个数
     */
    static void AddSaltNoice(cv::Mat &iSrc, int iNum);

    /**
     * @brief 均值滤波
     * @details
     * @param[in] iSrc 输入图像
     * @param[out] oDst 输出图像
     * @param[in] iFilterSize 滤波器的 size
     */
    static void MeanFilter(cv::Mat const &iSrc, cv::Mat &oDst, int iFilterSize);

    /**
     * @brief 高斯滤波
     * @details
     * @param[in] iSrc 输入图像
     * @param[out] oDst 输出图像
     * @param[in] iFilterSize 滤波器的 size
     * @param[in] iSigmaX 高斯分布X的标准差
     * @param[in] iSigmaY 高斯分布Y的标准差，不使用
     */
    static void GaussianFilter(
        cv::Mat const &iSrc,
        cv::Mat &oDst,
        int iFilterSize,
        double iSigmaX,
        double iSigmaY = 0.0);

    /**
     * @brief 中值滤波
     * @details
     * @param[in] iSrc 输入图像
     * @param[out] oDst 输出图像
     * @param[in] iFilterSize 滤波器的 size
     */
    static void MedianFilter(cv::Mat const &iSrc, cv::Mat &oDst, int iFilterSize);
};