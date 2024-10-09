#pragma once

#include "opencv2/core/mat.hpp"

namespace ImgNoice {

/**
 * @brief 加盐噪声
 * @param[in, out] iSrc 输入输出图像
 * @param iNum 盐噪声的数量
 */
 void AddSaltNoice(cv::Mat &iSrc, int iNum);

/**
 * @brief 加胡椒噪声
 * @param[in, out] iSrc 输入输出图像
 * @param iNum 胡椒噪声的数量
 */
 void AddPepperNoice(cv::Mat &iSrc, int iNum);

/**
 * @brief 加高斯噪声
 * @param[in, out] iSrc 输入输出图像
 * @param mean 高斯分布的均值
 * @param sigma 高斯分布的标准差
 */
 void AddGaussianNoice(cv::Mat &iSrc, double mean = 10, double sigma = 50);

/**
 * @brief 加均匀噪声
 * @param[in, out] iSrc 输入输出图像
 * @param low 均匀分布的下界
 * @param high 均匀分布的上界
 */
 void AddUniformNoice(cv::Mat &iSrc, double low = 10, double high = 50);

}; // namespace ImgNoice