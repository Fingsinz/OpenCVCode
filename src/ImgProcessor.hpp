#pragma once

#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"

namespace ImgProcessor {
/**
 * @brief 灰度反转，将图像亮暗对调，可以增强图像中的暗色区域细节。
 * @details y = 255 - x
 * @param[in] src 输入图像
 * @param[out] dst 输出图像
 */
void GrayInversion(cv::Mat const &src, cv::Mat &dst);

/**
 * @brief 灰度对数变换，扩展图像中的暗像素值，压缩高灰度值
 * @details y = c * ln(1 + x)
 * @param[in] src 输入图像
 * @param[out] dst 输出图像
 * @param[in] c 对数变换因子
 */
void GrayLogTrans(cv::Mat const &src, cv::Mat &dst, double c = 1.0);

/**
 * @brief 灰度 gamma 变换，扩展图像中的暗像素值，压缩高灰度值
 * @details y = c * x ^ gamma
 * @param[in] src 输入图像
 * @param[out] dst 输出图像
 * @param[in] c 因子
 * @param[in] gamma gamma 值
 */
void GrayGammaTrans(cv::Mat const &src, cv::Mat &dst, double c = 1.0, double gamma = 1.0);

/**
 * @brief 获取src的直方图
 * @details
 * @param[in] src 输入图像
 * @param[out] dst 直方图
 */
void GetHistogram(cv::Mat const &src, cv::Mat &dst);

/**
 * @brief 直方图均衡化
 * @details
 * @param[in] src 输入图像
 * @param[out] dst均衡化后的图像
 */
void HistEqualization(cv::Mat const &src, cv::Mat &dst);

/**
 * @brief 直方图匹配
 * @details
 * @param[in] src 输入图像
 * @param[in] pattern 模板图像
 * @param[out] dst 匹配后的图像
 */
void HistMatch(cv::Mat const &src, cv::Mat const &pattern, cv::Mat &dst);

/**
 * @brief 均值滤波
 * @details
 * @param[in] iSrc 输入图像
 * @param[out] oDst 输出图像
 * @param[in] iFilterSize 滤波器的 size
 */
void MeanFilter(cv::Mat const &iSrc, cv::Mat &oDst, int iFilterSize);

/**
 * @brief 高斯滤波
 * @details
 * @param[in] iSrc 输入图像
 * @param[out] oDst 输出图像
 * @param[in] iFilterSize 滤波器的 size
 * @param[in] iSigmaX 高斯分布X的标准差
 * @param[in] iSigmaY 高斯分布Y的标准差，不使用
 */
void GaussianFilter(cv::Mat const &iSrc, cv::Mat &oDst, int iFilterSize, double iSigmaX, double iSigmaY = 0.0);

/**
 * @brief 中值滤波
 * @details
 * @param[in] iSrc 输入图像
 * @param[out] oDst 输出图像
 * @param[in] iFilterSize 滤波器的 size
 */
void MedianFilter(cv::Mat const &iSrc, cv::Mat &oDst, int iFilterSize);

/**
 * @brief Laplacian滤波
 * @details
 * @param[in] iSrc 输入图像
 * @param[out] oDst 输出图像
 * @param[in] iFilterSize 滤波器的 size
 * @param[in] iAll 是否八邻域拉普拉斯核
 */
void LaplacianFilter(cv::Mat const &iSrc, cv::Mat &oDst, bool ibAll = false);

/**
 * @brief 算术均值滤波
 * @details
 * @param[in] iSrc 输入图像
 * @param[out] oDst 输出图像
 * @param[in] iFilterSize 滤波器的 size
 */
void ArithmeticMeanFilter(cv::Mat const &iSrc, cv::Mat &oDst, cv::Size iFilterSize);

/**
 * @brief 几何均值滤波
 * @details
 * @param[in] iSrc 输入图像
 * @param[out] oDst 输出图像
 * @param[in] iFilterSize 滤波器的 size
 */
void GeometricMeanFilter(cv::Mat const &iSrc, cv::Mat &oDst, cv::Size iFilterSize);

/**
 * @brief 谐波均值滤波
 * @details
 * @param[in] iSrc 输入图像
 * @param[out] oDst 输出图像
 * @param[in] iFilterSize 滤波器的 size
 */
void HarmonicMeanFilter(cv::Mat const &iSrc, cv::Mat &oDst, cv::Size iFilterSize);

/**
 * @brief 反谐波均值滤波
 * @details
 * @param[in] iSrc 输入图像
 * @param[out] oDst 输出图像
 * @param[in] iFilterSize 滤波器的 size
 * @param[in] q q 值，用于调整权重
 */
void AntiHarmonicMeanFilter(cv::Mat const &iSrc, cv::Mat &oDst, cv::Size iFilterSize, double q = 0);

/**
 * @brief 中值滤波
 * @details
 * @param[in] iSrc 输入图像
 * @param[out] oDst 输出图像
 * @param[in] iFilterSize 滤波器的 size
 */
void MedianFilter(cv::Mat const &iSrc, cv::Mat &oDst, cv::Size iFilterSize);

/**
 * @brief 最大/最小值滤波
 * @details
 * @param[in] iSrc 输入图像
 * @param[out] oDst 输出图像
 * @param[in] iFilterSize 滤波器的 size
 * @param[in] ibMax 是否求最大值
 */
void MinMaxFilter(cv::Mat const &iSrc, cv::Mat &oDst, cv::Size iFilterSize, bool ibMax = true);

/**
 * @brief 中点滤波
 * @details
 * @param[in] iSrc 输入图像
 * @param[out] oDst 输出图像
 * @param[in] iFilterSize 滤波器的 size
 */
void MidPointFilter(cv::Mat const &iSrc, cv::Mat &oDst, cv::Size iFilterSize);

/**
 * @brief 修正阿尔法均值滤波
 * @details
 * @param[in] iSrc 输入图像
 * @param[out] oDst 输出图像
 * @param[in] iFilterSize 滤波器的 size
 * @param[in] idD 邻域 S_{xy} 内删除 g(r,c) 的 d 个最低灰度值和 d 个最高灰度值
 */
void ModifiedAlphaMeanFilter(cv::Mat const &iSrc, cv::Mat &oDst, cv::Size iFilterSize, int idD);

/**
 * @brief 自适应局部降噪滤波
 * @details
 * @param[in] iSrc 输入图像
 * @param[out] oDst 输出图像
 * @param[in] iFilterSize 滤波器的 size
 */
void AdaptiveLocalFilter(cv::Mat const &iSrc, cv::Mat &oDst, cv::Size iFilterSize);

/**
 * @brief 自适应中值滤波
 * @details
 * @param[in] iSrc 输入图像
 * @param[out] oDst 输出图像
 * @param[in] iMaxSize 最大尺寸
 */
void AdaptiveMedianFilter(cv::Mat const &iSrc, cv::Mat &oDst, int iMaxSize);

/**
 * @brief BGR 转换为 HSL
 * @details
 * @param[in] iSrc 输入图像
 * @param[out] oDst 输出图像
 */
void BGR2HSL(cv::Mat const &iSrc, cv::Mat &oDst);

/**
 * @brief HSL 转换为 BGR
 * @details
 * @param[in] iSrc 输入图像
 * @param[out] oDst 输出图像
 */
void HSL2BGR(cv::Mat const &iSrc, cv::Mat &oDst);

/**
 * @brief 膨胀
 * @details
 * @param[in] iSrc 输入图像
 * @param[out] oDst 输出图像
 * @param[in] iFilterSize 滤波器的 size
 * @param[in] iNums 膨胀的次数
 * @param[in] ib3Ch 是否三通道
 */
void Erode(cv::Mat const &iSrc, cv::Mat &oDst, cv::Size iFilterSize, int iNums = 1, bool ib3Ch = false);

}; // namespace ImgProcessor
