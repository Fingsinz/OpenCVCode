﻿#pragma once

#include "opencv2/core/mat.hpp"

class ImgTest {
public:
    ImgTest() = default;
    ~ImgTest();

public:
    static void GetSrc();
    static void ShowResult();

    // 灰度变换
    static void TestGrayInversion();
    static void TestGrayLogTrans();
    static void TestGrayGammaTrans();

    // 直方图
    static void TestGetHistogram();
    static void TestHistEqualization();
    static void TestHistMatch();

    // 低通空间滤波器
    static void TestMeanFilter();
    static void TestGaussianFilter();

private:
    static cv::Mat src;
    static cv::Mat dst;
};