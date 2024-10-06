#pragma once

#include "opencv2/core/mat.hpp"

class ImgTest {
public:
    ImgTest() = default;
    ~ImgTest();

public:
    static void GetSrc();
    static void ShowResult();

    static void TestGrayInversion();
    static void TestGrayLogTrans();

private:
    static cv::Mat src;
    static cv::Mat dst;
};