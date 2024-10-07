﻿#include "ImgTest.hpp"
#include "ImgNoice.hpp"
#include "ImgProcessor.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

cv::Mat ImgTest::src;
cv::Mat ImgTest::dst;

ImgTest::~ImgTest() {
    src.release();
    dst.release();
    cv::destroyAllWindows();
}

void ImgTest::GetSrc(bool ibFlag) {
    src.release();
    if (ibFlag) {
        src = cv::imread("../../res/lena.jpg", cv::IMREAD_COLOR);
    } else {
        src = cv::imread("../../res/lena.jpg", cv::IMREAD_GRAYSCALE);
    }
}

void ImgTest::ShowResult() {
    cv::imshow("src", src);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestGrayInversion() {
    dst.release();
    ImgProcessor::GrayInversion(src, dst);

    ShowResult();
}

void ImgTest::TestGrayLogTrans() {
    dst.release();
    ImgProcessor::GrayLogTrans(src, dst, 5.0);

    ShowResult();
}

void ImgTest::TestGrayGammaTrans() {
    dst.release();
    ImgProcessor::GrayGammaTrans(src, dst, 3.0, 0.9);

    ShowResult();
}

void ImgTest::TestGetHistogram() {
    dst.release();
    ImgProcessor::GetHistogram(src, dst);

    ShowResult();
}

void ImgTest::TestHistEqualization() {
    dst.release();
    ImgProcessor::HistEqualization(src, dst);

    cv::Mat hist1, hist2;
    ImgProcessor::GetHistogram(src, hist1);
    ImgProcessor::GetHistogram(dst, hist2);
    cv::imshow("hist1", hist1);
    cv::imshow("hist2", hist2);

    ShowResult();
}

void ImgTest::TestHistMatch() {
    dst.release();
    cv::Mat pattern(src.size(), CV_8UC1, cv::Scalar(255));
    ImgProcessor::HistMatch(src, pattern, dst);
    cv::imshow("pattern", pattern);

    ShowResult();
}

void ImgTest::TestMeanFilter() {
    dst.release();
    cv::Mat saltImg = src.clone();

    ImgNoice::AddSaltNoice(saltImg, 5000);
    ImgProcessor::MeanFilter(saltImg, dst, 5);

    cv::Mat tmp;
    // OpenCV 内置函数
    cv::blur(saltImg, tmp, cv::Size(5, 5));
    cv::imshow("blur", tmp);

    ShowResult();
}

void ImgTest::TestGaussianFilter() {
    dst.release();
    cv::Mat saltImg = src.clone();

    ImgNoice::AddSaltNoice(saltImg, 5000);
    ImgProcessor::GaussianFilter(saltImg, dst, 5, 7.0);

    cv::Mat tmp;
    // OpenCV 内置函数
    cv::GaussianBlur(saltImg, tmp, cv::Size(5, 5), 7.0);
    cv::imshow("Gaussian blur", tmp);

    ShowResult();
}

void ImgTest::TestMedianFilter() {
    dst.release();
    cv::Mat saltImg = src.clone();

    ImgNoice::AddSaltNoice(saltImg, 5000);
    ImgProcessor::MedianFilter(saltImg, dst, 5);

    cv::Mat tmp;
    // OpenCV 内置函数
    cv::medianBlur(saltImg, tmp, 5);
    cv::imshow("median blur", tmp);

    ShowResult();
}

void ImgTest::TestLaplacianFilter() {
    cv::Mat tmp1, tmp2, tmp3;

    ImgProcessor::LaplacianFilter(src, tmp1, false);
    ImgProcessor::LaplacianFilter(src, tmp2, true);
    // OpenCV 内置函数
    cv::Laplacian(src, dst, CV_8UC3, 3, 1, 0, cv::BORDER_REFLECT);
    dst += src;

    cv::imshow("4-filter", tmp1);
    cv::imshow("8-filter", tmp2);

    ShowResult();
}

void ImgTest::TestNoice() {
    dst.release();
    src = cv::imread("../../res/test.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat pic1 = src.clone(), pic2 = src.clone(), pic3 = src.clone();

    ImgNoice::AddSaltNoice(pic1, 5000);
    ImgNoice::AddGaussianNoice(pic2, 10, 50);
    ImgNoice::AddUniformNoice(pic3, 10, 50);
    cv::rectangle(pic1, cv::Rect(900, 0, 100, 100), cv::Scalar(0, 0, 255), 5);
    cv::rectangle(pic2, cv::Rect(900, 0, 100, 100), cv::Scalar(0, 0, 255),5);
    cv::rectangle(pic3, cv::Rect(900, 0, 100, 100), cv::Scalar(0, 0, 255), 5);

    cv::namedWindow("椒盐噪声", cv::WINDOW_NORMAL);
    cv::namedWindow("高斯噪声", cv::WINDOW_NORMAL);
    cv::namedWindow("均匀噪声", cv::WINDOW_NORMAL);
    cv::imshow("椒盐噪声", pic1);
    cv::imshow("高斯噪声", pic2);
    cv::imshow("均匀噪声", pic3);

    cv::Mat hist1, hist2, hist3;
    ImgProcessor::GetHistogram(pic1(cv::Rect(900, 0, 100, 100)), hist1);
    ImgProcessor::GetHistogram(pic2(cv::Rect(900, 0, 100, 100)), hist2);
    ImgProcessor::GetHistogram(pic3(cv::Rect(900, 0, 100, 100)), hist3);
    cv::imshow("椒盐噪声直方图", hist1);
    cv::imshow("高斯噪声直方图", hist2);
    cv::imshow("均匀噪声直方图", hist3);

    cv::waitKey(0);
    cv::destroyAllWindows();
}