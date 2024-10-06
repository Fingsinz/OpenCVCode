#include "ImgTest.hpp"
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

void ImgTest::GetSrc() {
    src.release();
    src = cv::imread("../../res/lena.jpg");
}

void ImgTest::ShowResult() {
    cv::imshow("src", src);
    cv::imshow("dst", dst);
    cv::waitKey(0);
}

void ImgTest::TestGrayInversion() {
    dst.release();
    ImgProcessor::GrayInversion(src, dst);
}

void ImgTest::TestGrayLogTrans() {
    dst.release();
    ImgProcessor::GrayLogTrans(src, dst, 5.0);
}

void ImgTest::TestGrayGammaTrans() {
    dst.release();
    ImgProcessor::GrayGammaTrans(src, dst, 3.0, 0.9);
}

void ImgTest::TestGetHistogram() {
    dst.release();
    ImgProcessor::GetHistogram(src, dst);
}

void ImgTest::TestHistEqualization() {
    dst.release();
    ImgProcessor::HistEqualization(src, dst);

    cv::Mat hist1, hist2;
    ImgProcessor::GetHistogram(src, hist1);
    ImgProcessor::GetHistogram(dst, hist2);
    cv::imshow("hist1", hist1);
    cv::imshow("hist2", hist2);
}

void ImgTest::TestHistMatch() {
    dst.release();
    cv::Mat pattern(src.size(), CV_8UC1, cv::Scalar(255));
    ImgProcessor::HistMatch(src, pattern, dst);
    cv::imshow("pattern", pattern);
}

void ImgTest::TestMeanFilter() {
    dst.release();
    ImgProcessor::AddSaltNoice(src, 5000);
    
    ImgProcessor::MeanFilter(src, dst, 5);
    
    cv::Mat tmp;
    // OpenCV 内置函数
    cv::blur(src, tmp, cv::Size(5, 5));
    cv::imshow("blur", tmp);
}