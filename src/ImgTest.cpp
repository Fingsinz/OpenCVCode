#include "ImgTest.hpp"
#include "ImgProcessor.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"

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