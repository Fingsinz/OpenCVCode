#include "ImgTest.hpp"
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

void ImgTest::TestGrayInversion() {
    dst.release();
    ImgProcessor::GrayInversion(src, dst);

    cv::imshow("src", src);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestGrayLogTrans() {
    dst.release();
    ImgProcessor::GrayLogTrans(src, dst, 5.0);

    cv::imshow("src", src);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestGrayGammaTrans() {
    dst.release();
    ImgProcessor::GrayGammaTrans(src, dst, 3.0, 0.9);

    cv::imshow("src", src);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestGetHistogram() {
    dst.release();
    ImgProcessor::GetHistogram(src, dst);

    cv::imshow("src", src);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestHistEqualization() {
    dst.release();
    ImgProcessor::HistEqualization(src, dst);

    cv::Mat hist1, hist2;
    ImgProcessor::GetHistogram(src, hist1);
    ImgProcessor::GetHistogram(dst, hist2);
    cv::imshow("hist1", hist1);
    cv::imshow("hist2", hist2);

    cv::imshow("src", src);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestHistMatch() {
    dst.release();
    cv::Mat pattern(src.size(), CV_8UC1, cv::Scalar(255));
    ImgProcessor::HistMatch(src, pattern, dst);
    cv::imshow("pattern", pattern);

    cv::imshow("src", src);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
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

    cv::imshow("src", saltImg);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
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

    cv::imshow("src", saltImg);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
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

    cv::imshow("src", saltImg);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
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

    cv::imshow("src", src);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestNoice() {
    dst.release();
    src.release();
    src = cv::imread("../../res/test.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat pic1 = src.clone(), pic2 = src.clone(), pic3 = src.clone();

    ImgNoice::AddSaltNoice(pic1, 5000);
    ImgNoice::AddGaussianNoice(pic2, 10, 50);
    ImgNoice::AddUniformNoice(pic3, 10, 50);
    cv::rectangle(pic1, cv::Rect(900, 0, 100, 100), cv::Scalar(0, 0, 255), 5);
    cv::rectangle(pic2, cv::Rect(900, 0, 100, 100), cv::Scalar(0, 0, 255), 5);
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
    src.release();
    GetSrc();
}

void ImgTest::TestArithMeanFilter() {
    dst.release();
    cv::Mat saltImg = src.clone();

    ImgNoice::AddSaltNoice(saltImg, 5000);
    ImgProcessor::ArithmeticMeanFilter(saltImg, dst, cv::Size(5, 5));

    cv::Mat tmp;
    // OpenCV 内置函数
    cv::blur(saltImg, tmp, cv::Size(5, 5));
    cv::imshow("blur", tmp);

    cv::imshow("src", saltImg);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestGeoMeanFilter() {
    dst.release();
    cv::Mat saltImg = src.clone();

    ImgNoice::AddSaltNoice(saltImg, 5000);
    ImgProcessor::GeometricMeanFilter(saltImg, dst, cv::Size(5, 5));

    cv::Mat tmp;
    // OpenCV 内置函数
    cv::blur(saltImg, tmp, cv::Size(5, 5));
    cv::imshow("blur", tmp);

    cv::imshow("src", saltImg);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestHarmMeanFilter() {
    dst.release();
    cv::Mat saltImg = src.clone();

    ImgNoice::AddSaltNoice(saltImg, 5000);
    ImgProcessor::HarmonicMeanFilter(saltImg, dst, cv::Size(5, 5));

    cv::Mat tmp;
    // OpenCV 内置函数
    cv::blur(saltImg, tmp, cv::Size(5, 5));
    cv::imshow("blur", tmp);

    cv::imshow("src", saltImg);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestAntiHarmeanFilter() {
    dst.release();
    cv::Mat saltImg = src.clone();

    ImgNoice::AddSaltNoice(saltImg, 5000);
    ImgProcessor::AntiHarmonicMeanFilter(saltImg, dst, cv::Size(5, 5), -1.5);

    cv::Mat tmp;
    // OpenCV 内置函数
    cv::blur(saltImg, tmp, cv::Size(5, 5));
    cv::imshow("blur", tmp);

    cv::imshow("src", saltImg);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestMedianFilter1() {
    dst.release();
    cv::Mat saltImg = src.clone();

    ImgNoice::AddSaltNoice(saltImg, 5000);
    ImgProcessor::MedianFilter(saltImg, dst, cv::Size(5, 5));

    cv::Mat tmp;
    // OpenCV 内置函数
    cv::blur(saltImg, tmp, cv::Size(5, 5));
    cv::imshow("blur", tmp);

    cv::imshow("src", saltImg);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestMinMaxFilter() {
    dst.release();
    cv::Mat saltImg = src.clone();
    cv::Mat pepperImg = src.clone();

    ImgNoice::AddSaltNoice(saltImg, 5000);
    ImgNoice::AddPepperNoice(pepperImg, 5000);

    cv::Mat out1, out2;
    ImgProcessor::MinMaxFilter(pepperImg, out1, cv::Size(5, 5), true);
    ImgProcessor::MinMaxFilter(saltImg, out2, cv::Size(5, 5), false);

    cv::imshow("带盐噪声", saltImg);
    cv::imshow("带胡椒噪声", pepperImg);
    cv::imshow("去胡椒噪声", out1);
    cv::imshow("去盐噪声", out2);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestMidPointFilter() {
    dst.release();
    cv::Mat saltImg = src.clone();

    ImgNoice::AddUniformNoice(saltImg);
    ImgProcessor::MidPointFilter(saltImg, dst, cv::Size(3, 3));

    cv::Mat tmp;
    // OpenCV 内置函数
    cv::blur(saltImg, tmp, cv::Size(3, 3));
    cv::imshow("blur", tmp);

    cv::imshow("src", saltImg);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestModifiedAlphaMeanFilter() {
    dst.release();
    cv::Mat saltImg = src.clone();

    ImgNoice::AddSaltNoice(saltImg, 5000);
    ImgProcessor::ModifiedAlphaMeanFilter(saltImg, dst, cv::Size(5, 5), 3);

    cv::Mat tmp;
    // OpenCV 内置函数
    cv::blur(saltImg, tmp, cv::Size(3, 3));
    cv::imshow("blur", tmp);

    cv::imshow("src", saltImg);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestAdaptiveLocalFilter() {
    dst.release();
    cv::Mat saltImg = src.clone();

    ImgNoice::AddGaussianNoice(saltImg);
    ImgProcessor::AdaptiveLocalFilter(saltImg, dst, cv::Size(5, 5));

    cv::Mat tmp;
    // OpenCV 内置函数
    cv::blur(saltImg, tmp, cv::Size(3, 3));
    cv::imshow("blur", tmp);

    cv::imshow("src", saltImg);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestAdaptiveMedianFilter() {
    dst.release();
    cv::Mat saltImg = src.clone();

    ImgNoice::AddSaltNoice(saltImg, 5000);
    ImgProcessor::AdaptiveMedianFilter(saltImg, dst, 5);

    cv::Mat tmp;
    // OpenCV 内置函数
    cv::blur(saltImg, tmp, cv::Size(3, 3));
    cv::imshow("blur", tmp);

    cv::imshow("src", saltImg);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestBGRHLS() {
    dst.release();
    cv::Mat cvtSrc = src.clone(), cvtDst, mBGR2HLS, mHLS2BGR;

    ImgProcessor::BGR2HSL(src, dst);
    ImgProcessor::HSL2BGR(dst, mHLS2BGR);

    cv::cvtColor(cvtSrc, cvtDst, cv::COLOR_BGR2HLS);
    cv::cvtColor(cvtDst, mHLS2BGR, cv::COLOR_HLS2BGR);

    cv::imshow("原图", src);
    cv::imshow("自实现BGR2HLS", dst);
    cv::imshow("自实现HSL2BGR", mHLS2BGR);
    cv::imshow("cv函数BGR2HLS", cvtDst);
    cv::imshow("cv函数HSL2BGR", mHLS2BGR);

    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestErode() {
    dst.release();

    cv::imshow("原图", src);
    if (src.channels() == 3) {
        cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
    }
    cv::threshold(dst, dst, 100, 255, cv::THRESH_BINARY);
    cv::imshow("二值化", dst);
    ImgProcessor::Erode(src, dst, cv::Size(3, 3), 1, false);
    cv::imshow("腐蚀", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestDilate() {
    dst.release();

    cv::imshow("原图", src);
    if (src.channels() == 3) {
        cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
    }
    cv::threshold(dst, dst, 100, 255, cv::THRESH_BINARY);
    cv::imshow("二值化", dst);
    ImgProcessor::Dilate(src, dst, cv::Size(3, 3), 1, false);
    cv::imshow("膨胀", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestOpen() {
    dst.release();
    src = cv::imread("../../res/pic2.png", cv::IMREAD_GRAYSCALE);
    cv::imshow("原图", src);
    cv::threshold(src, dst, 100, 255, cv::THRESH_BINARY);
    cv::imshow("二值化", dst);
    ImgProcessor::OpenOperation(src, dst, cv::Size(3, 3), 1, false);
    cv::imshow("开操作", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void ImgTest::TestClose() {
    dst.release();
    src = cv::imread("../../res/pic2.png", cv::IMREAD_GRAYSCALE);
    cv::imshow("原图", src);
    cv::threshold(src, dst, 100, 255, cv::THRESH_BINARY);
    cv::imshow("二值化", dst);
    ImgProcessor::CloseOperation(src, dst, cv::Size(3, 3), 1, false);
    cv::imshow("闭操作", dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
}