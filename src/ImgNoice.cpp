#include "ImgNoice.hpp"
#include "opencv2/core.hpp"
#include <random>

void ImgNoice::AddSaltNoice(cv::Mat &iSrc, int iNum) {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> randomRow(0, iSrc.rows - 1);
    std::uniform_int_distribution<int> randomCol(0, iSrc.cols - 1);

    for (int k = 0; k < iNum; ++k) {
        int i = randomRow(generator);
        int j = randomCol(generator);
        if (iSrc.channels() == 3) {
            iSrc.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
        } else if (iSrc.channels() == 1) {
            iSrc.at<uchar>(i, j) = 255;
        }
    }
}

void ImgNoice::AddPepperNoice(cv::Mat &iSrc, int iNum) {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> randomRow(0, iSrc.rows - 1);
    std::uniform_int_distribution<int> randomCol(0, iSrc.cols - 1);

    for (int k = 0; k < iNum; ++k) {
        int i = randomRow(generator);
        int j = randomCol(generator);
        if (iSrc.channels() == 3) {
            iSrc.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
        } else if (iSrc.channels() == 1) {
            iSrc.at<uchar>(i, j) = 0;
        }
    }
}

void ImgNoice::AddGaussianNoice(cv::Mat &iSrc, double mean /*= 10*/, double sigma /*= 50*/) {
    cv::RNG rng;
    cv::Mat noice = iSrc.clone();
    rng.fill(noice, cv::RNG::NORMAL, mean, sigma);
    cv::add(iSrc, noice, iSrc);
}

void ImgNoice::AddUniformNoice(cv::Mat &iSrc, double low /*= 10*/, double high /*= 50*/) {
    cv::RNG rng;
    cv::Mat noice = iSrc.clone();
    rng.fill(noice, cv::RNG::UNIFORM, low, high);
    cv::add(iSrc, noice, iSrc);
}