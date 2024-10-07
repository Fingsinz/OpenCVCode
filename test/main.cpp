#include "../src/ImgTest.hpp"

int main(int argc, char const *argv[]) {
    // 获取原图像
    ImgTest::GetSrc();

    // // 进行灰度变换测试
    // ImgTest::TestGrayInversion();
    // ImgTest::TestGrayLogTrans();
    // ImgTest::TestGrayGammaTrans();

    // // 进行直方图及其均衡化、匹配测试
    // ImgTest::TestGetHistogram();
    // ImgTest::TestHistEqualization();
    // ImgTest::TestHistMatch();

    // // 进行低通滤波器测试
    // ImgTest::TestMeanFilter();
    // ImgTest::TestGaussianFilter();
    // ImgTest::TestMedianFilter();
    
    // // 进行高通滤波器测试
    // ImgTest::TestLaplacianFilter();
    
    // // 进行噪声观察测试
    // ImgTest::TestNoice();

    // // 进行均值滤波器测试
    // ImgTest::TestArithMeanFilter();
    // ImgTest::TestGeoMeanFilter();
    // ImgTest::TestHarmMeanFilter();
    // ImgTest::TestAntiHarmeanFilter();

    // // 进行统计排序滤波器测试
    ImgTest::TestMedianFilter1();

    return 0;
}