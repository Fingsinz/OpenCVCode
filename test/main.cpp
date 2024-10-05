#include "../src/ImgTest.hpp"

int main(int argc, char const *argv[]) {
    ImgTest::GetSrc();
    ImgTest::TestGrayInversion();
    ImgTest::ShowResult();
    return 0;
}