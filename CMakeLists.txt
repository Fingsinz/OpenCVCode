cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(OpenCVPro CXX)

list(APPEND CMAKE_PREFIX_PATH "E:/OpenCV/opencv4.5.2/build/x64/vc15/lib")
find_package(OpenCV REQUIRED)

file(GLOB SRC src/*.cpp)
file(GLOB SRC_INC src/*.hpp)
file(GLOB TEST test/*.cpp)
file(GLOB TEST_INC test/*.hpp)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(main ${SRC_INC} ${SRC} ${TEST_INC} ${TEST})
target_link_libraries(main ${OpenCV_LIBS})