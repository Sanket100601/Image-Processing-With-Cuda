#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include<time.h>
#include <iostream>
#include <string>
#include <stdio.h>

#define BLOCK_SIZE      16
#define FILTER_WIDTH    3       
#define FILTER_HEIGHT   3      

using namespace std;
using namespace cv;
//struct timeval t1;

// The wrapper is used to call boxFilter 
extern "C" void boxFilter_CPU(const cv::Mat & input, cv::Mat & output)
{

    int64 t0 = cv::getTickCount();
    /*for (int i = 1; i < 5; i = i + 2)
    {*/
        blur(input, output, Size(FILTER_WIDTH, FILTER_HEIGHT), Point(-1, -1));
    //}

    int64 t1 = cv::getTickCount();
    double secs = (t1 - t0) / cv::getTickFrequency();

    cout << "\nProcessing time for CPU (ms): " << secs * 1000 << "\n";
}

extern "C" void laplacianFilter_CPU(const cv::Mat & input, cv::Mat & output)
{
    cv::Mat input_gray;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;

    int64 t0 = cv::getTickCount();

    /// Remove noise by blurring with a Gaussian filter
    //GaussianBlur(input, input, Size(3, 3), 0, 0, BORDER_DEFAULT);

    // laplacian filter
    Laplacian(input, output, CV_16S, kernel_size, scale, delta, BORDER_DEFAULT);

    int64 t1 = cv::getTickCount();
    double secs = (t1 - t0) / cv::getTickFrequency();

    cout << "\nProcessing time for CPU (ms): " << secs * 1000 << "\n";
}

// The wrapper is use to call median filter 
extern "C" void medianFilter_CPU(const cv::Mat & input, cv::Mat & output)
{

    cv::Mat input_gray;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    int64 t0 = cv::getTickCount();

    for (int i = 1; i < 9; i = i + 2)
        medianBlur(input, output, i);


    int64 t1 = cv::getTickCount();
    double secs = (t1 - t0) / cv::getTickFrequency();

    cout << "\nProcessing time on CPU (ms): " << secs * 1000 << "\n";
}

// The wrapper is used to call sharpening filter 
extern "C" void sharpeningFilter_CPU(const cv::Mat & input, cv::Mat & output)
{
    Point anchor = Point(-1, -1);
    double delta = 0;
    int ddepth = -1;
    int kernel_size;

    int64 t0 = cv::getTickCount();

    /// Update kernel size for a normalized box filter
    kernel_size = 3;

    cv::Mat kernel = (Mat_<double>(kernel_size, kernel_size) << -1, -1, -1, -1, 3, -1, -1, -1, -1);

    // Apply 2D filter to image
    filter2D(input, output, ddepth, kernel, anchor, delta, BORDER_DEFAULT);

    int64 t1 = cv::getTickCount();
    double secs = (t1 - t0) / cv::getTickFrequency();

    cout << "\nProcessing time on CPU (ms): " << secs * 1000 << "\n";
}

// The wrapper is used to call sharpening filter 
extern "C" void sobelFilter_CPU(const cv::Mat & input, cv::Mat & output)
{
    Point anchor = Point(-1, -1);
    double delta = 0;
    int ddepth = -1;
    int kernel_size;

    int64 t0 = cv::getTickCount();

    /// Update kernel size for a normalized box filter
    kernel_size = 3;

    cv::Mat output1;
    cv::Mat kernel1 = (Mat_<double>(kernel_size, kernel_size) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    /// Apply 2D filter
    filter2D(input, output1, ddepth, kernel1, anchor, delta, BORDER_DEFAULT);


    cv::Mat output2;
    cv::Mat kernel2 = (Mat_<double>(kernel_size, kernel_size) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    /// Apply 2D filter
    filter2D(input, output2, ddepth, kernel2, anchor, delta, BORDER_DEFAULT);

    output = output1 + output2;

    output.convertTo(output, CV_32F, 1.0 / 255, 0);
    output *= 255;

    int64 t1 = cv::getTickCount();
    double secs = (t1 - t0) / cv::getTickFrequency();

    cout << "\nProcessing time on CPU (ms): " << secs * 1000 << "\n";
}
