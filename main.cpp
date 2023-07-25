#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>

using namespace std;


extern "C" bool BoxFilter_GPU_Wrapper(const cv::Mat & input, cv::Mat & output);
extern "C" bool boxFilter_CPU(const cv::Mat & input, cv::Mat & output);
extern "C" void LaplacianFilter_GPU_Wrapper(const cv::Mat & input, cv::Mat & output);
extern "C" void laplacianFilter_CPU(const cv::Mat & input, cv::Mat & output);
extern "C" bool medianFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output);
extern "C" bool medianFilter_CPU(const cv::Mat & input, cv::Mat & output);
extern "C" bool sharpeningFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output);
extern "C" bool sharpeningFilter_CPU(const cv::Mat & input, cv::Mat & output);
extern "C" bool sobelFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output);
extern "C" bool sobelFilter_CPU(const cv::Mat & input, cv::Mat & output);

static bool Original_Image = false;
static bool BoxFilter = false;
static bool LaplacianFilter = false;
static bool medianFilter = false;
static bool sharpeningFilter = false;
static bool sobelFilter = true;

// Program main
int main(int argc, char** argv) {
    // name of image
    string image_name = "cr7";

    // input & output file names
    string input_file = image_name + ".jpeg";
    string output_file_cpu = image_name + "_cpu.jpeg";
    string output_file_gpu = image_name + "_gpu.jpeg";

    
#if 1

    if (Original_Image == true)
    {
        // Read input image 
        cv::Mat srcImage = cv::imread(input_file, cv::IMREAD_UNCHANGED);
        if (srcImage.empty())
        {
            std::cout << "Image Not Found: " << input_file << std::endl;
            return -1;
        }
        cout << "\ninput image size: " << srcImage.cols << " " << srcImage.rows << " " << srcImage.channels() << "\n";
        cv::imshow("Original Image", srcImage);
        cv::waitKey(-1);
    }
    else 
    if (BoxFilter == true)
    {
        // Read input image 
        cv::Mat srcImage = cv::imread(input_file, cv::IMREAD_UNCHANGED);
       
        // Declare the output image  
        cv::Mat dstImage(srcImage.size(), srcImage.type());
        if (srcImage.empty())
        {
            std::cout << "Image Not Found: " << input_file << std::endl;
            return -1;
        }
        cout << "\ninput image size: " << srcImage.cols << " " << srcImage.rows << " " << srcImage.channels() << "\n";
        
        // run box filter on GPU  
        BoxFilter_GPU_Wrapper(srcImage, dstImage);
        cv::imshow("BoxFilter", dstImage);

        // run box filter on CPU  
        boxFilter_CPU(srcImage, dstImage);
        //cv::imshow("CPU", dstImage);
        cv::waitKey(-1);
    }
    else
    if(LaplacianFilter == true)
    {
        // Read input image 
        cv::Mat srcImage = cv::imread(input_file, cv::IMREAD_GRAYSCALE);

       
        if (srcImage.empty())
        {
            std::cout << "Image Not Found: " << input_file << std::endl;
            return -1;
        }
        cout << "\ninput image size: " << srcImage.cols << " " << srcImage.rows << " " << srcImage.channels() << "\n";

        // Declare the output image  
        cv::Mat dstImage(srcImage.size(), srcImage.type());

        // run box filter on GPU  
        LaplacianFilter_GPU_Wrapper(srcImage, dstImage);
        dstImage.convertTo(dstImage, CV_32F, 1.0 / 255, 0);
        dstImage *= 255;
       //cv::imshow("GPU", dstImage);
       //cv::waitKey(-1);

        // run box filter on CPU  
        laplacianFilter_CPU(srcImage, dstImage);
        dstImage.convertTo(dstImage, CV_32F, 1.0 / 255, 0);
        dstImage *= 255;
        cv::imshow("laplacianFilter", dstImage);
        cv::waitKey(-1);
    }
    else
    if(medianFilter == true)
    {
        // Read input image 
        cv::Mat srcImage = cv::imread(input_file, cv::IMREAD_UNCHANGED);


        if (srcImage.empty())
        {
            std::cout << "Image Not Found: " << input_file << std::endl;
            return -1;
        }
        cout << "\ninput image size: " << srcImage.cols << " " << srcImage.rows << " " << srcImage.channels() << "\n";

        // Declare the output image  
        cv::Mat dstImage(srcImage.size(), srcImage.type());

        // run box filter on GPU  
        medianFilter_GPU_wrapper(srcImage, dstImage);

         // run box filter on CPU  
        medianFilter_CPU(srcImage, dstImage);
        cv::imshow("medianFilter", dstImage);
        cv::waitKey(-1);
    }
    else 
    if (sharpeningFilter)
    {
        // Read input image 
        cv::Mat srcImage = cv::imread(input_file, cv::IMREAD_UNCHANGED);
        if (srcImage.empty())
        {
            std::cout << "Image Not Found: " << input_file << std::endl;
            return -1;
        }
        cout << "\ninput image size: " << srcImage.cols << " " << srcImage.rows << " " << srcImage.channels() << "\n";

        // Declare the output image  
        cv::Mat dstImage(srcImage.size(), srcImage.type());

        // run median filter on CPU  
        sharpeningFilter_CPU(srcImage, dstImage);
        

        // run median filter on GPU  
        sharpeningFilter_GPU_wrapper(srcImage, dstImage);

        cv::imshow("sharpeningFilter", dstImage);
        cv::waitKey(-1);
    }
    else if (sobelFilter == true)
    {
        // Read input image 
        cv::Mat srcImage = cv::imread(input_file, cv::IMREAD_UNCHANGED);
        if (srcImage.empty())
        {
            std::cout << "Image Not Found: " << input_file << std::endl;
            return -1;
        }
        cout << "\ninput image size: " << srcImage.cols << " " << srcImage.rows << " " << srcImage.channels() << "\n";

        // Declare the output image  
        cv::Mat dstImage(srcImage.size(), srcImage.type());

        // run sobel edge detection filter on GPU  
        sobelFilter_GPU_wrapper(srcImage, dstImage);
        // normalization to 0-255
        dstImage.convertTo(dstImage, CV_32F, 1.0 / 255, 0);
        dstImage *= 255;
        
        // run sobel edge detection filter on CPU  
        sobelFilter_CPU(srcImage, dstImage);
        
        // normalization to 0-255
        dstImage.convertTo(dstImage, CV_32F, 1.0 / 255, 0);
        dstImage *= 255;
        cv::imshow("sharpeningFilter", dstImage);
        cv::waitKey(-1);
    }
    
#else


    // Read input image
    cv::Mat srcImage = cv::imread(input_file, cv::IMREAD_GRAYSCALE);

    // Declare the output image  
    cv::Mat dstImage(srcImage.size(), srcImage.type());
    if (srcImage.empty())
    {
        std::cout << "Image Not Found: " << input_file << std::endl;
        return -1;
    }
    cout << "\ninput image size: " << srcImage.cols << " " << srcImage.rows << " " << srcImage.channels() << "\n";

    // convert RGB to gray scale
    //cv::cvtColor(srcImage, srcImage, cv::CV_BGR2GRAY);

    // Declare the output image  
    cv::Mat dstImage(srcImage.size(), srcImage.type());

    // run box filter on GPU  
   // BoxFilter_GPU_Wrapper(srcImage, dstImage);
    // Output image
    //imwrite(output_file_gpu, dstImage);
   LaplacianFilter_GPU_Wrapper(srcImage, dstImage);
    // run sobel edge detection filter on GPU  
    //sobelFilter_GPU_wrapper(srcImage, dstImage);
    dstImage.convertTo(dstImage, CV_32F, 1.0 / 255, 0);
    dstImage *= 255;
    // run median filter on GPU  
    //medianFilter_GPU_wrapper(srcImage, dstImage);
     // run median filter on GPU  
   // sharpeningFilter_GPU_wrapper(srcImage, dstImage);
    cv::imshow("GPU", dstImage);
    cv::waitKey(-1);


    // run box filter on CPU  
    //boxFilter_CPU(srcImage, dstImage);
    // Output image
    //imwrite(output_file_cpu, dstImage);
    laplacianFilter_CPU(srcImage, dstImage);
    // run sobel edge detection filter on CPU  
   // sobelFilter_CPU(srcImage, dstImage);
    dstImage.convertTo(dstImage, CV_32F, 1.0 / 255, 0);
    dstImage *= 255;
    //medianFilter_CPU(srcImage, dstImage);
    // run median filter on CPU  
    //sharpeningFilter_CPU(srcImage, dstImage);
    cv::imshow("CPU", dstImage);
    cv::waitKey(-1);
#endif
    return 0;
}
