#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include<device_functions.h>
#include<cuda_runtime_api.h>
#include "device_launch_parameters.h"

#define BLOCK_SIZE    16
#define FILTER_WIDTH  5
#define FILTER_HEIGHT 5

using namespace std;

// Run BOX FILTER on GPU
__global__ void boxFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height, int channel)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Only Threads inside Image Will Write The Results 
	if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
	{
		for (int c = 0; c < channel; c++)
		{
			// Sum of pixel values
			float sum = 0;
			
			// number of filter pixels
			float Ks = 0;

			// Loop inside the filter pixel pixel to average the pixel values 
			for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++) 
			{
				for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++) 
				{
					float fl = srcImage[((y + ky) * width + (x + kx)) * channel + c];
					sum += fl;
					Ks += 1;
				}
			}
			dstImage[(y * width + x) * channel + c] = sum / Ks;
		}
	}
}

__global__ void laplacianFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float kernel[3][3] = { 0, -1, 0, -1, 4, -1, 0, -1, 0 };
	//float kernel[3][3] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};   
	// only threads inside image will write results
	if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
	{
		// Sum of pixel values 
		float sum = 0;
		// Loop inside the filter to average pixel values
		for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++) {
			for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++) {
				float fl = srcImage[((y + ky) * width + (x + kx))];
				sum += fl * kernel[ky + FILTER_HEIGHT / 2][kx + FILTER_WIDTH / 2];
			}
		}
		dstImage[(y * width + x)] = sum;
	}
}

// Sort the function On Device 
__device__ void sort(unsigned char* filterVector)
{
	for (int i = 0; i < FILTER_WIDTH * FILTER_HEIGHT; i++)
	{
		for (int j = i + 1; j < FILTER_WIDTH * FILTER_HEIGHT; j++)
		{
			if (filterVector[i] > filterVector[j])
			{
				// Swap the Variables
				unsigned char temp = filterVector[i];
				filterVector[i] = filterVector[j];
				filterVector[j] = temp;
			}
		}
	}
}

// Run Median Filter on GPU
__global__ void medianFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height, int channel)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// only threads inside image will write results
	if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
	{
		for (int c = 0; c < channel; c++)
		{
			unsigned char filterVector[FILTER_WIDTH * FILTER_HEIGHT];
			// Loop inside the filter to average pixel values
			for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++) {
				for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++) {
					filterVector[ky * FILTER_WIDTH + kx] = srcImage[((y + ky) * width + (x + kx)) * channel + c];
				}
			}
			// Sorting values of filter   
			sort(filterVector);
			dstImage[(y * width + x) * channel + c] = filterVector[(FILTER_WIDTH * FILTER_HEIGHT) / 2];
		}
	}
}

__global__ void sharpeningFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height, int channel)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float kernel[FILTER_WIDTH][FILTER_HEIGHT] = { -1, -1, -1, -1, 9, -1, -1, -1, -1 };
	// only threads inside image will write results
	if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
	{
		for (int c = 0; c < channel; c++)
		{
			// Sum of pixel values 
			float sum = 0;
			// Loop inside the filter to average pixel values
			for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++) {
				for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++) {
					float fl = srcImage[((y + ky) * width + (x + kx)) * channel + c];
					sum += fl * kernel[ky + FILTER_HEIGHT / 2][kx + FILTER_WIDTH / 2];
				}
			}
			dstImage[(y * width + x) * channel + c] = sum;
		}
	}
}

__global__ void sobelFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float Kx[3][3] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	float Ky[3][3] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

	// only threads inside image will write results
	if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
	{
		// Gradient in x-direction 
		float Gx = 0;
		// Loop inside the filter to average pixel values
		for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++) {
			for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++) {
				float fl = srcImage[((y + ky) * width + (x + kx))];
				Gx += fl * Kx[ky + FILTER_HEIGHT / 2][kx + FILTER_WIDTH / 2];
			}
		}
		float Gx_abs = Gx < 0 ? -Gx : Gx;

		// Gradient in y-direction 
		float Gy = 0;
		// Loop inside the filter to average pixel values
		for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++) {
			for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++) {
				float fl = srcImage[((y + ky) * width + (x + kx))];
				Gy += fl * Ky[ky + FILTER_HEIGHT / 2][kx + FILTER_WIDTH / 2];
			}
		}
		float Gy_abs = Gy < 0 ? -Gy : Gy;

		dstImage[(y * width + x)] = Gx_abs + Gy_abs;
	}
}

// the wrapper to Write the BOX FILTER
extern "C" void BoxFilter_GPU_Wrapper(const cv::Mat & input, cv::Mat & output)
{
	// Use Cuda Event To catch the time 
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// calcukate number of input channels
	int channel = input.step / input.cols;

	// calculate nuber of input and output bytes in each block
	const int inputSize = input.cols * input.rows * channel;
	const int outputSize = output.cols * output.rows * channel;

	unsigned char* d_input; 
	unsigned char* d_output;

	// Allocate device memory
	cudaMalloc((void**)&d_input, inputSize);
	cudaMalloc((void**)&d_output, outputSize);

	// Copy data From opencv input image to Device memory
	cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);

	// Specify the Block Size
	const dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);

	// Calculate the  Grid size to Calculate The whole image
	const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);

	// Start Time 
	cudaEventRecord(start);

	// Launch BOX FILTER kernel on Cuda
	boxFilter << <grid, block >> > (d_input, d_output, output.cols, output.rows, channel);
	
	// stop timer 
	cudaEventRecord(stop);

	// copy data from Device memory to output image
	cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

	//Free the Device Memory
	cudaFree(d_input);
	cudaFree(d_output);

	cudaEventSynchronize(stop);
	float milliseconds = 0;

	// Calculate  Elapsed Time in millisecond
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "\nProcessing time for GPU (ms): " << milliseconds << "\n";
}

// the wrapper to Write the BOX FILTER
extern "C" void LaplacianFilter_GPU_Wrapper(const cv::Mat & input, cv::Mat & output)
{
	// Use cuda event to catch time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Calculate number of input & output bytes in each block
	const int inputSize = input.cols * input.rows;
	const int outputSize = output.cols * output.rows;
	unsigned char* d_input, * d_output;

	// Allocate device memory
	cudaMalloc((void**)&d_input, inputSize);
	cudaMalloc((void**)&d_output, outputSize);

	// Copy data from OpenCV input image to device memory
	cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);

	// Specify block size
	const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

	// Calculate grid size to cover the whole image
	const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);

	// Start time
	cudaEventRecord(start);

	// Run BoxFilter kernel on CUDA 
	laplacianFilter << <grid, block >> > (d_input, d_output, output.cols, output.rows);

	// Stop time
	cudaEventRecord(stop);

	//Copy data from device memory to output image
	cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

	//Free the device memory
	cudaFree(d_input);
	cudaFree(d_output);

	cudaEventSynchronize(stop);
	float milliseconds = 0;

	// Calculate elapsed time in milisecond  
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "\nProcessing time for GPU (ms): " << milliseconds << "\n";
}

// The wrapper to call median filter 
extern "C" void medianFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output)
{
	// Use cuda event to catch time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Calculate number of image channels
	int channel = input.step / input.cols;

	// Calculate number of input & output bytes in each block
	const int inputSize = input.cols * input.rows * channel;
	const int outputSize = output.cols * output.rows * channel;
	unsigned char* d_input, * d_output;

	// Allocate device memory
	cudaMalloc((void**) & d_input, inputSize);
	cudaMalloc((void**)&d_output, outputSize);

	// Copy data from OpenCV input image to device memory
	cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);

	// Specify block size
	const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

	// Calculate grid size to cover the whole image
	const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);

	// Start time
	cudaEventRecord(start);

	// Run BoxFilter kernel on CUDA 
	medianFilter << <grid, block >> > (d_input, d_output, output.cols, output.rows, channel);

	// Stop time
	cudaEventRecord(stop);

	//Copy data from device memory to output image
	cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

	//Free the device memory
	cudaFree(d_input);
	cudaFree(d_output);

	cudaEventSynchronize(stop);
	float milliseconds = 0;

	// Calculate elapsed time in milisecond  
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "\nProcessing time on GPU (ms): " << milliseconds << "\n";
}

// The wrapper is used to call sharpening filter 
extern "C" void sharpeningFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output)
{
	// Use cuda event to catch time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Calculate number of image channels
	int channel = input.step / input.cols;

	// Calculate number of input & output bytes in each block
	const int inputSize = input.cols * input.rows * channel;
	const int outputSize = output.cols * output.rows * channel;
	unsigned char* d_input, * d_output;

	// Allocate device memory
	cudaMalloc((void**)&d_input, inputSize);
	cudaMalloc((void**) & d_output, outputSize);

	// Copy data from OpenCV input image to device memory
	cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);

	// Specify block size
	const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

	// Calculate grid size to cover the whole image
	const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);

	// Start time
	cudaEventRecord(start);

	// Run BoxFilter kernel on CUDA 
	sharpeningFilter << <grid, block >> > (d_input, d_output, output.cols, output.rows, channel);

	// Stop time
	cudaEventRecord(stop);

	//Copy data from device memory to output image
	cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

	//Free the device memory
	cudaFree(d_input);
	cudaFree(d_output);

	cudaEventSynchronize(stop);
	float milliseconds = 0;

	// Calculate elapsed time in milisecond  
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "\nProcessing time on GPU (ms): " << milliseconds << "\n";
}

// The wrapper is use to call sobel edge detection filter 
extern "C" void sobelFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output)
{
	// Use cuda event to catch time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Calculate number of input & output bytes in each block
	const int inputSize = input.cols * input.rows;
	const int outputSize = output.cols * output.rows;
	unsigned char* d_input, * d_output;

	// Allocate device memory
	cudaMalloc((void**)& d_input, inputSize);
	cudaMalloc((void**)&d_output, outputSize);

	// Copy data from OpenCV input image to device memory
	cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);

	// Specify block size
	const dim3 block(BLOCK_SIZE, BLOCK_SIZE);

	// Calculate grid size to cover the whole image
	const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);

	// Start time
	cudaEventRecord(start);

	// Run Sobel Edge Detection Filter kernel on CUDA 
	sobelFilter << <grid, block >> > (d_input, d_output, output.cols, output.rows);

	// Stop time
	cudaEventRecord(stop);

	//Copy data from device memory to output image
	cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

	//Free the device memory
	cudaFree(d_input);
	cudaFree(d_output);

	cudaEventSynchronize(stop);
	float milliseconds = 0;

	// Calculate elapsed time in milisecond  
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "\nProcessing time on GPU (ms): " << milliseconds << "\n";
}