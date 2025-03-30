// Erick Sun
// CSEN 145 Fa 2024
// This program takes a directory of images and performs convolution on all
// of the applicable images in the directory
#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cuda_runtime.h>

namespace fs = std::filesystem;
using namespace std;
using namespace std::chrono;

const int BATCH_SIZE = 512;
const int KERNEL_SIZE = 3;
const float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {1, 1, 1},
    {1, -8, 1},
    {1, 1, 1}
};
const float kernal_1D[KERNEL_SIZE * KERNEL_SIZE] = {1, 1, 1, 1, -8, 1, 1, 1, 1};

// Utility function to check for CUDA errors
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA kernel for 2D matrix convolution
__global__ void convolution2DKernel(const uint8_t* input, uint8_t* output, int* height_width, size_t* size_ptr, float* kernel) {
    // How large the convolution kernel is in one direction
    int offset = KERNEL_SIZE / 2;
    // The z ID corresponds to the img ID
    int img_index = blockIdx.z;
    size_t pixel_offset = size_ptr[img_index];
    int height = height_width[img_index * 2];
    int width = height_width[img_index * 2 + 1];

    // Find IDs
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int local_idx = idx - pixel_offset;
    int local_idy = idy - pixel_offset;

    // Load kernel into shared memory
    __shared__ float shared_kernel[KERNEL_SIZE * KERNEL_SIZE];
    if (threadIdx.x < KERNEL_SIZE && threadIdx.y < KERNEL_SIZE) {
        shared_kernel[threadIdx.y * KERNEL_SIZE + threadIdx.x] = kernel[threadIdx.y * KERNEL_SIZE + threadIdx.x];
    }
    __syncthreads();

    // Do convolution
    if(local_idx >= offset && local_idx <= width - offset && 
       local_idy >= offset && local_idy <= height - offset) {
        float sum = 0.0f;
        for(int i = -offset; i < offset; ++i) {
            for(int j = -offset; j < offset; ++j) {
                int index = (local_idy + i) * width + (local_idx + j) + pixel_offset;
                sum += input[index] * shared_kernel[(i + offset) * KERNEL_SIZE + j + offset];
            }
        }
        output[local_idy * width + local_idx + pixel_offset] = min(max(int(sum), 0), 255);
    }
}

int main(int argc, char* argv[]) {
    // Start timing the total execution time
    auto total_start = high_resolution_clock::now();
    auto total_start_2 = total_start;
    int totalcount = 0;

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_directory>" << endl;
        return -1;
    }

    fs::path input_dir = argv[1];

    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        cerr << "Error: Input directory does not exist or is not a directory." << endl;
        return -1;
    }

    vector<string> image_files;
    long process_time = 0;

    // Get list of image files in the input directory
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            image_files.push_back(entry.path().string());
        }
    }

    if (image_files.empty()) {
        cerr << "Error: No PNG image files found in the input directory." << endl;
        return -1;
    }

    // Assuming all images have the same dimensions, load the first image to get the dimensions
    int width, height, channels;
    uint8_t* img_data = stbi_load(image_files[0].c_str(), &width, &height, &channels, 1);
    if (!img_data) {
        cerr << "Error: Unable to open or read the image file: " << image_files[0] << endl;
        return -1;
    }
    delete[] img_data;
    int max_height;
    int max_width;
    int total_len;
    
    // Batch images
    for(int batch_start = 0; batch_start < image_files.size(); batch_start += BATCH_SIZE) {
        int sub_batch_size = std::min(BATCH_SIZE, (int) image_files.size() - batch_start);
        max_width = width;
        max_height = height;
        total_len = 0;

        // Image related arrays
        uint8_t** img_data_array = new uint8_t*[sub_batch_size];
        int* img_height_width_array = new int[sub_batch_size * 2];
        size_t* img_size_ptr_array = new size_t[sub_batch_size + 1];
        
        int channels;
        img_size_ptr_array[0] = 0;

        // Load images in batches
        for(int i = 0; i < sub_batch_size; ++i) {
            img_data_array[i] = stbi_load(image_files[batch_start + i].c_str(), &img_height_width_array[i * 2 + 1], &img_height_width_array[i * 2], &channels, 1);
            if (!img_data_array[i]) {
                cerr << "Error: Unable to open or read the image file: " << image_files[batch_start + i] << endl;
                return -1;
            }
            max_height = max(max_height, img_height_width_array[i * 2]);
            max_width = max(max_width, img_height_width_array[i * 2 + 1]);
            total_len += img_height_width_array[i * 2] * img_height_width_array[i * 2 + 1];
            img_size_ptr_array[i + 1] = img_size_ptr_array[i] + img_height_width_array[i * 2] * img_height_width_array[i * 2 + 1];
        }
        
        uint8_t* output = new uint8_t[img_size_ptr_array[sub_batch_size]];

        // REMOVE TEST LATER
        assert(img_size_ptr_array[sub_batch_size] == total_len);

        // Allocate memory on GPU
        uint8_t* d_input;
        uint8_t* d_output;
        int* d_img_height_width;
        size_t* d_size_ptr;
        float* d_kernel;

        CUDA_CHECK(cudaMalloc(&d_input, total_len * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_output, total_len * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_img_height_width, sub_batch_size * 2 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_size_ptr, (sub_batch_size + 1) * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float)));
        
        cout << "Finish preprocess, offloading batch size of: " << sub_batch_size << " to GPU\n";
        
        cudaMemcpy(d_img_height_width, img_height_width_array, sub_batch_size * 2 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_size_ptr, img_size_ptr_array, (sub_batch_size + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, kernal_1D, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

        // Call CUDA memcpy for each img
        for(int i = 0; i < sub_batch_size; ++i) {
            // Find starting index of img #i
            int img_offset = img_size_ptr_array[i];
            cudaMemcpy(d_input + img_offset, img_data_array[i], (img_size_ptr_array[i + 1] - img_size_ptr_array[i]) * sizeof(uint8_t), cudaMemcpyHostToDevice);
            stbi_image_free(img_data_array[i]);
        }

        auto load_start = high_resolution_clock::now();


        dim3 blockSize(16, 16, 1);
        dim3 gridSize((max_width + blockSize.x - 1) / blockSize.x,
                      (max_height + blockSize.y - 1) / blockSize.y,
                       sub_batch_size);
        convolution2DKernel<<<gridSize, blockSize>>>(d_input, d_output, d_img_height_width, d_size_ptr, d_kernel);

        cudaMemcpy(output, d_output, total_len * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        delete[] img_data_array;
        delete[] img_height_width_array;
        delete[] img_size_ptr_array;
        delete[] output;

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_img_height_width);
        cudaFree(d_size_ptr);
        cudaFree(d_kernel);

        auto load_end = high_resolution_clock::now();


        cout << "Convolution finish\n";
        process_time += duration_cast<milliseconds>(load_end - load_start).count();

    }

    auto total_end = high_resolution_clock::now();
    cout << "Total execution time: " 
         << duration_cast<milliseconds>(total_end - total_start).count() << " ms\n" ;
    cout << "Total convolution time: " << process_time << " ms\n";
    cout << "Number of images processed: " << image_files.size() << "\n";

    return 0;
}