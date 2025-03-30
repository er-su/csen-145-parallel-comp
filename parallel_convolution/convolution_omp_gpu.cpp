// Erick Sun
// CSEN 145 Fa 2024
// This program takes a directory of images and performs convolution on all
// of the applicable images in the directory

 /*
 * THIS IS CURRENTLY NOT CORRECTLY OFFLOADING TO THE GPU
 */

#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <omp.h>
#include <unordered_map>

namespace fs = std::filesystem;
using namespace std;
using namespace std::chrono;

const int KERNEL_SIZE = 3;
const float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {1, 1, 1},
    {1, -8, 1},
    {1, 1, 1}
};

struct ImageData {
    uint8_t* data;
    int width;
    int height;
    int channels;
};

float kernal_1D[KERNEL_SIZE * KERNEL_SIZE] = {1, 1, 1, 1, -8, 1, 1, 1, 1};

// A const that gives the proportion of the CPU
const float CPU_TO_GPU = 0.8;
const int GPU_BATCH_SIZE = 2048;

// Function to apply 2D convolution on a 1D array
void convolution2DompGPU(const uint8_t* input, uint8_t* output, int height, int width, float* kernel, int kernel_size, int batch_size) {
    int total_size = height * width * batch_size;
    int img_size = height * width;
    int offset = kernel_size / 2;
    #pragma omp target teams map(to: input[0:total_size], kernel[0:9]) map(from: output[0:total_size]) firstprivate(offset) shared(img_size, batch_size) num_teams(batch_size)
    {
        #pragma omp distribute 
        for(int i = 0; i < batch_size; ++i) {
            #pragma omp parallel for collapse(2)
            for (int y = offset; y < height - offset; ++y) {
                for (int x = offset; x < width - offset; ++x) {
                    float sum = 0.0f;
                    for (int ky = -offset; ky <= offset; ++ky) {
                        for (int kx = -offset; kx <= offset; ++kx) {
                            int pixel_index = (y + ky) * width + (x + kx);
                            sum += input[pixel_index + i * img_size] * kernel[(ky + offset) * 3 + kx + offset];
                        }
                    }
                    output[y * width + x + i * img_size] = min(max(int(sum), 0), 255);
                }
            }
        }
    }
}

// Function to apply 2D convolution on a 1D array
void convolution2DompCPU(const uint8_t* input, uint8_t* output, int width, int height) {
    int offset = KERNEL_SIZE / 2;
    for (int y = offset; y < height - offset; ++y) {
        for (int x = offset; x < width - offset; ++x) {
            float sum = 0.0f;
            for (int ky = -offset; ky <= offset; ++ky) {
                for (int kx = -offset; kx <= offset; ++kx) {
                    int pixel_index = (y + ky) * width + (x + kx);
                    sum += input[pixel_index] * kernel[ky + offset][kx + offset];
                }
            }
            output[y * width + x] = min(max(int(sum), 0), 255);
        }
    }
}

void loadAndCategorizeImages(const std::vector<std::string>& image_paths, std::unordered_map<std::string, std::vector<ImageData>>& categorized_images) {
    #pragma omp parallel for
    for (size_t i = 0; i < image_paths.size(); ++i) {
        const auto& path = image_paths[i];
        int width, height, channels;
        uint8_t* data = stbi_load(path.c_str(), &width, &height, &channels, 1);

        if (data) {
            // Create a unique key based on dimensions
            std::string dimensionKey = std::to_string(width) + "x" + std::to_string(height);

            // Use critical section to safely access the shared map
            #pragma omp critical
            {
                categorized_images[dimensionKey].emplace_back(ImageData{data, width, height, channels});
            }
        } else {
            #pragma omp critical
            {
                std::cerr << "Failed to load image: " << path << "\n";
            }
        }
    }
}

void freeLoadedImages(std::unordered_map<std::string, std::vector<ImageData>>& categorized_images) {
    for (auto& pair : categorized_images) {
        for (auto& image : pair.second) {
            stbi_image_free(image.data);
        }
    }
}

// Function to create 1D arrays of pixel data for each dimension
std::unordered_map<std::string, uint8_t*> create1DArrays(const std::unordered_map<std::string, std::vector<ImageData>>& categorized_images) {
    std::unordered_map<std::string, uint8_t*> oneD_array;
    // Fix this
    for(auto &pair : categorized_images) {
        int size_of_img = pair.second[0].width * pair.second[0].height;
        int total_size = size_of_img * pair.second.size();
        cout << "Image size is " << size_of_img << " with the total size being " << total_size << "\n";

        oneD_array[pair.first] = new uint8_t[total_size];
        #pragma omp parallel for shared(size_of_img, total_size, oneD_array) 
        for(int i = 0; i < pair.second.size(); ++i) {
            std::copy(pair.second[i].data, pair.second[i].data + size_of_img, oneD_array[pair.first] + i * size_of_img);
        }
    }
    return oneD_array;
}

std::unordered_map<std::string, std::pair<int, int>> extractDimensions(const std::unordered_map<std::string, uint8_t*>& oneD_array, 
                                                                       const std::unordered_map<std::string, std::vector<ImageData>>& categorized_images) {
    std::unordered_map<std::string, std::pair<int, int>> dimensions;

    for (const auto& pair : categorized_images) {
        const std::string& key = pair.first;
        if (pair.second.empty()) {
            continue; // Skip if there are no images for this key
        }

        // Assuming all images under the same key have the same dimensions
        int width = pair.second[0].width;
        int height = pair.second[0].height;

        // Store the dimensions in the map
        dimensions[key] = std::make_pair(width, height);
    }

    return dimensions;
}

// Function to free the 1D arrays
void freeOneDArrays(std::unordered_map<std::string, uint8_t*>& oneD_array) {
    for (auto& pair : oneD_array) {
        delete[] pair.second; // Free each 1D array
    }
}

int main(int argc, char* argv[]) {

    // Start timing the total execution time
    auto total_start = high_resolution_clock::now();
    auto load_start = total_start;
    auto load_end = total_start;
    long process_time = 0;

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

    // Load the first image to get the dimensions
    int width, height, channels;
    uint8_t* img_data = stbi_load(image_files[0].c_str(), &width, &height, &channels, 1);
    if (!img_data) {
        cerr << "Error: Unable to open or read the image file: " << image_files[0] << endl;
        return -1;
    }
    stbi_image_free(img_data); // Free the loaded image data

    int nt;

    // Find number of threads
    #pragma omp parallel
    {
        nt = omp_get_num_threads();
        #pragma omp single
        cout << "Running with threads: " << nt << "\n";
    }

    int sum = 0;
    std::unordered_map<std::string, std::vector<ImageData>> map_of_sizes;
    loadAndCategorizeImages(image_files, map_of_sizes);
    for(auto pair : map_of_sizes) {
        cout << "Images w dimensions " << pair.first << " with count: " << pair.second.size() << "\n";
        sum += pair.second.size();
    }
    cout << "Total image count is: " << image_files.size() << " compared to " << sum << "\n";

    // Unordered map of <String, 1D_arrays> where the dimension string is the key for the corresponding array
    // Since the unordered map uses the same keys as 
    auto Array1D = create1DArrays(map_of_sizes);
    auto dimensions = extractDimensions(Array1D, map_of_sizes);
    int GPU_CPU_split = CPU_TO_GPU * dimensions.size();
    const int CPU_BATCH_SIZE = nt - 1;

    auto total_mid = high_resolution_clock::now();
    cout << "Time to do processing = " << duration_cast<milliseconds>(total_mid - total_start).count() << "\n";

    // Start processing
    #pragma omp parallel shared(GPU_CPU_split, GPU_BATCH_SIZE, CPU_BATCH_SIZE, dimensions, Array1D, map_of_sizes)
    {
        // Kernel invocation
        #pragma omp master
        {
            int GPU_count = 0;
            for(const auto &pair : Array1D) {
                // Call kernel
                const std::string key = pair.first;
                int total_img_count = map_of_sizes[key].size();
                int width = dimensions[key].first;
                int height = dimensions[key].second;
                int img_size = width * height;
                for(int i = 0; i < total_img_count; i += GPU_BATCH_SIZE) {
                    int actual_batch = min(GPU_BATCH_SIZE, total_img_count - i);
                    uint8_t* temp_img = new uint8_t[img_size * actual_batch];
                    uint8_t* temp_output = new uint8_t[img_size * actual_batch];
                    cout << "Sending dimensions of " << width << "x" << height << " with count of " << actual_batch << "\n";
                    std::copy(&pair.second[img_size * i], &pair.second[img_size * i] + img_size * actual_batch, temp_img);
                    convolution2DompGPU(temp_img, temp_output, height, width, kernal_1D, KERNEL_SIZE, actual_batch);

                    delete[] temp_img;
                    delete[] temp_output;
                }

            }
            
        }
    }

    freeLoadedImages(map_of_sizes);
    freeOneDArrays(Array1D);
    
    
    //process_time += duration_cast<milliseconds>(load_end - load_start).count();

    //cout << "Loaded and processed " << image_files.size() << " images in " << process_time << " ms" << endl;

    auto total_end = high_resolution_clock::now();
    cout << "Total execution time: " 
         << duration_cast<milliseconds>(total_end - total_start).count() << " ms" << endl;

    return 0;
}