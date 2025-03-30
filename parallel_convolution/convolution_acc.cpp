// Erick Sun
// CSEN 145 Fa 2024
// This program takes a directory of images and performs convolution on all
// of the applicable images in the directory
#include <iostream>
#include <chrono>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <openacc.h>
#include <dirent.h>
#include <unordered_map>

using namespace std;
using namespace std::chrono;

struct ImageData {
    uint8_t* data;
    int width;
    int height;
    int channels;
};

const int KERNEL_SIZE = 3;
const int GPU_BATCH_SIZE = 256;
#pragma acc declare create(KERNEL_SIZE)
const float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {1, 1, 1},
    {1, -8, 1},
    {1, 1, 1}
};
float kernal_1D[9] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
#pragma acc declare create(kernel)

// Function to apply 2D convolution on a 1D array
void convolution2Dacc(const uint8_t* input, uint8_t* output, int height, int width, float* kernel, int kernel_size, int batch_size) {
    int total_size = height * width * batch_size;
    int img_size = height * width;
    int offset = kernel_size / 2;
    #pragma acc kernels copyin(input[0:total_size]) copyin(kernel[0:9]) copyout(output[0:total_size]) num_gangs(batch_size)
    {
        #pragma acc loop independent 
        for(int i = 0; i < batch_size; ++i) {
            #pragma acc loop collapse(2) independent vector(height * width)
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

// Loads and categorizes images into different dimension sizes in a map
void loadAndCategorizeImages(const std::vector<std::string>& image_paths, std::unordered_map<std::string, std::vector<ImageData>>& categorized_images) {
    for (size_t i = 0; i < image_paths.size(); ++i) {
        const auto& path = image_paths[i];
        int width, height, channels;
        uint8_t* data = stbi_load(path.c_str(), &width, &height, &channels, 1);

        if (data) {
            // Create a unique key based on dimensions
            std::string dimension_key = std::to_string(width) + "x" + std::to_string(height);

            // Use critical section to safely access the shared map
            categorized_images[dimension_key].emplace_back(ImageData{data, width, height, channels});
        } else {
        
            std::cerr << "Failed to load image: " << path << "\n";
        }
    }
}

// Calls stbi_image_free for all images in the categorized images map
void freeLoadedImages(std::unordered_map<std::string, std::vector<ImageData>>& categorized_images) {
    for (auto& pair : categorized_images) {
        for (auto& image : pair.second) {
            stbi_image_free(image.data);
        }
    }
}

// Function to create 1D arrays of pixel data for each dimension
std::unordered_map<std::string, uint8_t*> create1DArrays(const std::unordered_map<std::string, std::vector<ImageData>>& categorized_images) {
    std::unordered_map<std::string, uint8_t*> oneD_arrays;
    for(auto &pair : categorized_images) {
        int size_of_img = pair.second[0].width * pair.second[0].height;
        int total_size = size_of_img * pair.second.size();
        std::cout << "Image size is " << size_of_img << " with the total size being " << total_size << "\n";

        oneD_arrays[pair.first] = new uint8_t[total_size];
        for(int i = 0; i < pair.second.size(); ++i) {
            // Go thru each array and copy into the new 1D array
            std::copy(pair.second[i].data, pair.second[i].data + size_of_img, oneD_arrays[pair.first] + i * size_of_img);
        }
    }
    return oneD_arrays;
}

// Extracts the dimensions from the corresponding key and places into an unordered map
std::unordered_map<std::string, std::pair<int, int>> extractDimensions(const std::unordered_map<std::string, uint8_t*>& oneD_arrays, 
                                                                       const std::unordered_map<std::string, std::vector<ImageData>>& categorized_images) {
    std::unordered_map<std::string, std::pair<int, int>> dimensions;
    for (const auto& pair : categorized_images) {
        const std::string& key = pair.first;
        if (pair.second.empty()) {
            continue;
        }

        int width = pair.second[0].width;
        int height = pair.second[0].height;
        dimensions[key] = std::make_pair(width, height);
    }

    return dimensions;
}

// Function to free the 1D arrays
void freeOneDArrays(std::unordered_map<std::string, uint8_t*>& oneDArrays) {
    for (auto& pair : oneDArrays) {
        // Free each 1D array
        delete[] pair.second; 
    }
}

int main(int argc, char* argv[]) {

    // Start timing the total execution time
    auto total_start = high_resolution_clock::now();

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_directory>" << endl;
        return -1;
    }

    // Load all iamges
    vector<string> image_files;
    string input_dir = argv[1];
    DIR* dir;
    struct dirent* ent;
    if((dir = opendir(input_dir.c_str())) != NULL) {
        while((ent = readdir(dir)) != NULL) {
            //printf("%s\n", ent->d_name);
            image_files.push_back(input_dir + "/" + ent->d_name);
        }
        closedir(dir);
    }
    else {
        perror("Unable to open directory");
        return -1;
    }

    // Remove unnecessary files
    image_files.erase(image_files.begin());
    image_files.erase(image_files.begin());
    
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

    // Prepare output array
    uint8_t* output_data = new uint8_t[width * height]();

    // Process each image
    std::unordered_map<std::string, std::vector<ImageData>> map_of_sizes;
    loadAndCategorizeImages(image_files, map_of_sizes);
    for(auto pair : map_of_sizes) {
        std::cout << "Images w dimensions " << pair.first << " with count: " << pair.second.size() << "\n";
    }

    auto array_1D = create1DArrays(map_of_sizes);
    auto dimensions = extractDimensions(array_1D, map_of_sizes);

    auto processing_start = high_resolution_clock::now();
    for(const auto &pair : array_1D) {
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

            std::cout << "Sending dimensions of " << width << "x" << height << " with count of " << actual_batch << "\n";

            std::copy(&pair.second[img_size * i], &pair.second[img_size * i] + img_size * actual_batch, temp_img);
            convolution2Dacc(temp_img, temp_output, height, width, kernal_1D, KERNEL_SIZE, actual_batch);

            delete[] temp_img;
            delete[] temp_output;

            
        }
    }

    auto processing_end = high_resolution_clock::now();

    freeLoadedImages(map_of_sizes);
    freeOneDArrays(array_1D);

    auto total_end = high_resolution_clock::now();
    
    std::cout << "Total execution time: " << duration_cast<milliseconds>(total_end - total_start).count() << " ms" << endl;
    std::cout << "Total convolution time: " << duration_cast<milliseconds>(processing_end - processing_start).count() << " ms\n";
    std::cout << "Number of images processed: " << image_files.size() << "\n";

    return 0;
}