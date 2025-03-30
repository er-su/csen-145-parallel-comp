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
#include <omp.h>

namespace fs = std::filesystem;
using namespace std;
using namespace std::chrono;

const int KERNEL_SIZE = 3;
const float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {1, 1, 1},
    {1, -8, 1},
    {1, 1, 1}
};

// Function to apply 2D convolution on a 1D array
void convolution2Domp(const uint8_t* input, uint8_t* output, int width, int height) {
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

int main(int argc, char* argv[]) {

    // Start timing the total execution time
    auto total_start = high_resolution_clock::now();

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


    int nt = 0;
    long process_time = 0;

    // Process each image in parallel
    #pragma omp parallel
    {
        nt = omp_get_num_threads();
        uint8_t* local_output_data = new uint8_t[width * height](); // Local output for each thread
        uint8_t* local_img_data = nullptr; // Local image data for each thread

        #pragma omp for reduction(+:process_time)
        for (size_t i = 0; i < image_files.size(); ++i) {

            local_img_data = stbi_load(image_files[i].c_str(), &width, &height, &channels, 1);
            if (!local_img_data) {
                cerr << "Error: Unable to open or read the image file: " << image_files[i] << endl;
                continue;
            }

            // Execute in parallel
            auto load_start = high_resolution_clock::now();
            convolution2Domp(local_img_data, local_output_data, width, height);
            auto load_end = high_resolution_clock::now();
            
            process_time += duration_cast<milliseconds>(load_end - load_start).count();

            stbi_image_free(local_img_data); // Free the local image data
        }
        delete[] local_output_data; // Free local output data
    }

    auto total_end = high_resolution_clock::now();
    cout << "Total execution time: " << duration_cast<milliseconds>(total_end - total_start).count() << " ms" << endl;
    cout << "Total convolution time: " << process_time / nt << " ms\n";
    cout << "Number of images processed: " << image_files.size() << "\n";



    return 0;
}