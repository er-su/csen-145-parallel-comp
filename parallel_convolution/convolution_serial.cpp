// Erick Sun
// CSEN 145 Fa 2024
// This program takes a directory of images and performs convolution on all
// of the applicable images in the directory
#include <iostream>
#include <filesystem>
#include <chrono>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
void convolution2D(const uint8_t* input, uint8_t* output, int width, int height) {
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

    // Prepare output array
    uint8_t* output_data = new uint8_t[width * height]();

    // Process each image
    int w, h;
    for (const auto& input_file : image_files) {

        img_data = stbi_load(input_file.c_str(), &w, &h, &channels, 1);
        if (!img_data) {
            cerr << "Error: Unable to open or read the image file: " << input_file << endl;
            continue;
        }
        if(w != width || h != height) {
            delete[] output_data;
            output_data = new uint8_t[w * h]();
            width = w;
            height = h;
        }
        auto load_start = high_resolution_clock::now();

        convolution2D(img_data, output_data, width, height);

        auto load_end = high_resolution_clock::now();
        process_time += duration_cast<milliseconds>(load_end - load_start).count();

        stbi_image_free(img_data);
    }

    auto total_end = high_resolution_clock::now();
    cout << "Total execution time: " 
         << duration_cast<milliseconds>(total_end - total_start).count() << " ms" << endl;
    cout << "Total convolution time: " << process_time << " ms\n";
    cout << "Number of images processed: " << image_files.size() << "\n";

    return 0;
}