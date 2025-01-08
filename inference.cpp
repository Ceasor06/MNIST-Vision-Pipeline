#include <torch/script.h> // LibTorch headers
#include <opencv2/opencv.hpp> // OpenCV headers
#include <iostream>
#include <memory>
#include <vector>
#include <filesystem> // For folder traversal

namespace fs = std::filesystem; // Shorten filesystem namespace

int main(int argc, char* argv[]) {
    // 1. Validate Command-Line Arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.pt> <path_to_folder>\n";
        return -1;
    }

    std::string model_path = argv[1];
    std::string folder_path = argv[2];

    // 2. Load the TorchScript Model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path); // Load the pre-trained model
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }
    std::cout << "Model loaded successfully.\n";

    // 3. Check for MPS or CPU
    bool use_mps = false; // Flag to track MPS usage

    if (torch::hasMPS()) {
        std::cout << "MPS is available! Running on GPU." << std::endl;
        model.to(torch::kMPS);
        use_mps = true; // Enable MPS flag
    } else {
        std::cout << "Running on CPU." << std::endl;
    }

    // 4. Process Each Image in the Folder
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        std::string image_path = entry.path().string(); // Get the image path
        std::cout << "Processing image: " << image_path << std::endl;

        // 5. Load and Preprocess the Image
        cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE); // Read as grayscale
        if (img.empty()) {
            std::cerr << "Error: Could not read image at " << image_path << std::endl;
            continue; // Skip this image
        }

        // Resize image to 28x28 if needed
        if (img.cols != 28 || img.rows != 28) {
            cv::resize(img, img, cv::Size(28, 28));
        }

        // Normalize image
        img.convertTo(img, CV_32F, 1.0 / 255.0); // Scale to [0, 1]
        img = (img - 0.1307) / 0.3081;

        // Convert OpenCV image to Torch Tensor [1, 1, 28, 28]
        at::Tensor input_tensor = torch::from_blob(
            img.data, {1, 1, 28, 28}, torch::kFloat);

        // Move input tensor to MPS or CPU based on availability
        if (use_mps) {
            input_tensor = input_tensor.to(torch::kMPS);
        }

        // 6. Perform Forward Pass
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        at::Tensor output = model.forward(inputs).toTensor(); // Model inference

        // Move output back to CPU for processing if MPS was used
        if (use_mps) {
            output = output.to(torch::kCPU);
        }

        // 7. Extract Prediction and Confidence Scores
        auto probabilities = torch::softmax(output, 1);         // Apply softmax
        auto prediction = probabilities.argmax(1).item<int>(); // Get prediction

        std::cout << "Predicted digit: " << prediction << std::endl;
        std::cout << "Confidence scores: " << probabilities << std::endl;

        std::ofstream result_file("results.txt", std::ios::app); // Append to results
        result_file << "Image: " << image_path 
                    << " | Prediction: " << prediction 
                    << " | Confidence Scores: " << probabilities << std::endl;
        result_file.close();
    }
    return 0;
}
