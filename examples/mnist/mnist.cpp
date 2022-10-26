#define N_COLORS 2

#include <dphdc.hpp>
#include <ResultsHandler.hpp>
#include <readExeInputs.hpp>

#include "readDataset.hpp"

std::vector<unsigned char> generateColorsVector() {
    std::vector<unsigned char> to_return(N_COLORS);

    for (unsigned char i = 0; i < N_COLORS - 1; i++) {
        to_return[i] = i;
    }
    return to_return;
}

int main(int argc, char **argv) {
    Inputs inputs = readExeInputs(argc, argv);
    cl::sycl::queue q{ACCELERATOR_CMAKE_QUEUE()};
    ResultsHandler results_handler;
    results_handler.vector_size = inputs.vector_size;

    Data train_data = readDataset(PROJECT_PATH_CMAKE  "/examples/mnist/datasets/train-images-idx3-ubyte",
                                  PROJECT_PATH_CMAKE "/examples/mnist/datasets/train-labels-idx1-ubyte");
    Data test_data = readDataset(PROJECT_PATH_CMAKE  "/examples/mnist/datasets/t10k-images-idx3-ubyte",
                                 PROJECT_PATH_CMAKE "/examples/mnist/datasets/t10k-labels-idx1-ubyte");

    dphdc::HDRepresentation<unsigned char> colors_representation(inputs.vector_size, dphdc::vectors_generator::random,
                                                                 q,
                                                                 generateColorsVector());
    std::vector<std::vector<bool>> colors_on_host = colors_representation.getVectors();

    dphdc::HDMatrix position_vectors(inputs.vector_size, static_cast<int>(train_data.image_size),
                                     dphdc::vectors_generator::random, q);

    results_handler.accelerator = colors_representation.getAssociatedAccelerator();

    results_handler.startTraining();
    dphdc::HDMatrix associative_memory = colors_representation.encodeWithXOR(train_data.image_data,
                                                                             position_vectors).reduceToLabelsBundle(
            train_data.labels_data);
    q.wait();
    results_handler.stopTraining();

    results_handler.startTesting();
    dphdc::HDMatrix encoded_entries = colors_representation.encodeWithXOR(test_data.image_data, position_vectors);
    results_handler.success_rate =
            associative_memory.testModel(encoded_entries, test_data.labels_data,
                                         dphdc::distance_method::hamming_distance) * 100;
    q.wait();
    results_handler.stopTesting();

    results_handler.printToTerminal();
    results_handler.printToFile(PROJECT_PATH_CMAKE "/results/session/results-mnist.csv");

    return 0;
}