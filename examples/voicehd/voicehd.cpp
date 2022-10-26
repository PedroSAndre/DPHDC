#include <dphdc.hpp>
#include <readExeInputs.hpp>
#include <ResultsHandler.hpp>
#include "readDataset.hpp"

#include <memory>

#define REPRESENTATION_VECTOR_SIZE 20

std::vector<int> generateRepresentationVector() {
    std::vector<int> to_return(REPRESENTATION_VECTOR_SIZE);

    for (int i = 0; i < REPRESENTATION_VECTOR_SIZE; i++) {
        to_return[i] = i;
    }

    return to_return;
}


int main(int argc, char **argv) {
    cl::sycl::queue q{ACCELERATOR_CMAKE_QUEUE()};
    ResultsHandler results_handler{};
    results_handler.vector_size = readExeInputs(argc, argv).vector_size;

    dphdc::HDRepresentation<int> representation(results_handler.vector_size, dphdc::vectors_generator::full_level, q,
                                                generateRepresentationVector());
    std::unique_ptr<dphdc::HDMatrix> position_vectors;
    std::unique_ptr<dphdc::HDMatrix> associative_memory;

    {
        dataset train_dataset = readDataset(PROJECT_PATH_CMAKE "/examples/voicehd/dataset/isolet1+2+3+4.data");
        position_vectors = std::make_unique<dphdc::HDMatrix>(results_handler.vector_size, train_dataset.data[0].size(),
                                                             dphdc::vectors_generator::random, q);
        results_handler.startTraining();
        associative_memory = std::make_unique<dphdc::HDMatrix>(
                representation.encodeWithXOR(train_dataset.data, *position_vectors).reduceToLabelsBundle(
                        train_dataset.labels));
        q.wait();
        results_handler.stopTraining();
    }

    results_handler.accelerator = associative_memory->getAssociatedAccelerator();

    {
        dataset test_dataset = readDataset(PROJECT_PATH_CMAKE "/examples/voicehd/dataset/isolet5.data");
        results_handler.startTesting();
        dphdc::HDMatrix encoded_test_entries = representation.encodeWithXOR(test_dataset.data, *position_vectors);
        results_handler.success_rate = associative_memory->testModel(encoded_test_entries, test_dataset.labels,
                                                                     dphdc::distance_method::cosine) * 100;
        results_handler.stopTesting();
    }

    results_handler.printToTerminal();
    results_handler.printToFile(PROJECT_PATH_CMAKE "/results/session/results-voicehd.csv");

    return 0;
}