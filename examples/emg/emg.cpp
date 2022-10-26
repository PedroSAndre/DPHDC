#include <dphdc.hpp>
#include <readExeInputs.hpp>
#include <ResultsHandler.hpp>
#include "readDataset.hpp"

#define REPRESENTATION_VECTOR_SIZE 21

std::vector<int> generateRepresentationVector() {
    std::vector<int> to_return(REPRESENTATION_VECTOR_SIZE);

    for (int i = 0; i < REPRESENTATION_VECTOR_SIZE; i++) {
        to_return[i] = i;
    }

    return to_return;
}


int main(int argc, char **argv) {
    cl::sycl::queue q{ACCELERATOR_CMAKE_QUEUE()};

    ResultsHandler results_handler;
    results_handler.vector_size = readExeInputs(argc, argv).vector_size;

    dphdc::HDRepresentation<int> interval_representation(results_handler.vector_size,
                                                         dphdc::vectors_generator::full_level, q,
                                                         generateRepresentationVector());
    dphdc::HDMatrix position_vectors(results_handler.vector_size, 4, dphdc::vectors_generator::random, q);
    results_handler.accelerator = position_vectors.getAssociatedAccelerator();

    for (char i = 0; i < 5; i++) {
        dataset dataset = readDataset(PROJECT_PATH_CMAKE "/examples/emg/datasets", i);

        results_handler.startTraining();
        dphdc::HDMatrix associative_memory = interval_representation.encodeWithXOR(dataset.data.train_data,
                                                                                   position_vectors).reduceToLabelsBundle(
                dataset.labels.train_labels);
        results_handler.stopTraining();

        results_handler.startTesting();
        dphdc::HDMatrix encoded_test_entries = interval_representation.encodeWithXOR(dataset.data.test_data,
                                                                                     position_vectors);
        results_handler.success_rate = associative_memory.testModel(encoded_test_entries, dataset.labels.test_labels,
                                                                    dphdc::distance_method::cosine) * 100;
        results_handler.stopTesting();

        results_handler.printToTerminal();
        results_handler.printToFile(PROJECT_PATH_CMAKE "/results/session/results-emg.csv");
    }

    return 0;
}