#include <dphdc.hpp>
#include <readExeInputs.hpp>
#include <ResultsHandler.hpp>
#include "readDataset.hpp"

int main(int argc, char **argv) {
    ResultsHandler results_handler;
    Inputs inputs_from_console = readExeInputs(argc, argv);

    results_handler.vector_size = inputs_from_console.vector_size;

    std::pair<std::vector<std::vector<char>>, std::vector<std::string>> train_data = readDataset(
            PROJECT_PATH_CMAKE "/examples/hdna/datasets/batsTrain.fas");
    std::pair<std::vector<std::vector<char>>, std::vector<std::string>> test_data = readDataset(
            PROJECT_PATH_CMAKE "/examples/hdna/datasets/batsTest.fas");

    cl::sycl::queue q{ACCELERATOR_CMAKE_QUEUE()};
    std::vector<char> dna_bases = {'A', 'C', 'G', 'T'};
    dphdc::HDRepresentation<char> hd_representation_dna_bases(inputs_from_console.vector_size,
                                                              dphdc::vectors_generator::random,
                                                              q, dna_bases);

    results_handler.accelerator = hd_representation_dna_bases.getAssociatedAccelerator();

    results_handler.startTraining();
    dphdc::HDMatrix associative_memory = hd_representation_dna_bases.encodeWithBundle(train_data.first,
                                                                                      dphdc::permutation::shift_right).reduceToLabelsBundle(
            train_data.second);
    q.wait();
    results_handler.stopTraining();

    results_handler.startTesting();
    dphdc::HDMatrix encoded_test_entries = hd_representation_dna_bases.encodeWithBundle(test_data.first,
                                                                                        dphdc::permutation::shift_right);
    results_handler.success_rate =
            associative_memory.testModel(encoded_test_entries, test_data.second,
                                         dphdc::distance_method::hamming_distance) * 100;
    q.wait();
    results_handler.stopTesting();

    results_handler.printToTerminal();
    results_handler.printToFile(PROJECT_PATH_CMAKE "/results/session/results-hdna.csv");
    return 0;
}