#include <numeric>
#include <dphdc.hpp>
#include <readExeInputs.hpp>
#include <ResultsHandler.hpp>
#include "cifar10_reader.hpp"

#define N_LABELS 10
#define N_U_CHARS 256

template<typename T>
std::vector<std::size_t> sortPermutation(const std::vector<T> &vec) {
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&](std::size_t i, std::size_t j) { return vec[i] < vec[j]; });
    return p;
}

template<typename T>
void applyPermutationInPlace(std::vector<T> &vec, const std::vector<std::size_t> &p) {
    std::vector<bool> done(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (done[i]) {
            continue;
        }
        done[i] = true;
        std::size_t prev_j = i;
        std::size_t j = p[i];
        while (i != j) {
            std::swap(vec[prev_j], vec[j]);
            done[j] = true;
            prev_j = j;
            j = p[j];
        }
    }
}

size_t indexLastOccurrence(const std::vector<uint8_t> &vector_to_analyse, const uint8_t &to_analyse) {
    size_t index_last_occurrence = 0;

    for (; vector_to_analyse[index_last_occurrence] != to_analyse; index_last_occurrence++);

    for (; vector_to_analyse[index_last_occurrence] == to_analyse; index_last_occurrence++);

    return --index_last_occurrence;
}

std::vector<uint8_t> getUCharsVector() {
    std::vector<uint8_t> to_return(N_U_CHARS);
    for (uint16_t i = 0; i < N_U_CHARS; i++) {
        to_return[i] = (uint8_t) i;
    }
    return to_return;
}

std::vector<std::string> labelsToString(const std::vector<uint8_t> &labels_to_convert) {
    std::vector<std::string> to_return(labels_to_convert.size());

    for (size_t i = 0; i < to_return.size(); i++) {
        to_return[i] = std::to_string(labels_to_convert[i]);
    }

    return to_return;
}

std::vector<uint16_t> generatePositionsVector() {
    std::vector<uint16_t> to_return(1024);

    for (uint16_t i = 0; i < 1024; i++) {
        to_return[i] = i;
    }

    return to_return;
};

std::vector<std::vector<uint16_t>> generateVectorsPosition() {
    std::vector<std::vector<uint16_t>> to_return(1024, std::vector<uint16_t>(1));

    for (uint16_t i = 0; i < 1024; i++) {
        to_return[i][0] = i;
    }

    return to_return;
};


dphdc::HDMatrix generatePositionColorVectors(size_t vector_size, cl::sycl::queue &q) {
    std::vector<dphdc::HDMatrix> to_use_in_constructor;
    to_use_in_constructor.reserve(3);

    dphdc::HDRepresentation<uint16_t> position_vectors(vector_size, dphdc::vectors_generator::full_level, q,
                                                       generatePositionsVector());
    dphdc::HDMatrix red_vector(vector_size, 1, dphdc::vectors_generator::random, q);
    dphdc::HDMatrix green_vector(vector_size, 1, dphdc::vectors_generator::random, q);
    dphdc::HDMatrix blue_vector(vector_size, 1, dphdc::vectors_generator::random, q);

    to_use_in_constructor.push_back(position_vectors.encodeWithXOR(generateVectorsPosition(), red_vector));
    to_use_in_constructor.push_back(position_vectors.encodeWithXOR(generateVectorsPosition(), green_vector));
    to_use_in_constructor.push_back(position_vectors.encodeWithXOR(generateVectorsPosition(), blue_vector));

    dphdc::HDMatrix to_return(to_use_in_constructor);

    return to_return;
}

int main(int argc, char **argv) {
    ResultsHandler results_handler;
    results_handler.vector_size = readExeInputs(argc, argv).vector_size;

    cl::sycl::queue q{ACCELERATOR_CMAKE_QUEUE()};

    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    {
        std::vector<size_t> training_permutation = sortPermutation<uint8_t>(dataset.training_labels);
        std::vector<size_t> testing_permutation = sortPermutation<uint8_t>(dataset.test_labels);

        applyPermutationInPlace<std::vector<uint8_t>>(dataset.training_images, training_permutation);
        applyPermutationInPlace<uint8_t>(dataset.training_labels, training_permutation);
        applyPermutationInPlace<std::vector<uint8_t>>(dataset.test_images, testing_permutation);
        applyPermutationInPlace<uint8_t>(dataset.test_labels, testing_permutation);
    }

    std::vector<std::string> train_labels = labelsToString(dataset.training_labels);
    std::vector<std::string> test_labels = labelsToString(dataset.test_labels);

    dphdc::HDRepresentation<uint8_t> intensity_representation(results_handler.vector_size,
                                                              dphdc::vectors_generator::full_level, q,
                                                              getUCharsVector());
    dphdc::HDMatrix position_color_vectors(results_handler.vector_size, 3072, dphdc::vectors_generator::random, q);
    results_handler.accelerator = intensity_representation.getAssociatedAccelerator();

    dphdc::HDMatrix associative_memory(1, 1, dphdc::vectors_generator::none, q);
    {
        size_t label_range[2];
        label_range[1] = 0;
        std::vector<dphdc::HDMatrix> associative_memory_before_joining;
        associative_memory_before_joining.reserve(N_LABELS);
        results_handler.startTraining();
        for (size_t i = 0; i < N_LABELS; i++) {
            label_range[0] = label_range[1];
            label_range[1] = indexLastOccurrence(dataset.training_labels, i) + 1;

            associative_memory_before_joining.push_back(intensity_representation.encodeWithXOR(
                    {&dataset.training_images[label_range[0]], &dataset.training_images[label_range[1]]},
                    position_color_vectors).reduceToLabelsBundle(
                    {&train_labels[label_range[0]], &train_labels[label_range[1]]}));
        }

        associative_memory = dphdc::HDMatrix(associative_memory_before_joining);
        q.wait();
        results_handler.stopTraining();
    }

    {
        size_t labels_range[2];
        labels_range[1] = 0;
        std::vector<std::string> model_strings;
        model_strings.reserve(test_labels.size());
        results_handler.startTesting();
        for (size_t i = 0; i < N_LABELS; i++) {
            labels_range[0] = labels_range[1];
            labels_range[1] = indexLastOccurrence(dataset.test_labels, i) + 1;
            dphdc::HDMatrix encoded_test_entries = intensity_representation.encodeWithXOR(
                    {&dataset.test_images[labels_range[0]], &dataset.test_images[labels_range[1]]},
                    position_color_vectors);
            std::vector<std::string> temp_vector = associative_memory.queryModel(encoded_test_entries,
                                                                                 dphdc::distance_method::hamming_distance);
            model_strings.insert(model_strings.end(), temp_vector.begin(), temp_vector.end());
        }

        size_t successes = 0;
        for (size_t i = 0; i < test_labels.size(); i++) {
            if (test_labels[i] == model_strings[i]) {
                successes++;
            }
        }

        results_handler.success_rate = ((float) successes / (float) test_labels.size()) * 100;
        q.wait();
        results_handler.stopTesting();
    }


    results_handler.printToTerminal();
    results_handler.printToFile(PROJECT_PATH_CMAKE "/results/session/results-cifar10.csv");

    return 0;
}