#ifndef DPHDC_HDMATRIX_HPP
#define DPHDC_HDMATRIX_HPP

#include <CL/sycl.hpp>
#include "selectors.hpp"

namespace dphdc {
    namespace permutation {
        enum permutation {
            no_permutation, shift_right
        };
    }

    namespace vectors_generator {
        enum vectors_generator {
            none, all_true, random, half_level, full_level, circular
        };
    }

    namespace distance_method {
        enum distance_method {
            hamming_distance, cosine
        };
    }

    class HDMatrix {
    protected:
        cl::sycl::queue associated_queue;
        std::vector<std::string> labels = {};

        // Buffer helper functions //
        void copyBuffer(cl::sycl::buffer<bool, 2> &origin, cl::sycl::buffer<bool, 2> &dest);

        cl::sycl::buffer<bool, 2> copyBuffer(cl::sycl::buffer<bool, 2> &origin);

        cl::sycl::buffer<short int, 2> generateInitializeAccumulators(cl::sycl::range<2> range);

        // Permutation functions //
        void shiftRight(cl::sycl::buffer<bool, 2> &buffer_to_shift, cl::sycl::buffer<bool, 2> &duplicate_of_buffer);

        // Copy from bool vector in host memory - used with vector generators //
        void copyBoolVector(const bool *vector_to_copy);

        // Vector Generators - Usually just used in the constructor //
        void constantVectorGenerator(bool value);

        void randomVectorGenerator();

        void levelVectorGenerator(bool full_level);

        void circularVectorGenerator();

        // Reduce labels //
        std::vector<unsigned int> processReduceLabels(const std::vector<std::string> &provided_labels,
                                                      std::vector<std::string> &unique_labels) const;

        // Reduces an accumulator to a normal HDC vector //
        void normalizeAccumulator(cl::sycl::buffer<short int, 2> &accumulators, cl::sycl::buffer<bool, 2> &destination);

        // Query distances //
        // Returns the index of the closest vector in the memory using hamming distance //
        std::vector<unsigned int> hammingDistanceIndexVector(HDMatrix &encoded_test_entries);

        // Returns the index of the closest vector in the memory using cosine similarity//
        std::vector<unsigned int> cosineIndexVector(HDMatrix &encoded_test_entries);

        // Saves and Reads all class info into a binary file //
        void saveBuffer(std::ofstream &file);

        void readBuffer(std::ifstream &file);

        void saveLabels(std::ofstream &file);

        void readLabels(std::ifstream &file);

        // Switch functions //
        void switchAcceleratorSelector(dphdc::selector accelerator_selector);

        void switchVectorGenerator(dphdc::vectors_generator::vectors_generator vectors_type);

    public:
        cl::sycl::buffer<bool, 2> vectors_buff;

        HDMatrix(int vector_size, int n_vectors, vectors_generator::vectors_generator vectors_type,
                 dphdc::selector accelerator_selector);

        HDMatrix(int vector_size, int n_vectors, vectors_generator::vectors_generator vectors_type, cl::sycl::queue &q);

        explicit HDMatrix(std::vector<HDMatrix> &matrices_to_join);

        HDMatrix(const std::string &full_file_path, dphdc::selector accelerator_selector);

        HDMatrix(const std::string &full_file_path, cl::sycl::queue &q);

        // Reducers according to labels //
        HDMatrix reduceToLabelsBundle(const std::vector<std::string> &provided_labels);

        float testModel(HDMatrix &encoded_test_entries, const std::vector<std::string> &test_data_labels,
                        distance_method::distance_method method_to_use);

        std::vector<std::string>
        queryModel(HDMatrix &encoded_test_entries, distance_method::distance_method method_to_use);

        // Getter and Setters //
        void storeMatrix(std::string full_file_path);

        std::vector<std::vector<bool>> getVectors();

        void setVectors(const std::vector<std::vector<bool>> &vectors_to_set);

        std::string getAssociatedAccelerator();

        [[nodiscard]] const std::vector<std::string> &getLabels() const;

        void setAssociatedQueue(const sycl::queue &associatedQueue);
    };

} // dphdc
#endif //DPHDC_HDMATRIX_HPP
