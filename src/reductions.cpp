#include "HDMatrix.hpp"

namespace dphdc {
    // Helper function //
    std::vector<unsigned int> HDMatrix::processReduceLabels(const std::vector<std::string> &provided_labels,
                                                            std::vector<std::string> &unique_labels) const {
        if (provided_labels.size() != this->vectors_buff.get_range()[0]) {
            throw std::invalid_argument("The number of labels provided");
        }
        unique_labels = {};
        std::vector<unsigned int> vectors_correspondence(provided_labels.size());

        for (unsigned int i = 0; i < provided_labels.size(); i++) {
            auto iterator = std::find(unique_labels.begin(), unique_labels.end(), provided_labels[i]);
            if (iterator == unique_labels.end()) {
                unique_labels.push_back(provided_labels[i]);
                iterator = std::find(unique_labels.begin(), unique_labels.end(), provided_labels[i]);
            }
            vectors_correspondence[i] = iterator - unique_labels.begin();
        }

        return vectors_correspondence;
    }

    HDMatrix HDMatrix::reduceToLabelsBundle(const std::vector<std::string> &labels_provided) {
        std::vector<std::string> unique_labels;
        std::vector<unsigned int> vector_correspondence = this->processReduceLabels(labels_provided, unique_labels);
        HDMatrix associative_memory((int) this->vectors_buff.get_range()[1], (int) unique_labels.size(),
                                    dphdc::vectors_generator::none, this->associated_queue);
        associative_memory.labels = unique_labels;

        {
            cl::sycl::buffer<short int, 2> buff_accumulators = this->generateInitializeAccumulators(
                    cl::sycl::range<2>(unique_labels.size(), this->vectors_buff.get_range()[1]));

            unsigned int aux;
            for (size_t i = 0; i < vector_correspondence.size(); i++) {
                aux = vector_correspondence[i];
                this->associated_queue.submit([&](cl::sycl::handler &h) {
                    cl::sycl::accessor acc_encoded_vectors(this->vectors_buff, h, cl::sycl::read_only);
                    cl::sycl::accessor acc_accumulators(buff_accumulators, h, cl::sycl::read_write);
                    h.parallel_for(cl::sycl::range<1>(buff_accumulators.get_range()[1]), [=](cl::sycl::id<1> k) {
                        if (acc_encoded_vectors[i][k]) {
                            acc_accumulators[aux][k] += 1;
                        } else {
                            acc_accumulators[aux][k] -= 1;
                        }
                    });
                });
            }

            associative_memory.normalizeAccumulator(buff_accumulators, associative_memory.vectors_buff);
        }

        return associative_memory;
    }
}