#include "HDMatrix.hpp"

namespace dphdc {
    std::vector<unsigned int> HDMatrix::hammingDistanceIndexVector(HDMatrix &encoded_test_entries) {
        cl::sycl::range<3> total_range(encoded_test_entries.vectors_buff.get_range()[0],
                                       this->vectors_buff.get_range()[0], this->vectors_buff.get_range()[1]);

        std::vector<std::vector<size_t>> distance_vectors(total_range[0], std::vector<size_t>(total_range[1]));
        {
            std::unique_ptr<size_t[]> distance_vectors_temp(new size_t[total_range[0] * total_range[1]]);
            {
                cl::sycl::range<2> first_two_range(total_range[0], total_range[1]);
                cl::sycl::buffer<size_t, 2> buff_distance_vectors(distance_vectors_temp.get(), first_two_range);

                this->associated_queue.submit([&](cl::sycl::handler &h) {
                    cl::sycl::accessor acc_distance_vectors(buff_distance_vectors, h, cl::sycl::write_only);
                    h.parallel_for(first_two_range, [=](cl::sycl::id<2> local_range) {
                        size_t i = local_range[0];
                        size_t j = local_range[1];
                        acc_distance_vectors[i][j] = 0;
                    });
                });

                for (size_t k = 0; k < total_range[2]; k++) {
                    this->associated_queue.submit([&](cl::sycl::handler &h) {
                        cl::sycl::accessor acc_distance_vectors(buff_distance_vectors, h, cl::sycl::read_write);
                        cl::sycl::accessor acc_encoded_test_entries(encoded_test_entries.vectors_buff, h,
                                                                    cl::sycl::read_only);
                        cl::sycl::accessor acc_model_entries(this->vectors_buff, h, cl::sycl::read_only);
                        h.parallel_for(first_two_range, [=](cl::sycl::id<2> local_range) {
                            size_t i = local_range[0];
                            size_t j = local_range[1];
                            if (acc_encoded_test_entries[i][k] ^ acc_model_entries[j][k]) {
                                acc_distance_vectors[i][j] += 1;
                            }
                        });
                    });
                }
            }


            unsigned int aux = 0;
            for (unsigned int i = 0; i < distance_vectors.size(); i++) {
                for (unsigned int j = 0; j < distance_vectors[0].size(); j++) {
                    distance_vectors[i][j] = distance_vectors_temp.get()[aux];
                    aux++;
                }
            }
        }

        std::vector<unsigned int> to_return(distance_vectors.size());
        for (unsigned int i = 0; i < distance_vectors.size(); i++) {
            to_return[i] = std::min_element(distance_vectors[i].begin(), distance_vectors[i].end()) -
                           distance_vectors[i].begin();
        }

        return to_return;
    }

    std::vector<unsigned int> HDMatrix::cosineIndexVector(HDMatrix &encoded_test_entries) {
        cl::sycl::range<3> total_range(encoded_test_entries.vectors_buff.get_range()[0],
                                       this->vectors_buff.get_range()[0], this->vectors_buff.get_range()[1]);

        std::vector<std::vector<long long int>> distance_vectors(total_range[0],
                                                                 std::vector<long long int>(total_range[1]));
        {
            std::unique_ptr<long long int[]> distance_vectors_temp(new long long int[total_range[0] * total_range[1]]);
            {
                cl::sycl::range<2> first_two_range(total_range[0], total_range[1]);
                cl::sycl::buffer<long long int, 2> buff_distance_vectors(distance_vectors_temp.get(), first_two_range);

                this->associated_queue.submit([&](cl::sycl::handler &h) {
                    cl::sycl::accessor acc_distance_vectors(buff_distance_vectors, h, cl::sycl::write_only);
                    h.parallel_for(first_two_range, [=](cl::sycl::id<2> local_range) {
                        size_t i = local_range[0];
                        size_t j = local_range[1];
                        acc_distance_vectors[i][j] = 0;
                    });
                });

                for (size_t k = 0; k < total_range[2]; k++) {
                    this->associated_queue.submit([&](cl::sycl::handler &h) {
                        cl::sycl::accessor acc_distance_vectors(buff_distance_vectors, h, cl::sycl::read_write);
                        cl::sycl::accessor acc_encoded_test_entries(encoded_test_entries.vectors_buff, h,
                                                                    cl::sycl::read_only);
                        cl::sycl::accessor acc_model_entries(this->vectors_buff, h, cl::sycl::read_only);
                        h.parallel_for(first_two_range, [=](cl::sycl::id<2> local_range) {
                            size_t i = local_range[0];
                            size_t j = local_range[1];
                            if (acc_encoded_test_entries[i][k] == acc_model_entries[j][k]) {
                                acc_distance_vectors[i][j] += 1;
                            } else {
                                acc_distance_vectors[i][j] -= 1;
                            }
                        });
                    });
                }
            }


            unsigned int aux = 0;
            for (unsigned int i = 0; i < distance_vectors.size(); i++) {
                for (unsigned int j = 0; j < distance_vectors[0].size(); j++) {
                    distance_vectors[i][j] = distance_vectors_temp.get()[aux];
                    aux++;
                }
            }
        }

        std::vector<unsigned int> to_return(distance_vectors.size());
        for (unsigned int i = 0; i < distance_vectors.size(); i++) {
            to_return[i] = std::max_element(distance_vectors[i].begin(), distance_vectors[i].end()) -
                           distance_vectors[i].begin();
        }

        return to_return;
    }

    float HDMatrix::testModel(HDMatrix &encoded_test_entries, const std::vector<std::string> &test_data_labels,
                              distance_method::distance_method method_to_use) {
        if (this->labels.size() != this->vectors_buff.get_range()[0]) {
            throw std::invalid_argument("Model labels and vector entries do not match");
        }
        if (test_data_labels.size() != encoded_test_entries.vectors_buff.get_range()[0]) {
            throw std::invalid_argument("Test labels and test vector entries do not match");
        }
        if (this->vectors_buff.get_range()[1] != encoded_test_entries.vectors_buff.get_range()[1]) {
            throw std::invalid_argument("HDC vector size needs to match");
        }

        std::vector<unsigned int> indexes_vector;
        switch (method_to_use) {
            case distance_method::hamming_distance:
                indexes_vector = this->hammingDistanceIndexVector(encoded_test_entries);
                break;
            case distance_method::cosine:
                indexes_vector = this->cosineIndexVector(encoded_test_entries);
                break;
        }


        unsigned int entries_success = 0;
        for (unsigned int i = 0; i < indexes_vector.size(); i++) {
            if (this->labels[indexes_vector[i]] == test_data_labels[i]) {
                entries_success++;
            }
        }

        return ((float) entries_success) / ((float) encoded_test_entries.vectors_buff.get_range()[0]);
    }

    std::vector<std::string>
    HDMatrix::queryModel(HDMatrix &encoded_test_entries, distance_method::distance_method method_to_use) {
        if (this->labels.size() != this->vectors_buff.get_range()[0]) {
            throw std::invalid_argument("Model labels and vector entries do not match");
        }
        if (this->vectors_buff.get_range()[1] != encoded_test_entries.vectors_buff.get_range()[1]) {
            throw std::invalid_argument("HDC vector size needs to match");
        }

        std::vector<unsigned int> indexes_vector;
        switch (method_to_use) {
            case distance_method::hamming_distance:
                indexes_vector = this->hammingDistanceIndexVector(encoded_test_entries);
                break;
            case distance_method::cosine:
                indexes_vector = this->cosineIndexVector(encoded_test_entries);
                break;
        }

        std::vector<std::string> to_return(indexes_vector.size());

        for (unsigned int i = 0; i < indexes_vector.size(); i++) {
            to_return[i] = this->labels[indexes_vector[i]];
        }

        return to_return;
    }
} //dphdc