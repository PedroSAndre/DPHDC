#include "HDRepresentation.hpp"

namespace dphdc {
    template<class TypeOfDataToRepresent>
    std::unique_ptr<int[]>
    HDRepresentation<TypeOfDataToRepresent>::convertData(const std::vector<std::vector<TypeOfDataToRepresent>> &data,
                                                         int &max_element_size) {
        max_element_size = std::max_element(data.begin(), data.end(), [](const std::vector<TypeOfDataToRepresent> &lhs,
                                                                         const std::vector<TypeOfDataToRepresent> &rhs) -> bool {
            return lhs.size() < rhs.size();
        })->size();

        if (max_element_size <= 0 || data.empty()) {
            throw std::invalid_argument("Data and elements provided to encoder need to be bigger than 0");
        }

        std::unique_ptr<int[]> converted_data(new int[max_element_size * data.size()]);

        for (size_t i = 0; i < max_element_size * data.size(); i++) {
            converted_data.get()[i] = -1;
        }

        for (size_t i = 0; i < data.size(); i++) {
            for (size_t j = 0; j < data[i].size(); j++) {
                if (this->data_translation.count(data[i][j])) {
                    converted_data.get()[i * max_element_size + j] = this->data_translation[data[i][j]] - 1;
                } else {
                    converted_data.get()[i * max_element_size + j] = -1;
                }
            }
        }

        return converted_data;
    }


    template<class TypeOfDataToRepresent>
    HDMatrix HDRepresentation<TypeOfDataToRepresent>::encodeWithBundle(
            const std::vector<std::vector<TypeOfDataToRepresent>> &data, permutation::permutation permutation_to_use) {
        int max_element_size;
        std::unique_ptr<int[]> converted_data = this->convertData(data, max_element_size);


        HDMatrix encoded_vectors_matrix(this->vectors_buff.get_range()[1], data.size(), dphdc::vectors_generator::none,
                                        this->associated_queue);

        {
            cl::sycl::buffer<short int, 2> buff_accumulators = this->generateInitializeAccumulators(
                    cl::sycl::range<2>(data.size(), this->vectors_buff.get_range()[1]));

            cl::sycl::buffer<bool, 2> buff_copy_of_representation = this->copyBuffer(this->vectors_buff);
            cl::sycl::buffer<bool, 2> buff_copy_of_representation_duplicate = this->copyBuffer(this->vectors_buff);

            cl::sycl::buffer<int, 2> buff_data(converted_data.get(), cl::sycl::range<2>(data.size(), max_element_size));
            buff_data.set_write_back(false);

            for (unsigned int j = 0; j < max_element_size; j++) {
                this->associated_queue.submit([&](cl::sycl::handler &h) {
                    cl::sycl::accessor acc_accumulators(buff_accumulators, h, cl::sycl::read_write);
                    cl::sycl::accessor acc_representation(buff_copy_of_representation, h, cl::sycl::read_only);
                    cl::sycl::accessor acc_data(buff_data, h, cl::sycl::read_only);
                    h.parallel_for(cl::sycl::range<2>(data.size(), this->vectors_buff.get_range()[1]),
                                   [=](cl::sycl::id<2> local_range) {
                                       size_t i = local_range[0];
                                       size_t k = local_range[1];
                                       if (acc_data[i][j] >= 0) {
                                           if (acc_representation[acc_data[i][j]][k]) {
                                               acc_accumulators[i][k] += 1;
                                           } else {
                                               acc_accumulators[i][k] -= 1;
                                           }
                                       }
                                   });
                });

                switch (permutation_to_use) {
                    case permutation::shift_right:
                        this->shiftRight(buff_copy_of_representation, buff_copy_of_representation_duplicate);
                        break;
                    default:
                        break;
                }
            }


            this->normalizeAccumulator(buff_accumulators, encoded_vectors_matrix.vectors_buff);
        }

        return encoded_vectors_matrix;
    }

    template<class TypeOfDataToRepresent>
    HDMatrix
    HDRepresentation<TypeOfDataToRepresent>::encodeWithXOR(const std::vector<std::vector<TypeOfDataToRepresent>> &data,
                                                           HDMatrix &position_vectors) {
        for (unsigned int i = 0; i < data.size(); i++) {
            if (position_vectors.vectors_buff.get_range()[0] != data[i].size()) {
                throw std::invalid_argument(
                        "Every data entry needs to have the same size and this must be the amount of position vectors provided");
            }
        }
        if (this->vectors_buff.get_range()[1] != position_vectors.vectors_buff.get_range()[1]) {
            throw std::invalid_argument("Vector size of position vectors needs to be the same as HDRepresentation");
        }

        int max_element_size = 0;
        std::unique_ptr<int[]> converted_data = this->convertData(data, max_element_size);
        HDMatrix encoded_vectors_matrix(this->vectors_buff.get_range()[1], data.size(), vectors_generator::none,
                                        this->associated_queue);

        {
            cl::sycl::buffer<short int, 2> buff_accumulators = this->generateInitializeAccumulators(
                    cl::sycl::range<2>(data.size(), this->vectors_buff.get_range()[1]));

            cl::sycl::buffer<int, 2> buff_data(converted_data.get(), cl::sycl::range<2>(data.size(), data[0].size()));
            buff_data.set_write_back(false);

            for (unsigned int j = 0; j < data[0].size(); j++) {
                this->associated_queue.submit([&](cl::sycl::handler &h) {
                    cl::sycl::accessor acc_accumulators(buff_accumulators, h, cl::sycl::read_write);
                    cl::sycl::accessor acc_data(buff_data, h, cl::sycl::read_only);
                    cl::sycl::accessor acc_representation(this->vectors_buff, h, cl::sycl::read_only);
                    cl::sycl::accessor acc_position_vectors(position_vectors.vectors_buff, h, cl::sycl::read_only);
                    h.parallel_for(cl::sycl::range<2>(data.size(), this->vectors_buff.get_range()[1]),
                                   [=](cl::sycl::id<2> local_range) {
                                       size_t i = local_range[0];
                                       size_t k = local_range[1];
                                       if (acc_data[i][j] >= 0) {
                                           if (acc_representation[acc_data[i][j]][k] ^ acc_position_vectors[j][k]) {
                                               acc_accumulators[i][k] += 1;
                                           } else {
                                               acc_accumulators[i][k] -= 1;
                                           }
                                       }
                                   });
                });
            }

            this->normalizeAccumulator(buff_accumulators, encoded_vectors_matrix.vectors_buff);
        }


        return encoded_vectors_matrix;
    }

    template<class TypeOfDataToRepresent>
    HDMatrix
    HDRepresentation<TypeOfDataToRepresent>::encodeWithXOR(const std::vector<std::vector<TypeOfDataToRepresent>> &data,
                                                           permutation::permutation permutation_to_use) {
        int max_element_size;
        std::unique_ptr<int[]> converted_data = this->convertData(data, max_element_size);


        HDMatrix encoded_vectors_matrix(this->vectors_buff.get_range()[1], data.size(), dphdc::vectors_generator::none,
                                        this->associated_queue);

        {
            cl::sycl::buffer<bool, 2> buff_copy_of_representation = this->copyBuffer(this->vectors_buff);
            cl::sycl::buffer<bool, 2> buff_copy_of_representation_duplicate = this->copyBuffer(this->vectors_buff);

            cl::sycl::buffer<int, 2> buff_data(converted_data.get(), cl::sycl::range<2>(data.size(), max_element_size));
            buff_data.set_write_back(false);

            for (unsigned int j = 0; j < max_element_size; j++) {
                this->associated_queue.submit([&](cl::sycl::handler &h) {
                    cl::sycl::accessor acc_encoded_vectors(encoded_vectors_matrix.vectors_buff, h,
                                                           cl::sycl::read_write);
                    cl::sycl::accessor acc_representation(buff_copy_of_representation, h, cl::sycl::read_only);
                    cl::sycl::accessor acc_data(buff_data, h, cl::sycl::read_only);
                    h.parallel_for(cl::sycl::range<2>(data.size(), this->vectors_buff.get_range()[1]),
                                   [=](cl::sycl::id<2> local_range) {
                                       size_t i = local_range[0];
                                       size_t k = local_range[1];
                                       if (acc_data[i][j] >= 0) {
                                           acc_encoded_vectors[i][k] ^= acc_representation[acc_data[i][j]][k];
                                       }
                                   });
                });

                switch (permutation_to_use) {
                    case permutation::shift_right:
                        this->shiftRight(buff_copy_of_representation, buff_copy_of_representation_duplicate);
                        break;
                    default:
                        break;
                }
            }
        }

        return encoded_vectors_matrix;
    }
}

#include "supported_data.hpp"