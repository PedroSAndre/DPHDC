#include "HDRepresentation.hpp"

namespace dphdc {
    std::vector<std::vector<bool>> HDMatrix::getVectors() {
        cl::sycl::range<2> buffer_range = this->vectors_buff.get_range();
        bool to_return_temp[buffer_range[0] * buffer_range[1]];

        {
            cl::sycl::buffer<bool, 2> to_return_temp_buff(to_return_temp, buffer_range);

            this->associated_queue.submit([&](cl::sycl::handler &h) {
                cl::sycl::accessor acc_vector(this->vectors_buff, h, cl::sycl::read_only);
                cl::sycl::accessor acc_to_return(to_return_temp_buff, h, cl::sycl::write_only);
                h.parallel_for(buffer_range, [=](cl::sycl::id<2> i) {
                    acc_to_return[i[0]][i[1]] = acc_vector[i[0]][i[1]];
                });
            });
        }

        std::vector<std::vector<bool>> to_return(buffer_range[0], std::vector<bool>(buffer_range[1]));

        for (unsigned int i = 0; i < buffer_range[0]; i++) {
            for (unsigned int j = 0; j < buffer_range[1]; j++) {
                to_return[i][j] = to_return_temp[i * buffer_range[1] + j];
            }
        }

        return to_return;
    }

    void HDMatrix::setVectors(const std::vector<std::vector<bool>> &vectors_to_set) {
        cl::sycl::range<2> buffer_range = this->vectors_buff.get_range();
        if (vectors_to_set.size() != buffer_range[0]) {
            throw std::invalid_argument("Dimensions of vector to set and HDMatrix do not match");
        }
        for (const std::vector<bool> &i: vectors_to_set) {
            if (i.size() != buffer_range[1]) {
                throw std::invalid_argument("Dimensions of vector to set and HDMatrix do not match");
            }
        }

        bool temp_to_copy[vectors_to_set.size() * vectors_to_set[0].size()];

        for (unsigned long i = 0; i < vectors_to_set.size(); i++) {
            for (unsigned long j = 0; j < vectors_to_set[0].size(); j++) {
                temp_to_copy[i * vectors_to_set[0].size() + j] = vectors_to_set[i][j];
            }
        }

        {
            cl::sycl::buffer<bool, 2> buff_to_copy(temp_to_copy, buffer_range);
            buff_to_copy.set_write_back(false);

            this->associated_queue.submit([&](cl::sycl::handler &h) {
                cl::sycl::accessor acc_to_copy(buff_to_copy, h, cl::sycl::read_only);
                cl::sycl::accessor acc_this(this->vectors_buff, h, cl::sycl::write_only);
                h.parallel_for(buffer_range, [=](cl::sycl::id<2> i) {
                    acc_this[i[0]][i[1]] = acc_to_copy[i[0]][i[1]];
                });
            });
        }
    }

    std::string HDMatrix::getAssociatedAccelerator() {
        return this->associated_queue.get_device().get_info<cl::sycl::info::device::name>();
    }

    const std::vector<std::string> &HDMatrix::getLabels() const {
        return labels;
    }

    template<class TypeOfDataToRepresent>
    const std::unordered_map<TypeOfDataToRepresent, int> &
    HDRepresentation<TypeOfDataToRepresent>::getDataTranslation() const {
        return data_translation;
    }

    void HDMatrix::setAssociatedQueue(const sycl::queue &associatedQueue) {
        associated_queue = associatedQueue;
    }
}

#include "supported_data.hpp"