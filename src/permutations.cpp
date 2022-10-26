#include "HDMatrix.hpp"

namespace dphdc {
    void
    HDMatrix::shiftRight(cl::sycl::buffer<bool, 2> &buffer_to_shift, cl::sycl::buffer<bool, 2> &duplicate_of_buffer) {
        if (buffer_to_shift.get_range() != duplicate_of_buffer.get_range()) {
            throw std::invalid_argument("Cannot shift vectors with a duplicate of different size");
        }
        size_t vector_size = buffer_to_shift.get_range()[1];
        this->associated_queue.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc_buffer_shift(buffer_to_shift, h, cl::sycl::write_only);
            cl::sycl::accessor acc_duplicate(duplicate_of_buffer, h, cl::sycl::read_only);
            h.parallel_for(this->vectors_buff.get_range(), [=](cl::sycl::id<2> local_range) {
                size_t i = local_range[0];
                size_t k = local_range[1];
                if (k == 0) {
                    acc_buffer_shift[i][k] = acc_duplicate[i][vector_size - 1];
                } else {
                    acc_buffer_shift[i][k] = acc_duplicate[i][k - 1];
                }
            });
        });

        this->copyBuffer(buffer_to_shift, duplicate_of_buffer);
    }
}