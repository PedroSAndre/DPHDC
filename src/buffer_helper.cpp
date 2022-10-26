#include "HDMatrix.hpp"

namespace dphdc {
    void HDMatrix::copyBuffer(cl::sycl::buffer<bool, 2> &origin, cl::sycl::buffer<bool, 2> &dest) {
        if (dest.get_range() != origin.get_range()) {
            throw std::invalid_argument("Cannot copy buffer with different range");
        }
        this->associated_queue.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc_origin(origin, h, cl::sycl::read_only);
            cl::sycl::accessor acc_dest(dest, h, cl::sycl::write_only);
            h.parallel_for(this->vectors_buff.get_range(), [=](cl::sycl::id<2> local_range) {
                size_t i = local_range[0];
                size_t k = local_range[1];
                acc_dest[i][k] = acc_origin[i][k];
            });
        });
    }

    cl::sycl::buffer<bool, 2> HDMatrix::copyBuffer(sycl::buffer<bool, 2> &origin) {
        cl::sycl::buffer<bool, 2> to_return(origin.get_range());

        this->associated_queue.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc_dest(to_return, h, cl::sycl::write_only);
            cl::sycl::accessor acc_origin(origin, h, cl::sycl::read_only);
            h.parallel_for(origin.get_range(), [=](cl::sycl::id<2> i) {
                acc_dest[i[0]][i[1]] = acc_origin[i[0]][i[1]];
            });
        });

        return to_return;
    }

    cl::sycl::buffer<short int, 2> HDMatrix::generateInitializeAccumulators(sycl::range<2> range) {
        cl::sycl::buffer<short int, 2> accumulators_to_return(range);

        this->associated_queue.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc_accumulators(accumulators_to_return, h, cl::sycl::write_only);
            h.parallel_for(range, [=](cl::sycl::id<2> i) {
                acc_accumulators[i[0]][i[1]] = 0;
            });
        });

        return accumulators_to_return;
    }

    void HDMatrix::normalizeAccumulator(sycl::buffer<short int, 2> &accumulators, sycl::buffer<bool, 2> &destination) {
        if (accumulators.get_range() != destination.get_range()) {
            throw std::invalid_argument("Accumulators and destination need to have same range");
        }
        this->associated_queue.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc_result(destination, h, cl::sycl::write_only);
            cl::sycl::accessor acc_accumulators(accumulators, h, cl::sycl::read_only);
            h.parallel_for(accumulators.get_range(), [=](cl::sycl::id<2> local_range) {
                size_t i = local_range[0];
                size_t k = local_range[1];
                if (acc_accumulators[i][k] >= 0) {
                    acc_result[i][k] = true;
                } else {
                    acc_result[i][k] = false;
                }
            });
        });
    }
}
