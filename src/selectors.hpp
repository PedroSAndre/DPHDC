#ifndef DPHDC_SELECTORS_HPP
#define DPHDC_SELECTORS_HPP

#include <CL/sycl.hpp>

#ifdef FPGA_CMAKE

#include <sycl/ext/intel/fpga_extensions.hpp>

#endif //FPGA_CMAKE

namespace dphdc {
    enum selector {
        cpu,
        gpu,
        cuda,
        fpga_emulator,
        fpga
    };

#ifdef CUDA_CMAKE
    class [[maybe_unused]] CUDASelector : public sycl::device_selector {
    public:
        int operator()(const sycl::device &device) const override {
            if (device.get_platform().get_backend() == sycl::backend::ext_oneapi_cuda) {
                return 1;
            } else {
                return -1;
            }
        }
    };

#endif
}

#endif //DPHDC_SELECTORS_HPP
