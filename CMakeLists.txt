### Basic Project info ###
cmake_minimum_required(VERSION 3.10.0)
set(CMAKE_CXX_STANDARD 20)
project(DPHDC)

### Project Version ###
add_definitions(-DPROJECT_VERSION_CMAKE=\"0.3\")

### Adding threads
#set(THREADS_PREFER_PTHREAD_FLAG ON)
#find_package(Threads REQUIRED)

### Downloading Google Test and making it available
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

#### Downloading Pybind Test and making it available
#FetchContent_Declare(
#        pybind11
#        GIT_REPOSITORY https://github.com/pybind/pybind11.git
#        GIT_TAG v2.10.0
#)
#set(PYBIND11_PYTHON_VERSION 3.8 CACHE STRING "")
#FetchContent_MakeAvailable(pybind11)

### Defining the compiler depending on device to be used ###
IF (ACCELERATOR_FLAG MATCHES "CUDA")
    set(CMAKE_CXX_COMPILER "clang++")
ELSE ()
    set(CMAKE_CXX_COMPILER "dpcpp")
ENDIF ()

### Defining ahead of time compilation flags and device selectors ###
IF (ACCELERATOR_FLAG MATCHES "CPU")
    set(COMPILER_FLAGS "-fsycl-targets=spir64_x86_64")
    add_definitions(-DACCELERATOR_CMAKE_DPHDC=dphdc::cpu)
    add_definitions(-DACCELERATOR_CMAKE_QUEUE=cl::sycl::cpu_selector)
ELSEIF (ACCELERATOR_FLAG MATCHES "GPU")
    set(COMPILER_FLAGS "-fsycl-targets=spir64_gen -Xs \"-device *\"")
    add_definitions(-DACCELERATOR_CMAKE_DPHDC=dphdc::gpu)
    add_definitions(-DACCELERATOR_CMAKE_QUEUE=cl::sycl::gpu_selector)
ELSEIF (ACCELERATOR_FLAG MATCHES "CUDA")
    set(COMPILER_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda")
    add_definitions(-DACCELERATOR_CMAKE_DPHDC=dphdc::cuda)
    add_definitions(-DACCELERATOR_CMAKE_QUEUE=dphdc::CUDASelector)
    add_definitions(-DCUDA_CMAKE=1)
ELSEIF (ACCELERATOR_FLAG MATCHES "FPGAEMU")
    set(COMPILER_FLAGS "-fintelfpga -DFPGA_EMULATOR")
    add_definitions(-DFPGA_CMAKE=1)
    add_definitions(-DACCELERATOR_CMAKE_QUEUE=cl::sycl::ext::intel::fpga_emulator_selector)
    add_definitions(-DACCELERATOR_CMAKE_DPHDC=dphdc::fpga_emulator)
ELSEIF (ACCELERATOR_FLAG MATCHES "ARRIA10GX")
    set(COMPILER_FLAGS "-fintelfpga -Xshardware -Xsboard=intel_a10gx_pac:pac_a10")
    add_definitions(-DFPGA_CMAKE=1)
    add_definitions(-DACCELERATOR_CMAKE_QUEUE=cl::sycl::ext::intel::fpga_selector)
    add_definitions(-DACCELERATOR_CMAKE_DPHDC=dphdc::fpga)
ELSEIF (ACCELERATOR_FLAG MATCHES "OPTIMIZATION")
    set(COMPILER_FLAGS "-fintelfpga -Xshardware -fsycl-link=early -Xsboard=intel_a10gx_pac:pac_a10")
    add_definitions(-DFPGA_CMAKE=1)
    add_definitions(-DACCELERATOR_CMAKE_QUEUE=cl::sycl::ext::intel::fpga_selector)
    add_definitions(-DACCELERATOR_CMAKE_DPHDC=dphdc::fpga)
ELSE ()
    set(COMPILER_FLAGS "")
    add_definitions(-DACCELERATOR_CMAKE_DPHDC=dphdc::cpu)
    add_definitions(-DACCELERATOR_CMAKE_QUEUE=cl::sycl::cpu_selector)
ENDIF ()

set(CMAKE_CXX_FLAGS "${COMPILER_FLAGS}")

### Making project path available in code ###
add_definitions(-DPROJECT_PATH_CMAKE=\"${PROJECT_SOURCE_DIR}\")


### Setting optimization flags ###
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

### Adding directories ###
add_subdirectory(src/)
add_subdirectory(tests/)
add_subdirectory(examples/)