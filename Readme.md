# DPHDC: A DataParallel framework for Hyperdimensional Computing
Data Parallel framework for Hyperdimensional Computing (DPHDC) is developed with  the aim of efficiently and robustly running classification tasks based on HDC across devices of different architectures while fully exploiting their processing capabilities. For this purpose, DPHDC was developed using the C++ programming language and the cross-platform SYCL abstraction layer (2020 specification).
Another goal of DPHDC is to also ensure an intuitive and easy to
use interface, both for beginner and experienced users, despite the considerable range of devices it can support.

It is also worth emphasizing that DPHDC is designed to be compatible with any SYCL-capable compiler. To ensure that DPHDC works with a particular setup, unit tests were developed and are provided with the library. These can be executed to ensure that the behaviour of all implemented functionalities is as expected.

## Documentation
Unfortunately, proper documentation for the library is not yet available due to time constraints. This should be fixed in the upcoming weeks.

## How to compile the library and run the provided classification examples/unit tests
Currently, DPHDC has only been tested on Ubuntu 20.04 LTS and Ubuntu 22.04 LTS using the Intel DPC++ compiler, although, as mentioned previously, it should be compatible with any SYCL-capable compiler.

### Dependencies
The only necessary tool to compile the DPHDC library is to have a SYCL-capable compiler installed.
The Intel DPC++ compiler can be downloaded through the Intel OneAPI base toolkit (available [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)). The most current version, at the time of writing, can be downloaded with the following command:

```
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18852/l_BaseKit_p_2022.3.0.8767.sh
```

After downloading the installation can start by using the following command:

```
sudo sh ./l_BaseKit_p_2022.3.0.8767.sh
```

After that, it is just necessary to follow the GUI installation prompts.
It is only necessary to install the Intel DPC++ compiler, all other features are optional.
It is highly recommended that the compiler is installed in the default location, i.e., at /opt/intel/oneapi/ .

It is also necessary to have CMake version 3.10 or above. It can be easily installed using 
```
sudo apt install cmake
```
### Compiling and Running
The first step before setting up and compiling the library is sourcing the Intel DPC++ compiler environment.
It is necessary to repeat this step every time the bash environment is reset.
If the compiler is installed in the default path, such can be achieved by

```
source /opt/intel/oneapi/setvars.sh
```

After cloning the library repository, the set-up of the CMake project can be done with the following command:

```
 cmake . -DACCELERATOR_FLAG=[ACCELERATOR] -DCMAKE_BUILD_TYPE=Release
```
where *[ACCELERATOR]* indicates the device intended to be targeted by the library. Currently available options are:

- *CPU* - For targeting any CPU;
- *GPU* - For targeting Intel GPUs;
- *CUDA* - For targeting CUDA compatible GPUs (it is necessary to have CUDA 11.6 or later installed);
- *FPGAEMU* - For targeting an FPGA emulator, that runs on the CPU;
- *ARRIA10GX* - For targeting the Intel Arria 10 GX FPGA card.

It is then possible to compile the unit tests by

```
make tests
```

To run them, it is necessary to change the current directory to the tests directory to then execute the application itself:

```
cd tests
./tests
cd ../
```

To compile and execute any example the process is identical. To compile:

```
make [EXAMPLE]
```

To run:
```
cd examples/[EXAMPLE]/
./[EXAMPLE] -vs [VECTORSIZE]
cd ../../
```

where *[VECTORSIZE]* indicates the hypervector size to use (>0) and *[EXAMPLE]* indicates the example to be compiled and executed. Currently available options are:

- *voicehd* - VoiceHD speech recognition application;
- *mnist* - Image classification of the MNIST dataset;
- *language* - European language recognition application;
- *hdna* - HDNA genome sequencing application;
- *emg* - Gesture recognition application.

After the successful execution of an example application, the expected outcome is as follows:

>Results\
Library Version: v0.3\
Finished at: Wed Oct 26 02:14:56 2022\
Accelerator: Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz\
Success Rate: 87.5561%\
Training Time: 6.2806s\
Testing Time: 1.94374s\
Vector Size: 10000