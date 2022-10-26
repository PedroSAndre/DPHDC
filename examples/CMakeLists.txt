add_subdirectory(hdna/)
add_subdirectory(mnist/)
add_subdirectory(language/)
add_subdirectory(voicehd/)
add_subdirectory(cifar10/)
add_subdirectory(emg/)

add_library(readExeInputs STATIC
        readExeInputs.cpp
        readExeInputs.hpp)
target_include_directories(readExeInputs BEFORE INTERFACE .)

add_library(ResultsHandler STATIC
        ResultsHandler.cpp
        ResultsHandler.hpp)
target_include_directories(ResultsHandler BEFORE INTERFACE .)