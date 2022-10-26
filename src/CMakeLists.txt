set(dphdc_cpp constructors.cpp
        buffer_helper.cpp
        permutations.cpp
        vector_generators.cpp
        reductions.cpp
        sets_gets.cpp
        encoders.cpp
        query.cpp
        storing_reading.cpp)
set(dphdc_hpp HDMatrix.hpp
        HDRepresentation.hpp
        selectors.hpp
        supported_data.hpp)

add_library(dphdc STATIC ${dphdc_cpp} ${dphdc_hpp})
#pybind11_add_module(pydphdc pybind.cpp ${dphdc_cpp} ${dphdc_hpp})

target_include_directories(dphdc BEFORE INTERFACE libraryheaders/)