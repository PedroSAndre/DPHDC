#include "readExeInputs.hpp"
#include <cstring>
#include <string>
#include <stdexcept>

Inputs readExeInputs(int argc, char **argv) {
    Inputs inputs_read;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-vs") == 0) {
            if (argc > i + 1) {
                i++;
                inputs_read.vector_size = std::stoi(argv[i]);
            }
        }
        if (std::strcmp(argv[i], "-ng") == 0) {
            if (argc > i + 1) {
                i++;
                inputs_read.n_gram = std::stoi(argv[i]);
            }
        }
    }

    if (inputs_read.vector_size <= 0) {
        throw std::invalid_argument("Vector size cannot be smaller than or equal to 0");
    }
    if (inputs_read.n_gram <= 0) {
        throw std::invalid_argument("N gram size cannot be smaller than or equal to 0");
    }

    return inputs_read;
}