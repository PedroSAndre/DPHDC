#include "HDRepresentation.hpp"

const std::string MATRIX_EXTENSION = ".dphdcm";
const std::string REPRESENTATION_EXTENSION = ".dphdcr";

namespace dphdc {
    void HDMatrix::saveBuffer(std::ofstream &file) {
        std::vector<std::vector<bool>> buffer_vector = this->getVectors();

        size_t n_entries = buffer_vector.size();
        size_t vector_size = buffer_vector[0].size();
        try {
            file.write(reinterpret_cast<const char *>(&n_entries), sizeof n_entries);
            file.write(reinterpret_cast<const char *>(&vector_size), sizeof vector_size);

            for (const auto &i: buffer_vector) {
                for (const auto &j: i) {
                    file.write(reinterpret_cast<const char *>(&j), sizeof j);
                }
            }
        }
        catch (const std::ofstream::failure &ex) {
            throw ex;
        }
    }

    void HDMatrix::saveLabels(std::ofstream &file) {
        size_t n_labels = this->labels.size();

        try {
            file.write(reinterpret_cast<const char *>(&n_labels), sizeof n_labels);
            ssize_t string_size{};
            for (const auto &i: this->labels) {
                string_size = static_cast<ssize_t>(i.size()) + 1;
                file.write(reinterpret_cast<const char *>(&string_size), sizeof string_size);
                file.write(i.c_str(), string_size);
            }
        }
        catch (const std::ofstream::failure &ex) {
            throw ex;
        }
    }

    void HDMatrix::storeMatrix(std::string full_file_path) {
        full_file_path.append(MATRIX_EXTENSION);
        std::ofstream file;
        try {
            file.open(full_file_path, std::ios::binary | std::ios::out | std::ios::trunc);
        }
        catch (const std::ofstream::failure &ex) {
            throw ex;
        }

        if (!file.is_open()) {
            throw std::invalid_argument("Cannot open file: " + full_file_path);
        }

        this->saveBuffer(file);
        this->saveLabels(file);

        file.close();
    }

    void HDMatrix::readBuffer(std::ifstream &file) {
        size_t n_entries;
        size_t vector_size;

        try {
            file.read(reinterpret_cast<char *>(&n_entries), sizeof n_entries);
            file.read(reinterpret_cast<char *>(&vector_size), sizeof vector_size);
        }
        catch (const std::ifstream::failure &ex) {
            throw ex;
        }

        std::unique_ptr<bool[]> data_read(new bool[n_entries * vector_size]);

        try {
            size_t aux = 0;
            for (size_t i = 0; i < n_entries; i++) {
                for (size_t j = 0; j < vector_size; j++) {
                    file.read(reinterpret_cast<char *>(&(data_read[aux])), sizeof(bool));
                    aux++;
                }
            }
        }
        catch (const std::ifstream::failure &ex) {
            throw ex;
        }

        this->vectors_buff = cl::sycl::buffer<bool, 2>(cl::sycl::range<2>(n_entries, vector_size));
        this->copyBoolVector(data_read.get());
    }

    void HDMatrix::readLabels(std::ifstream &file) {
        this->labels = {};
        size_t n_labels;

        try {
            file.read(reinterpret_cast<char *>(&n_labels), sizeof n_labels);
        }
        catch (const std::ifstream::failure &ex) {
            throw ex;
        }

        this->labels.reserve(n_labels);
        try {
            ssize_t string_size = 0;
            std::unique_ptr<char[]> temp_string;
            for (size_t i = 0; i < n_labels; i++) {
                file.read(reinterpret_cast<char *>(&string_size), sizeof string_size);

                temp_string = std::unique_ptr<char[]>(new char[string_size]);
                file.read(temp_string.get(), string_size);

                this->labels.emplace_back(temp_string.get());
            }
        }
        catch (const std::ifstream::failure &ex) {
            throw ex;
        }
    }

    template<class TypeOfDataToRepresent>
    void HDRepresentation<TypeOfDataToRepresent>::saveDataTranslation(std::ofstream &file) {
        try {
            size_t n_data_entries = this->data_translation.size();
            size_t size_of_data_to_represent = sizeof(TypeOfDataToRepresent);
            file.write(reinterpret_cast<const char *>(&n_data_entries), sizeof n_data_entries);
            file.write(reinterpret_cast<const char *>(&size_of_data_to_represent), sizeof size_of_data_to_represent);

            TypeOfDataToRepresent key;
            int value;
            for (const auto &i: this->data_translation) {
                key = i.first;
                value = i.second;
                file.write(reinterpret_cast<const char *>(&key), sizeof key);
                file.write(reinterpret_cast<const char *>(&value), sizeof value);
            }
        }
        catch (const std::ofstream::failure &ex) {
            throw ex;
        }
    }

    template<class TypeOfDataToRepresent>
    void HDRepresentation<TypeOfDataToRepresent>::storeRepresentation(std::string full_file_path) {
        full_file_path.append(REPRESENTATION_EXTENSION);
        std::ofstream file;
        try {
            file.open(full_file_path, std::ios::binary | std::ios::out | std::ios::trunc);
        }
        catch (const std::ofstream::failure &ex) {
            throw ex;
        }

        if (!file.is_open()) {
            throw std::invalid_argument("Cannot open file: " + full_file_path);
        }

        this->saveBuffer(file);
        this->saveDataTranslation(file);

        file.close();
    }


    template<class TypeOfDataToRepresent>
    void HDRepresentation<TypeOfDataToRepresent>::readDataTranslation(std::ifstream &file) {
        this->data_translation = {};
        size_t n_data_entries;
        size_t size_of_data_to_represent;

        try {
            file.read(reinterpret_cast<char *>(&n_data_entries), sizeof n_data_entries);
            file.read(reinterpret_cast<char *>(&size_of_data_to_represent), sizeof size_of_data_to_represent);
        }
        catch (const std::ofstream::failure &ex) {
            throw ex;
        }

        if (size_of_data_to_represent != sizeof(TypeOfDataToRepresent)) {
            throw std::invalid_argument("Cannot load a representation of a different type");
        }

        try {
            TypeOfDataToRepresent key;
            int value;
            for (size_t i = 0; i < n_data_entries; i++) {
                file.read(reinterpret_cast<char *>(&key), sizeof key);
                file.read(reinterpret_cast<char *>(&value), sizeof value);
                this->data_translation[key] = value;
            }
        }
        catch (const std::ofstream::failure &ex) {
            throw ex;
        }
    }
}

#include "supported_data.hpp"
