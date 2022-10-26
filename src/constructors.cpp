#include "HDRepresentation.hpp"
#include "HDMatrix.hpp"

#include <numeric>

namespace dphdc {
    void checkVectorEntriesSize(const int &vector_size, const int &n_vectors) {
        if (vector_size <= 0 || n_vectors <= 0) {
            throw std::invalid_argument("Vector size and number of vectors to generate need to be bigger than 0");
        }
    }

    void HDMatrix::switchAcceleratorSelector(dphdc::selector accelerator_selector) {
        switch (accelerator_selector) {
            case dphdc::cpu:
                this->associated_queue = cl::sycl::queue{cl::sycl::cpu_selector()};
                break;
            case dphdc::gpu:
                this->associated_queue = cl::sycl::queue{cl::sycl::gpu_selector()};
                break;
            case dphdc::cuda:
#ifdef CUDA_CMAKE
                this->associated_queue = cl::sycl::queue{dphdc::CUDASelector()};
#else
                throw std::invalid_argument("Cannot select CUDA when library was not compiled for cuda");
#endif
                break;
            case dphdc::fpga_emulator:
#ifdef FPGA_CMAKE
                this->associated_queue = cl::sycl::queue{cl::sycl::ext::intel::fpga_emulator_selector()};
#else
                throw std::invalid_argument(
                        "Cannot select FPGA emulator when library was not compiled for this device");
#endif
                break;
            case dphdc::fpga:
#ifdef FPGA_CMAKE
                this->associated_queue = cl::sycl::queue{cl::sycl::ext::intel::fpga_selector()};
#else
                throw std::invalid_argument("Cannot select FPGA when library was not compiled for this device");
#endif
                break;
        }
    }

    void HDMatrix::switchVectorGenerator(dphdc::vectors_generator::vectors_generator vectors_type) {
        switch (vectors_type) {
            case dphdc::vectors_generator::none:
                this->constantVectorGenerator(false);
                break;
            case dphdc::vectors_generator::all_true:
                this->constantVectorGenerator(true);
                break;
            case dphdc::vectors_generator::random:
                this->randomVectorGenerator();
                break;
            case dphdc::vectors_generator::half_level:
                this->levelVectorGenerator(false);
                break;
            case dphdc::vectors_generator::full_level:
                this->levelVectorGenerator(true);
                break;
            case vectors_generator::circular:
                this->circularVectorGenerator();
                break;
        }
    }


    HDMatrix::HDMatrix(int vector_size, int n_vectors, vectors_generator::vectors_generator vectors_type,
                       dphdc::selector accelerator_selector) : vectors_buff(
            cl::sycl::range<2>(n_vectors, vector_size)) {

        checkVectorEntriesSize(vector_size, n_vectors);
        this->switchAcceleratorSelector(accelerator_selector);
        this->switchVectorGenerator(vectors_type);
    }

    HDMatrix::HDMatrix(int vector_size, int n_vectors, vectors_generator::vectors_generator vectors_type,
                       sycl::queue &q) : vectors_buff(cl::sycl::range<2>(n_vectors, vector_size)) {

        checkVectorEntriesSize(vector_size, n_vectors);
        this->associated_queue = q;
        this->switchVectorGenerator(vectors_type);
    }

    HDMatrix::HDMatrix(std::vector<HDMatrix> &matrices_to_join) : vectors_buff(cl::sycl::range<2>(0, 0)) {
        bool has_labels;
        if (matrices_to_join[0].labels.empty()) {
            has_labels = false;
        } else {
            has_labels = true;
        }

        size_t vector_size = matrices_to_join[0].vectors_buff.get_range()[1];
        std::vector<size_t> n_entries(matrices_to_join.size());
        for (size_t i = 0; i < matrices_to_join.size(); i++) {
            if (vector_size != matrices_to_join[i].vectors_buff.get_range()[1]) {
                throw std::invalid_argument("All matrices must have the same vector size");
            }
            n_entries[i] = matrices_to_join[i].vectors_buff.get_range()[0];
            if (has_labels) {
                if (matrices_to_join[i].labels.size() != n_entries[i]) {
                    throw std::invalid_argument(
                            "All matrices must have a number of labels corresponding to the matrix entries");
                }
                for (const std::string &label: matrices_to_join[i].labels) {
                    this->labels.push_back(label);
                }
            }
        }

        this->associated_queue = matrices_to_join[0].associated_queue;

        size_t sum_n_entries = std::accumulate(n_entries.begin(), n_entries.end(), static_cast<size_t>(0));
        this->vectors_buff = cl::sycl::buffer<bool, 2>(cl::sycl::range<2>(sum_n_entries, vector_size));

        size_t current_general_index = 0;
        for (size_t i = 0; i < matrices_to_join.size(); i++) {
            this->associated_queue.submit([&](cl::sycl::handler &h) {
                cl::sycl::accessor acc_vector_this(this->vectors_buff, h, cl::sycl::write_only);
                cl::sycl::accessor acc_vector_matrix_to_copy(matrices_to_join[i].vectors_buff, h, cl::sycl::read_only);
                h.parallel_for(matrices_to_join[i].vectors_buff.get_range(), [=](cl::sycl::id<2> local_range) {
                    size_t j = local_range[0];
                    size_t k = local_range[1];
                    acc_vector_this[j + current_general_index][k] = acc_vector_matrix_to_copy[j][k];
                });
            });

            current_general_index += n_entries[i];
        }
    }

    HDMatrix::HDMatrix(const std::string &full_file_path, dphdc::selector accelerator_selector) : vectors_buff(
            cl::sycl::range<2>(0, 0)) {
        this->switchAcceleratorSelector(accelerator_selector);

        std::ifstream file;
        try {
            file.open(full_file_path, std::ios::binary | std::ios::in);
        } catch (const std::exception &ex) {
            throw ex;
        }

        if (!file.is_open()) {
            throw std::invalid_argument("Cannot open file: " + full_file_path);
        }

        this->readBuffer(file);
        this->readLabels(file);

        file.close();
    }

    HDMatrix::HDMatrix(const std::string &full_file_path, cl::sycl::queue &q) : vectors_buff(cl::sycl::range<2>(0, 0)) {
        this->associated_queue = q;

        std::ifstream file;
        try {
            file.open(full_file_path, std::ios::binary | std::ios::in);
        } catch (const std::ifstream::failure &ex) {
            throw ex;
        }

        this->readBuffer(file);
        this->readLabels(file);

        file.close();
    }

    template<class TypeOfDataToRepresent>
    void generateHashTable(std::unordered_map<TypeOfDataToRepresent, int> &map,
                           const std::vector<TypeOfDataToRepresent> &elements_to_represent) {
        for (unsigned int i = 1; i <= elements_to_represent.size(); i++) {
            map[elements_to_represent[i - 1]] = i;
        }
    }

    template<class TypeOfDataToRepresent>
    HDRepresentation<TypeOfDataToRepresent>::HDRepresentation(int vector_size,
                                                              vectors_generator::vectors_generator vectors_type,
                                                              dphdc::selector accelerator_selector,
                                                              const std::vector<TypeOfDataToRepresent> &elements_to_represent)
            : HDMatrix(vector_size, elements_to_represent.size(), vectors_type, accelerator_selector) {
        generateHashTable(this->data_translation, elements_to_represent);
    }

    template<class TypeOfDataToRepresent>
    HDRepresentation<TypeOfDataToRepresent>::HDRepresentation(int vector_size,
                                                              vectors_generator::vectors_generator vectors_type,
                                                              sycl::queue &q,
                                                              const std::vector<TypeOfDataToRepresent> &elements_to_represent)
            : HDMatrix(vector_size, elements_to_represent.size(), vectors_type, q) {
        generateHashTable(this->data_translation, elements_to_represent);
    }

    template<class TypeOfDataToRepresent>
    HDRepresentation<TypeOfDataToRepresent>::HDRepresentation(HDMatrix &to_copy,
                                                              const std::vector<TypeOfDataToRepresent> &elements_to_represent)
            : HDMatrix(to_copy) {
        if (this->vectors_buff.get_range()[0] != elements_to_represent.size()) {
            throw std::invalid_argument(
                    "Number of vectors in HDMatrix needs to be the same as in elements to represent");
        }

        generateHashTable(this->data_translation, elements_to_represent);
    }

    template<class TypeOfDataToRepresent>
    HDRepresentation<TypeOfDataToRepresent>::HDRepresentation(const std::string &full_file_path,
                                                              dphdc::selector accelerator_selector) : HDMatrix(1, 1,
                                                                                                               vectors_generator::none,
                                                                                                               accelerator_selector) {
        this->switchAcceleratorSelector(accelerator_selector);

        std::ifstream file;
        try {
            file.open(full_file_path, std::ios::binary | std::ios::in);
        } catch (const std::exception &ex) {
            throw ex;
        }

        if (!file.is_open()) {
            throw std::invalid_argument("Cannot open file: " + full_file_path);
        }

        this->readBuffer(file);
        this->readDataTranslation(file);

        file.close();
    }

    template<class TypeOfDataToRepresent>
    HDRepresentation<TypeOfDataToRepresent>::HDRepresentation(const std::string &full_file_path, sycl::queue &q)
            : HDMatrix(1, 1, vectors_generator::none, q) {
        this->associated_queue = q;

        std::ifstream file;
        try {
            file.open(full_file_path, std::ios::binary | std::ios::in);
        } catch (const std::ifstream::failure &ex) {
            throw ex;
        }

        this->readBuffer(file);
        this->readDataTranslation(file);

        file.close();
    }
}

#include "supported_data.hpp"