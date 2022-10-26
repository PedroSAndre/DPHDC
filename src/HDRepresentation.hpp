#ifndef DPHDC_HDREPRESENTATION_HPP
#define DPHDC_HDREPRESENTATION_HPP

#include "HDMatrix.hpp"


namespace dphdc {

    template<class TypeOfDataToRepresent>
    class HDRepresentation : public HDMatrix {
    protected:
        std::unordered_map<TypeOfDataToRepresent, int> data_translation;

        std::unique_ptr<int[]>
        convertData(const std::vector<std::vector<TypeOfDataToRepresent>> &data, int &max_element_size);

        void saveDataTranslation(std::ofstream &file);

        void readDataTranslation(std::ifstream &file);

    public:
        HDRepresentation(int vector_size, vectors_generator::vectors_generator vectors_type,
                         selector accelerator_selector,
                         const std::vector<TypeOfDataToRepresent> &elements_to_represent);

        HDRepresentation(int vector_size, vectors_generator::vectors_generator vectors_type, cl::sycl::queue &q,
                         const std::vector<TypeOfDataToRepresent> &elements_to_represent);

        HDRepresentation(HDMatrix &to_copy, const std::vector<TypeOfDataToRepresent> &elements_to_represent);

        HDRepresentation(const std::string &full_file_path, dphdc::selector accelerator_selector);

        HDRepresentation(const std::string &full_file_path, cl::sycl::queue &q);


        HDMatrix encodeWithBundle(const std::vector<std::vector<TypeOfDataToRepresent>> &data,
                                  permutation::permutation permutation_to_use);

        HDMatrix encodeWithXOR(const std::vector<std::vector<TypeOfDataToRepresent>> &data, HDMatrix &position_vectors);

        HDMatrix encodeWithXOR(const std::vector<std::vector<TypeOfDataToRepresent>> &data,
                               permutation::permutation permutation_to_use);

        void storeRepresentation(std::string full_file_path);

        const std::unordered_map<TypeOfDataToRepresent, int> &getDataTranslation() const;
    };
}


#endif //DPHDC_HDREPRESENTATION_HPP
