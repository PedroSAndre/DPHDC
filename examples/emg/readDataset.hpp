#ifndef DPHDC_READDATASET_HPP
#define DPHDC_READDATASET_HPP

#include <vector>
#include <string>

struct struc_data {
    std::vector<std::vector<int>> train_data;
    std::vector<std::vector<int>> test_data;
};

struct struc_labels {
    std::vector<std::string> train_labels;
    std::vector<std::string> test_labels;
};

struct dataset {
    struc_data data;
    struc_labels labels;
};

dataset readDataset(const std::string &full_path_to_dataset_folder, uint8_t subject_n);

#endif //DPHDC_READDATASET_HPP
