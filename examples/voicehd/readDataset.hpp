#ifndef DPHDC_READDATASET_HPP
#define DPHDC_READDATASET_HPP

#include <vector>
#include <string>

struct dataset {
    std::vector<std::vector<int>> data;
    std::vector<std::string> labels;
};

dataset readDataset(const std::string &full_path_to_file);

#endif //DPHDC_READDATASET_HPP
