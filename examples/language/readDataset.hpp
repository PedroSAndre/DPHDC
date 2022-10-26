#ifndef DPHDC_READDATASET_HPP
#define DPHDC_READDATASET_HPP

#include <vector>
#include <string>

struct Data {
    std::vector<std::vector<std::string>> sentence_data;
    std::vector<std::string> labels_data;
};

std::vector<std::string> splitString(const int &n, const std::string &string_to_convert);

std::string removeSpaces(const std::string &string_to_remove_spaces);

Data readDataset(const int &n, const std::string &full_path_to_folder);


#endif //DPHDC_READDATASET_HPP
