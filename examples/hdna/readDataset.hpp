#ifndef DPHDC_READDATASET_HPP
#define DPHDC_READDATASET_HPP

#include <vector>
#include <utility>
#include <string>

std::pair<std::vector<std::vector<char>>, std::vector<std::string>> readDataset(const std::string &full_path_to_file);


#endif //DPHDC_READDATASET_HPP
