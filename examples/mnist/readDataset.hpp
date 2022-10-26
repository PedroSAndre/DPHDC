#ifndef DPHDC_READDATASET_HPP
#define DPHDC_READDATASET_HPP

#include <vector>
#include <string>

struct Data {
    std::vector<std::vector<unsigned char>> image_data;
    std::vector<std::string> labels_data;
    unsigned int image_size = 0;
    unsigned int number_of_images = 0;
    unsigned int number_of_labels = 0;
};

Data readDataset(const std::string &full_path_to_file, const std::string &full_path_labels_file);


#endif //DPHDC_READDATASET_HPP
