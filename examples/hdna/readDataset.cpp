#include "readDataset.hpp"
#include <fstream>

#define SPECIES_NAME_START 10
#define N_BASES_PER_GENE 657

std::pair<std::vector<std::vector<char>>, std::vector<std::string>> readDataset(const std::string &full_path_to_file) {
    std::vector<std::string> labels = {};

    unsigned int start;
    unsigned int finish;

    std::string line_output;
    std::ifstream file(full_path_to_file);

    std::vector<std::string> genes_string;
    genes_string.emplace_back("");
    labels.emplace_back("");

    while (getline(file, line_output)) {
        start = line_output.find('|');
        finish = line_output.find('|', SPECIES_NAME_START);
        labels.back() = line_output.substr(start + 1, finish - start - 1);

        getline(file, line_output);
        genes_string.back() = line_output.substr(0, N_BASES_PER_GENE);

        genes_string.emplace_back("");
        labels.emplace_back("");
    }
    file.close();
    genes_string.pop_back();
    labels.pop_back();

    std::vector<std::vector<char>> genes_data(genes_string.size(), std::vector<char>(genes_string[0].size()));

    for (unsigned int i = 0; i < genes_string.size(); i++) {
        for (unsigned int j = 0; j < genes_string[i].size(); j++) {
            genes_data[i][j] = genes_string[i][j];
        }
    }

    std::pair<std::vector<std::vector<char>>, std::vector<std::string>> data(genes_data, labels);

    return data;
}