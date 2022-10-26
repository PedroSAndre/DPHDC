#include "readDataset.hpp"
#include <fstream>
#include <random>
#include <sstream>

const float training_percentage = 0.7;
const int downsampling_rate = 250;
const std::vector<std::string> data_files_names = {"COMPLETE_1.csv",
                                                   "COMPLETE_2.csv",
                                                   "COMPLETE_3.csv",
                                                   "COMPLETE_4.csv",
                                                   "COMPLETE_5.csv"};
const std::vector<std::string> labels_files_names = {"LABEL_1.csv",
                                                     "LABEL_2.csv",
                                                     "LABEL_3.csv",
                                                     "LABEL_4.csv",
                                                     "LABEL_5.csv",};

std::vector<size_t> shuffle_indexes;

void initializeShuffleIndexes(size_t size) {
    if (shuffle_indexes.empty()) {
        shuffle_indexes.reserve(size);
        for (size_t i = 0; i < size; i++) {
            shuffle_indexes.push_back(i);
        }

        std::shuffle(shuffle_indexes.begin(), shuffle_indexes.end(), std::mt19937(std::random_device()()));
    }
}

struc_data readData(const std::string &full_path_to_dataset_folder, uint8_t subject_n) {
    std::ifstream file(full_path_to_dataset_folder + "/" + data_files_names[subject_n]);
    std::string line_read;
    std::istringstream stream_line_read;
    std::string value_string;
    std::vector<float> freq_values;
    std::vector<std::vector<int>> all_data;
    std::vector<std::vector<int>> selected_data;
    struc_data to_return;


    while (std::getline(file, line_read)) {
        freq_values = {};

        stream_line_read = std::istringstream(line_read);
        while (std::getline(stream_line_read, value_string, ',')) {
            freq_values.push_back(stof(value_string));
        }

        all_data.emplace_back();
        for (const float &i: freq_values) {
            int aux = static_cast<int>(i);
            all_data.back().push_back(aux);
        }
    }

    file.close();

    for (size_t i = 0; i < all_data.size(); i = i + downsampling_rate) {
        selected_data.push_back(all_data[i]);
    }

    initializeShuffleIndexes(selected_data.size());

    std::vector<std::vector<int>> shuffled_selected_data(selected_data.size(),
                                                         std::vector<int>(selected_data[0].size()));
    for (size_t i = 0; i < shuffled_selected_data.size(); i++) {
        for (size_t j = 0; j < shuffled_selected_data[0].size(); j++) {
            shuffled_selected_data[i][j] = selected_data[shuffle_indexes[i]][j];
        }
    }

    long int n_train_entries = static_cast<long int>(static_cast<float>(shuffled_selected_data.size()) *
                                                     training_percentage);


    to_return.train_data = std::vector<std::vector<int>>(shuffled_selected_data.begin(),
                                                         shuffled_selected_data.begin() + n_train_entries);
    to_return.test_data = std::vector<std::vector<int>>(shuffled_selected_data.begin() + n_train_entries,
                                                        shuffled_selected_data.end());

    return to_return;
}

struc_labels readLabels(const std::string &full_path_to_dataset_folder, uint8_t subject_n) {
    std::ifstream file(full_path_to_dataset_folder + "/" + labels_files_names[subject_n]);
    std::string line_read;
    std::vector<std::string> all_data;
    std::vector<std::string> selected_data;
    struc_labels to_return;


    while (std::getline(file, line_read)) {
//        line_read.pop_back();
        all_data.emplace_back(line_read);
    }

    file.close();

    for (size_t i = 0; i < all_data.size(); i = i + downsampling_rate) {
        selected_data.push_back(all_data[i]);
    }

    initializeShuffleIndexes(selected_data.size());

    std::vector<std::string> shuffled_selected_data(selected_data.size());
    for (size_t i = 0; i < shuffled_selected_data.size(); i++) {
        shuffled_selected_data[i] = selected_data[shuffle_indexes[i]];
    }

    long int n_train_entries = static_cast<long int>(static_cast<float>(shuffled_selected_data.size()) *
                                                     training_percentage);


    to_return.train_labels = std::vector<std::string>(shuffled_selected_data.begin(),
                                                      shuffled_selected_data.begin() + n_train_entries);
    to_return.test_labels = std::vector<std::string>(shuffled_selected_data.begin() + n_train_entries,
                                                     shuffled_selected_data.end());

    return to_return;
}

dataset readDataset(const std::string &full_path_to_dataset_folder, uint8_t subject_n) {
    dataset to_return{};

    to_return.data = readData(full_path_to_dataset_folder, subject_n);
    to_return.labels = readLabels(full_path_to_dataset_folder, subject_n);

    shuffle_indexes = {};
    return to_return;
}