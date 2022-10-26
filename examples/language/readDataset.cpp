#include "readDataset.hpp"
#include <fstream>


Data readDataset(const int &n, const std::string &full_path_to_folder) {
    std::vector<std::string> language_names = {"Bulgarian", "Czech", "Danish", "German", "Greek", "English", "Estonian",
                                               "Finnish", "French", "Hungarian", "Italian", "Latvian", "Lithuanian",
                                               "Dutch", "Polish", "Portuguese", "Romanian", "Slovak", "Slovenian",
                                               "Spanish", "Swedish"};

    std::vector<std::string> file_names = {"bul.txt", "ces.txt", "dan.txt", "deu.txt", "ell.txt", "eng.txt", "est.txt",
                                           "fin.txt", "fra.txt", "hun.txt", "ita.txt", "lav.txt", "lit.txt", "nld.txt",
                                           "pol.txt", "por.txt", "ron.txt", "slk.txt", "slv.txt", "spa.txt", "swe.txt"};

    Data to_return;

    std::string line_output;
    std::string line_output_no_spaces;

    for (unsigned int i = 0; i < file_names.size(); i++) {
        std::ifstream file(full_path_to_folder + file_names[i]);

        if (file.is_open()) {
            while (getline(file, line_output)) {
                if (line_output.find_first_not_of(' ') != std::string::npos) {
                    line_output_no_spaces = removeSpaces(line_output);
                    if (line_output_no_spaces.size() >= n) {
                        to_return.sentence_data.push_back(splitString(n, line_output_no_spaces));
                        to_return.labels_data.push_back(language_names[i]);
                    }
                }
            }
        }
        file.close();
    }

    return to_return;
}

std::string removeSpaces(const std::string &string_to_remove_spaces) {
    std::string string_no_spaces;

    for (size_t i = string_to_remove_spaces.find_first_not_of(' ');
         i <= string_to_remove_spaces.find_last_not_of(' ');) {
        if (string_to_remove_spaces[i] == ' ') {
            while (string_to_remove_spaces[i + 1] == ' ')
                i++;
        }
        string_no_spaces += string_to_remove_spaces[i++];
    }

    return string_no_spaces;
}

std::vector<std::string> splitString(const int &n, const std::string &string_to_convert) {
    std::vector<std::string> split_string;

    for (unsigned int i = 0; i <= string_to_convert.size() - n; i++) {
        split_string.push_back(string_to_convert.substr(i, n));
    }

    return split_string;
}