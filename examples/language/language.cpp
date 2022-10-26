#include "defs.hpp"
#include "readDataset.hpp"
#include <dphdc.hpp>
#include <ResultsHandler.hpp>
#include <readExeInputs.hpp>

std::vector<std::string> language_names = {"Bulgarian", "Czech", "Danish", "German", "Greek", "English",
                                           "Estonian", "Finnish", "French", "Hungarian", "Italian", "Latvian",
                                           "Lithuanian", "Dutch", "Polish", "Portuguese", "Romanian", "Slovak",
                                           "Slovenian", "Spanish", "Swedish"};

size_t indexLastOccurrence(const std::vector<std::string> &vector_to_analyse, const std::string &string_to_analyse) {
    size_t index_last_occurrence = 0;

    for (; vector_to_analyse[index_last_occurrence] != string_to_analyse; index_last_occurrence++);

    for (; vector_to_analyse[index_last_occurrence] == string_to_analyse; index_last_occurrence++);

    return --index_last_occurrence;
}

std::vector<char> generateSingleCharVector() {
    std::vector<char> to_return(N_CHARS);

    for (unsigned int i = 0; i <= N_SMALL_Z - N_SMALL_A; i++) {
        to_return[i] = static_cast<char>(i + N_SMALL_A);
    }

    to_return.back() = ' ';
    return to_return;
}

std::pair<std::vector<std::vector<char>>, std::vector<std::string>>
generateNCharVector(int n, const std::vector<char> &single_char_vector) {
    unsigned long n_entries = static_cast<unsigned long>(pow(static_cast<double>(single_char_vector.size()), n));

    std::vector<std::vector<char>> n_gram_char_vector(n_entries, std::vector<char>(n));
    std::vector<std::string> n_gram_string_vector(n_entries);

    std::vector<unsigned long> indices(n, 0);

    for (unsigned long i = 0; i < n_entries; i++) {
        for (unsigned int j = 0; j < n; j++) {
            n_gram_char_vector[i][j] = single_char_vector[indices[j]];
        }

        for (const char &c: n_gram_char_vector[i]) {
            n_gram_string_vector[i].push_back(c);
        }

        for (int j = n - 1; j >= 0; j--) {
            if (++indices[j] == single_char_vector.size()) {
                indices[j] = 0;
            } else {
                break;
            }
        }
    }


    return std::make_pair(n_gram_char_vector, n_gram_string_vector);
}


dphdc::HDRepresentation<std::string> generateNGramRepresentation(int n, int vector_size, cl::sycl::queue &q) {
    if (n <= 0 || vector_size <= 0) {
        throw std::invalid_argument("n and vector size need to be bigger than 0");
    }
    std::vector<char> single_char_vector = generateSingleCharVector();

    std::vector<std::vector<char>> n_gram_char_vector;
    std::vector<std::string> n_gram_string_vector;

    {
        std::pair<std::vector<std::vector<char>>, std::vector<std::string>> temp = generateNCharVector(n,
                                                                                                       single_char_vector);
        n_gram_char_vector = temp.first;
        n_gram_string_vector = temp.second;
    }

    dphdc::HDRepresentation<char> char_representation(vector_size, dphdc::vectors_generator::random, q,
                                                      single_char_vector);

    dphdc::HDMatrix encoded_strings = char_representation.encodeWithXOR(n_gram_char_vector,
                                                                        dphdc::permutation::shift_right);

    return {encoded_strings, n_gram_string_vector};
}

void interactiveMode(dphdc::HDMatrix &associative_memory, dphdc::HDRepresentation<std::string> &n_gram_representation,
                     int n) {
    std::string user_input;
    bool running = false;

    std::cout << "Do you wish to run interactive mode? (y/n)\n";

    {
        bool done = false;
        while (!done) {
            getline(std::cin, user_input);
            if (removeSpaces(user_input) == "y") {
                running = true;
                done = true;
            } else if (removeSpaces(user_input) == "n") {
                done = true;
            } else if (removeSpaces(user_input) != "n") {
                std::cout << "Please use y or n\n";
            }
        }
    }

    if (running) {
        std::cout << "Welcome to interactive mode!\n";
        std::cout
                << "Please write a sentence in any of the 21 EU languages and the suspected language will be presented next\nInput q if you want to exit\n";
    }

    std::vector<std::vector<std::string>> input_vector(1);
    std::string answer;
    while (running) {
        getline(std::cin, user_input);
        if (removeSpaces(user_input) == "q") {
            break;
        } else if (removeSpaces(user_input).size() < n) {
            std::cout << "Sentences need to be bigger than " << n << " characters\n";
            continue;
        }
        input_vector.pop_back();
        input_vector.push_back(splitString(n, removeSpaces(user_input)));

        dphdc::HDMatrix encoded_entry = n_gram_representation.encodeWithBundle(input_vector,
                                                                               dphdc::permutation::no_permutation);
        answer = associative_memory.queryModel(encoded_entry, dphdc::distance_method::hamming_distance)[0];

        std::cout << "I think this language is: " << answer << "\n\n";
    }
}

int main(int argc, char **argv) {
    Inputs inputs = readExeInputs(argc, argv);
    ResultsHandler results_handler;
    results_handler.vector_size = inputs.vector_size;

    std::cout << "Using n-grams of size " << inputs.n_gram << "\n";

    cl::sycl::queue q{ACCELERATOR_CMAKE_QUEUE()};

    dphdc::HDRepresentation<std::string> n_gram_representation(1, dphdc::vectors_generator::none, q, {" "});
    dphdc::HDMatrix associative_memory(1, 1, dphdc::vectors_generator::none, q);

    {
        Data train_data = readDataset(inputs.n_gram, PROJECT_PATH_CMAKE "/examples/language/datasets/training/");
        size_t language_range[2];
        language_range[1] = 0;
        std::vector<dphdc::HDMatrix> associative_memory_before_joining;
        associative_memory_before_joining.reserve(language_names.size());
        results_handler.startTraining();
        n_gram_representation = generateNGramRepresentation(inputs.n_gram, inputs.vector_size, q);
        for (size_t i = 0; i < language_names.size(); i++) {
            language_range[0] = language_range[1];
            language_range[1] = indexLastOccurrence(train_data.labels_data, language_names[i]);

            associative_memory_before_joining.push_back(n_gram_representation.encodeWithBundle(
                    {&train_data.sentence_data[language_range[0]], &train_data.sentence_data[language_range[1] + 1]},
                    dphdc::permutation::no_permutation).reduceToLabelsBundle(
                    {&train_data.labels_data[language_range[0]], &train_data.labels_data[language_range[1] + 1]}));

            language_range[1]++;
        }

        associative_memory = dphdc::HDMatrix(associative_memory_before_joining);

        q.wait();
        results_handler.stopTraining();
    }

    results_handler.accelerator = n_gram_representation.getAssociatedAccelerator();

    {
        Data test_data = readDataset(inputs.n_gram, PROJECT_PATH_CMAKE "/examples/language/datasets/testing/");
        size_t language_range[2];
        language_range[1] = 0;
        std::vector<std::string> model_strings;
        model_strings.reserve(test_data.labels_data.size());
        results_handler.startTesting();
        for (size_t i = 0; i < language_names.size(); i++) {
            language_range[0] = language_range[1];
            language_range[1] = indexLastOccurrence(test_data.labels_data, language_names[i]);
            dphdc::HDMatrix encoded_test_entries = n_gram_representation.encodeWithBundle(
                    {&test_data.sentence_data[language_range[0]], &test_data.sentence_data[language_range[1] + 1]},
                    dphdc::permutation::no_permutation);
            std::vector<std::string> temp_vector = associative_memory.queryModel(encoded_test_entries,
                                                                                 dphdc::distance_method::cosine);
            model_strings.insert(model_strings.end(), temp_vector.begin(), temp_vector.end());

            language_range[1]++;
        }

        size_t successes = 0;
        for (size_t i = 0; i < test_data.labels_data.size(); i++) {
            if (test_data.labels_data[i] == model_strings[i]) {
                successes++;
            }
        }

        results_handler.success_rate = ((float) successes / (float) test_data.labels_data.size()) * 100;
        q.wait();
        results_handler.stopTesting();
    }

    results_handler.printToTerminal();
    results_handler.printToFile(PROJECT_PATH_CMAKE "/results/session/results-language.csv");

//    interactiveMode(associative_memory, n_gram_representation, inputs.n_gram);

    return 0;
}