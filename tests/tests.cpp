#include <gtest/gtest.h>
#include <dphdc.hpp>

// tests_constructors_generators //
TEST(tests_constructors_generators, fail_if_0_or_negative) {
    const int hd_vector_size = 10000;
    std::vector<char> chars_to_represent = {};
// Testing it does not accept negative or zero values //
    EXPECT_THROW(dphdc::HDRepresentation<char> test_representation(hd_vector_size, dphdc::vectors_generator::none,
                                                                   ACCELERATOR_CMAKE_DPHDC,
                         chars_to_represent), std::invalid_argument);

}

TEST(tests_constructors_generators, generator_none) {
    const int hd_vector_size = 10000;
    std::vector<char> chars_to_represent = {'A', 'B'};
// Testing constructor all false //
    dphdc::HDRepresentation<char> test_representation(hd_vector_size, dphdc::vectors_generator::none,
                                                      ACCELERATOR_CMAKE_DPHDC, chars_to_represent);

    std::vector<std::vector<bool>> vector_on_host = test_representation.getVectors();

    for (const std::vector<bool> &i: vector_on_host) {
        for (const bool &j: i) {
            EXPECT_FALSE(j);
        }
    }

// Printing device being used to terminal //
    std::cout << "Used device " << test_representation.

            getAssociatedAccelerator()

              << " to run test " << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}

TEST(tests_constructors_generators, generator_all_true) {
    const int hd_vector_size = 10000;
    std::vector<char> chars_to_represent = {'A', 'B'};
// Testing constructor all true //
    dphdc::HDRepresentation<char> test_representation = dphdc::HDRepresentation<char>(hd_vector_size,
                                                                                      dphdc::vectors_generator::all_true,
                                                                                      ACCELERATOR_CMAKE_DPHDC,
                                                                                      chars_to_represent);

    std::vector<std::vector<bool>> vector_on_host = test_representation.getVectors();

    for (const std::vector<bool> &i: vector_on_host) {
        for (const bool &j: i) {
            EXPECT_TRUE(j);
        }
    }
// Printing device being used to terminal //
    std::cout << "Used device " << test_representation.

            getAssociatedAccelerator()

              << " to run test " << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}

TEST(tests_constructors_generators, generator_random) {
    const int hd_vector_size = 10000;
    std::vector<char> chars_to_represent = {'A', 'B'};
// Testing constructor random //
    dphdc::HDRepresentation<char> test_representation = dphdc::HDRepresentation<char>(hd_vector_size,
                                                                                      dphdc::vectors_generator::random,
                                                                                      ACCELERATOR_CMAKE_DPHDC,
                                                                                      chars_to_represent);

    std::vector<std::vector<bool>> vector_on_host = test_representation.getVectors();

    for (const std::vector<bool> &i: vector_on_host) {
        EXPECT_TRUE(std::count(i.begin(), i.end(), false) >= 0.48 * hd_vector_size &&
                    std::count(i.begin(), i.end(), false) <= 0.52 * hd_vector_size);
        EXPECT_TRUE(std::count(i.begin(), i.end(), true) >= 0.48 * hd_vector_size &&
                    std::count(i.begin(), i.end(), true) <= 0.52 * hd_vector_size);
    }
// Printing device being used to terminal //
    std::cout << "Used device " << test_representation.

            getAssociatedAccelerator()

              << " to run test " << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}

TEST(tests_constructors_generators, generator_half_level) {
    const int hd_vector_size = 4;
    std::vector<char> chars_to_represent = {'A', 'B', 'C'};
// Testing constructor random //
    dphdc::HDRepresentation<char> test_representation = dphdc::HDRepresentation<char>(hd_vector_size,
                                                                                      dphdc::vectors_generator::half_level,
                                                                                      ACCELERATOR_CMAKE_DPHDC,
                                                                                      chars_to_represent);

    std::vector<std::vector<bool>> vector_on_host = test_representation.getVectors();

    unsigned int calc;
    for (unsigned int i = 1; i < chars_to_represent.

            size();

         i++) {
        calc = 0;
        for (unsigned int j = 0; j < vector_on_host[i].size(); j++) {
            if (vector_on_host[i][j] != vector_on_host[i - 1][j]) {
                calc++;
            }
        }
        EXPECT_EQ(calc, 1);
    }
// Printing device being used to terminal //
    std::cout << "Used device " << test_representation.getAssociatedAccelerator() << " to run test "
              << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}

TEST(tests_constructors_generators, generator_full_level) {
    const int hd_vector_size = 4;
    std::vector<char> chars_to_represent = {'A', 'B', 'C'};
// Testing constructor random //
    dphdc::HDRepresentation<char> test_representation = dphdc::HDRepresentation<char>(hd_vector_size,
                                                                                      dphdc::vectors_generator::full_level,
                                                                                      ACCELERATOR_CMAKE_DPHDC,
                                                                                      chars_to_represent);

    std::vector<std::vector<bool>> vector_on_host = test_representation.getVectors();

    unsigned int calc;
    for (unsigned int i = 1; i < chars_to_represent.size(); i++) {
        calc = 0;
        for (unsigned int j = 0; j < vector_on_host[i].

                size();

             j++) {
            if (vector_on_host[i][j] != vector_on_host[i - 1][j]) {
                calc++;
            }
        }
        EXPECT_EQ(calc, 2);
    }
// Printing device being used to terminal //
    std::cout << "Used device " << test_representation.getAssociatedAccelerator() << " to run test "
              << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}

TEST(tests_constructors_generators, generator_circular_even) {
    const int hd_vector_size = 10000;
    std::vector<int> ints_to_represent = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
// Testing constructor random //
    dphdc::HDRepresentation<int> test_representation(hd_vector_size, dphdc::vectors_generator::circular,
                                                     ACCELERATOR_CMAKE_DPHDC, ints_to_represent);

    std::vector<std::vector<bool>> vector_on_host = test_representation.getVectors();

    unsigned int calc;
    for (unsigned int i = 1; i < ints_to_represent.size(); i++) {
        calc = 0;
        for (unsigned int j = 0; j < vector_on_host[i].size(); j++) {
            if (vector_on_host[i][j] != vector_on_host[i - 1][j]) {
                calc++;
            }
        }
        EXPECT_EQ(calc, 2000);
    }
    calc = 0;
    for (unsigned int j = 0; j < vector_on_host[0].size(); j++) {
        if (vector_on_host[0][j] != vector_on_host.back()[j]) {
            calc++;
        }
    }
    EXPECT_EQ(calc, 2000);
    calc = 0;
    for (unsigned int j = 0; j < vector_on_host[0].size(); j++) {
        if (vector_on_host[0][j] != vector_on_host[5][j]) {
            calc++;
        }
    }
    EXPECT_EQ(calc, 10000);
// Printing device being used to terminal //
    std::cout << "Used device " << test_representation.getAssociatedAccelerator() << " to run test "
              << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}

TEST(tests_constructors_generators, generator_circular_odd) {
    const int hd_vector_size = 10000;
    std::vector<int> ints_to_represent = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
// Testing constructor random //
    dphdc::HDRepresentation<int> test_representation(hd_vector_size, dphdc::vectors_generator::circular,
                                                     ACCELERATOR_CMAKE_DPHDC, ints_to_represent);
    std::vector<std::vector<bool>> vector_on_host = test_representation.getVectors();

    unsigned int calc;
    for (unsigned int i = 1; i < ints_to_represent.size(); i++) {
        calc = 0;
        for (unsigned int j = 0; j < vector_on_host[i].size(); j++) {
            if (vector_on_host[i][j] != vector_on_host[i - 1][j]) {
                calc++;
            }
        }
        EXPECT_EQ(calc, 2000);
    }
    calc = 0;
    for (unsigned int j = 0; j < vector_on_host[0].size(); j++) {
        if (vector_on_host[0][j] != vector_on_host.back()[j]) {
            calc++;
        }
    }
    EXPECT_EQ(calc, 2000);
    calc = 0;
    for (unsigned int j = 0; j < vector_on_host[0].size(); j++) {
        if (vector_on_host[0][j] != vector_on_host[5][j]) {
            calc++;
        }
    }
    EXPECT_EQ(calc, 10000);
// Printing device being used to terminal //
    std::cout << "Used device " << test_representation.getAssociatedAccelerator() << " to run test "
              << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}

TEST(tests_constructors_generators, join_constructor) {
    const int hd_vector_size = 3;
    std::vector<dphdc::HDMatrix> test_matrices;
    test_matrices.emplace_back(hd_vector_size, 1, dphdc::vectors_generator::none, ACCELERATOR_CMAKE_DPHDC);
    test_matrices[0] = test_matrices[0].reduceToLabelsBundle({"0"});
    test_matrices.emplace_back(hd_vector_size, 2, dphdc::vectors_generator::all_true, ACCELERATOR_CMAKE_DPHDC);
    test_matrices[1] = test_matrices[1].reduceToLabelsBundle({"1", "2"});
    test_matrices.emplace_back(hd_vector_size, 3, dphdc::vectors_generator::none, ACCELERATOR_CMAKE_DPHDC);
    test_matrices[2] = test_matrices[2].reduceToLabelsBundle({"3", "4", "6"});

    dphdc::HDMatrix test_final_matrix(test_matrices);

    std::vector<std::vector<bool>> answer = {{false, false, false},
                                             {true,  true,  true},
                                             {true,  true,  true},
                                             {false, false, false},
                                             {false, false, false},
                                             {false, false, false}};

    std::vector<std::string> answer_labels = {"0", "1", "2", "3", "4", "6"};

    EXPECT_EQ(test_final_matrix.getVectors(), answer);
    EXPECT_EQ(test_final_matrix.getLabels(), answer_labels);

// Printing device being used to terminal //
    std::cout << "Used device " << test_matrices[0].getAssociatedAccelerator() << " to run test "
              << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}

TEST(tests_constructors_generators, set_vectors_test) {
    const int hd_vector_size = 3;
    std::vector<int> ints_to_represent = {0};
    dphdc::HDRepresentation<int> test_representation = dphdc::HDRepresentation<int>(hd_vector_size,
                                                                                    dphdc::vectors_generator::none,
                                                                                    ACCELERATOR_CMAKE_DPHDC,
                                                                                    ints_to_represent);

    std::vector<std::vector<bool>> vectors_to_set = {{false, true, false}};
    test_representation.setVectors(vectors_to_set);

    std::vector<std::vector<bool>> vectors_on_host = test_representation.getVectors();

    EXPECT_EQ(vectors_to_set, vectors_on_host);

// Printing device being used to terminal //
    std::cout << "Used device " << test_representation.

            getAssociatedAccelerator()

              << " to run test " << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}

TEST(tests_encoders, encode_with_bundle_no_postioning) {
    const int hd_vector_size = 4;
    std::vector<int> ints_to_represent = {0, 1};
    dphdc::HDRepresentation<int> test_representation = dphdc::HDRepresentation<int>(hd_vector_size,
                                                                                    dphdc::vectors_generator::none,
                                                                                    ACCELERATOR_CMAKE_DPHDC,
                                                                                    ints_to_represent);

    std::vector<std::vector<bool>> vectors_to_set = {{true, true,  false, false},
                                                     {true, false, true,  false}};
    test_representation.setVectors(vectors_to_set);

    std::vector<std::vector<int>> made_up_data = {{0, 0, 0, 0, 0},
                                                  {1, 1, 1, 1, 1},
                                                  {0, 1, 0, 1, 0}};

    std::vector<std::vector<bool>> vectors_on_host = test_representation.encodeWithBundle(made_up_data,
                                                                                          dphdc::permutation::no_permutation).getVectors();

    std::vector<std::vector<bool>> answer = {{true, true,  false, false},
                                             {true, false, true,  false},
                                             {true, true,  false, false}};

    EXPECT_EQ(answer, vectors_on_host);

// Printing device being used to terminal //
    std::cout << "Used device " << test_representation.

            getAssociatedAccelerator()

              << " to run test " << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}

TEST(tests_encoders, encode_with_bundle_shift_right) {
    const int hd_vector_size = 4;
    std::vector<int> ints_to_represent = {0, 1};
    dphdc::HDRepresentation<int> test_representation = dphdc::HDRepresentation<int>(hd_vector_size,
                                                                                    dphdc::vectors_generator::none,
                                                                                    ACCELERATOR_CMAKE_DPHDC,
                                                                                    ints_to_represent);

    std::vector<std::vector<bool>> vectors_to_set = {{true, true,  false, false},
                                                     {true, false, true,  false}};
    test_representation.setVectors(vectors_to_set);

    std::vector<std::vector<int>> made_up_data = {{0, 0, 0, 0, 0},
                                                  {1, 1, 1, 1, 1},
                                                  {0, 1, 0, 1, 0}};

    std::vector<std::vector<bool>> vectors_on_host = test_representation.encodeWithBundle(made_up_data,
                                                                                          dphdc::permutation::shift_right).getVectors();

    std::vector<std::vector<bool>> answer = {{true,  true,  false, false},
                                             {true,  false, true,  false},
                                             {false, true,  false, true}};

    EXPECT_EQ(answer, vectors_on_host);

// Printing device being used to terminal //
    std::cout << "Used device " << test_representation.

            getAssociatedAccelerator()

              << " to run test " << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}

TEST(tests_encoders, encode_with_XOR_position_vectors) {
    const int hd_vector_size = 4;
    std::vector<int> ints_to_represent = {0, 1};
    dphdc::HDRepresentation<int> test_representation = dphdc::HDRepresentation<int>(hd_vector_size,
                                                                                    dphdc::vectors_generator::none,
                                                                                    ACCELERATOR_CMAKE_DPHDC,
                                                                                    ints_to_represent);

    std::vector<std::vector<bool>> vectors_to_set = {{true, true,  false, false},
                                                     {true, false, true,  false}};
    test_representation.setVectors(vectors_to_set);

    dphdc::HDMatrix position_vectors(hd_vector_size, 4, dphdc::vectors_generator::none, ACCELERATOR_CMAKE_DPHDC);


    std::vector<std::vector<bool>> position_vectors_to_set = {{false, true,  true,  true},
                                                              {true,  false, true,  true},
                                                              {true,  true,  false, true},
                                                              {true,  true,  true,  false}};

    position_vectors.setVectors(position_vectors_to_set);

    std::vector<std::vector<int>> made_up_data = {{0, 0, 0, 0},
                                                  {1, 1, 1, 1},
                                                  {0, 1, 0, 1}};

    std::vector<std::vector<bool>> vectors_on_host = test_representation.encodeWithXOR(made_up_data,
                                                                                       position_vectors).getVectors();

    std::vector<std::vector<bool>> answer = {{false, false, true,  true},
                                             {false, true,  false, true},
                                             {false, false, false, true}};

    EXPECT_EQ(answer, vectors_on_host);

// Printing device being used to terminal //
    std::cout << "Used device " << test_representation.

            getAssociatedAccelerator()

              << " to run test " << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}

TEST(tests_encoders, encode_with_XOR_no_positioning) {
    const int hd_vector_size = 4;
    std::vector<int> ints_to_represent = {0, 1};
    dphdc::HDRepresentation<int> test_representation = dphdc::HDRepresentation<int>(hd_vector_size,
                                                                                    dphdc::vectors_generator::none,
                                                                                    ACCELERATOR_CMAKE_DPHDC,
                                                                                    ints_to_represent);

    std::vector<std::vector<bool>> vectors_to_set = {{true, true,  false, false},
                                                     {true, false, true,  false}};
    test_representation.setVectors(vectors_to_set);

    std::vector<std::vector<int>> made_up_data = {{0, 0, 0, 0, 0},
                                                  {1, 1, 1, 1, 1},
                                                  {0, 1, 0, 1, 0}};

    std::vector<std::vector<bool>> vectors_on_host = test_representation.encodeWithXOR(made_up_data,
                                                                                       dphdc::permutation::no_permutation).getVectors();

    std::vector<std::vector<bool>> answer = {{true, true,  false, false},
                                             {true, false, true,  false},
                                             {true, true,  false, false}};

    EXPECT_EQ(answer, vectors_on_host);

// Printing device being used to terminal //
    std::cout << "Used device " << test_representation.

            getAssociatedAccelerator()

              << " to run test " << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}

TEST(tests_encoders, encode_with_XOR_shift_right) {
    const int hd_vector_size = 4;
    std::vector<int> ints_to_represent = {0, 1};
    dphdc::HDRepresentation<int> test_representation = dphdc::HDRepresentation<int>(hd_vector_size,
                                                                                    dphdc::vectors_generator::none,
                                                                                    ACCELERATOR_CMAKE_DPHDC,
                                                                                    ints_to_represent);

    std::vector<std::vector<bool>> vectors_to_set = {{true, true,  false, false},
                                                     {true, false, true,  false}};
    test_representation.setVectors(vectors_to_set);

    std::vector<std::vector<int>> made_up_data = {{0, 0, 0, 0, 0},
                                                  {1, 1, 1, 1, 1},
                                                  {0, 1, 0, 1, 0}};

    std::vector<std::vector<bool>> vectors_on_host = test_representation.encodeWithXOR(made_up_data,
                                                                                       dphdc::permutation::shift_right).getVectors();

    std::vector<std::vector<bool>> answer = {{true,  true,  false, false},
                                             {true,  false, true,  false},
                                             {false, false, true,  true}};

    EXPECT_EQ(answer, vectors_on_host);

// Printing device being used to terminal //
    std::cout << "Used device " << test_representation.

            getAssociatedAccelerator()

              << " to run test " << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}

TEST(tests_reducers, reduce_with_bundle) {
    const int hd_vector_size = 4;
    std::vector<std::string> created_labels = {"0", "1", "0", "1", "0"};
    dphdc::HDMatrix test_matrix(hd_vector_size, 5, dphdc::vectors_generator::none, ACCELERATOR_CMAKE_DPHDC);

    std::vector<std::vector<bool>> vectors_to_set = {{true, true,  true,  false},
                                                     {true, true,  false, false},
                                                     {true, true,  false, false},
                                                     {true, false, true,  false},
                                                     {true, false, false, false}};
    test_matrix.setVectors(vectors_to_set);

    dphdc::HDMatrix associative_memory = test_matrix.reduceToLabelsBundle(created_labels);

    std::vector<std::vector<bool>> associative_memory_vector = associative_memory.getVectors();

    std::vector<std::vector<bool>> answer = {{true, true, false, false},
                                             {true, true, true,  false}};

    EXPECT_EQ(answer, associative_memory_vector);

// Printing device being used to terminal //
    std::cout << "Used device " << test_matrix.

            getAssociatedAccelerator()

              << " to run test " << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}

TEST(tests_query_model, hamming_distance_query) {
    const int hd_vector_size = 4;
    std::vector<std::string> created_labels = {"0", "1", "0", "1", "0"};
    dphdc::HDMatrix test_matrix(hd_vector_size, 5, dphdc::vectors_generator::none, ACCELERATOR_CMAKE_DPHDC);

    std::vector<std::vector<bool>> vectors_to_set = {{true, true,  true,  false},
                                                     {true, true,  false, false},
                                                     {true, true,  false, false},
                                                     {true, false, true,  false},
                                                     {true, false, false, false}};
    test_matrix.setVectors(vectors_to_set);

    dphdc::HDMatrix associative_memory = test_matrix.reduceToLabelsBundle(created_labels);

    std::vector<std::vector<bool>> associative_memory_vector = associative_memory.getVectors();

    std::vector<std::vector<bool>> answer = {{true, true, false, false},
                                             {true, true, true,  false}};


    dphdc::HDMatrix created_encoded_test_entries(hd_vector_size, 2, dphdc::vectors_generator::none,
                                                 ACCELERATOR_CMAKE_DPHDC);
    created_encoded_test_entries.setVectors(answer);

    std::vector<std::string> answer_labels = {"0", "1"};

    EXPECT_EQ(associative_memory.queryModel(created_encoded_test_entries, dphdc::distance_method::hamming_distance),
              answer_labels);

// Printing device being used to terminal //
    std::cout << "Used device " << test_matrix.

            getAssociatedAccelerator()

              << " to run test " << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}


TEST(tests_test_model, hamming_distance_test) {
    const int hd_vector_size = 4;
    std::vector<std::string> created_labels = {"0", "1", "0", "1", "0"};
    dphdc::HDMatrix test_matrix(hd_vector_size, 5, dphdc::vectors_generator::none, ACCELERATOR_CMAKE_DPHDC);

    std::vector<std::vector<bool>> vectors_to_set = {{true, true,  true,  false},
                                                     {true, true,  false, false},
                                                     {true, true,  false, false},
                                                     {true, false, true,  false},
                                                     {true, false, false, false}};
    test_matrix.setVectors(vectors_to_set);

    dphdc::HDMatrix associative_memory = test_matrix.reduceToLabelsBundle(created_labels);

    std::vector<std::vector<bool>> associative_memory_vector = associative_memory.getVectors();

    std::vector<std::vector<bool>> answer = {{true, true, false, false},
                                             {true, true, true,  false}};


    dphdc::HDMatrix created_encoded_test_entries(hd_vector_size, 2, dphdc::vectors_generator::none,
                                                 ACCELERATOR_CMAKE_DPHDC);
    created_encoded_test_entries.setVectors(answer);

    std::vector<std::string> test_labels = {"0", "1"};

    EXPECT_EQ(associative_memory.testModel(created_encoded_test_entries, test_labels,
                                           dphdc::distance_method::hamming_distance), 1);

// Printing device being used to terminal //
    std::cout << "Used device " << test_matrix.

            getAssociatedAccelerator()

              << " to run test " << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}

TEST(tests_query_model, cosine_query) {
    const int hd_vector_size = 4;
    std::vector<std::string> created_labels = {"0", "1", "0", "1", "0"};
    dphdc::HDMatrix test_matrix(hd_vector_size, 5, dphdc::vectors_generator::none, ACCELERATOR_CMAKE_DPHDC);

    std::vector<std::vector<bool>> vectors_to_set = {{true, true,  true,  false},
                                                     {true, true,  false, false},
                                                     {true, true,  false, false},
                                                     {true, false, true,  false},
                                                     {true, false, false, false}};
    test_matrix.setVectors(vectors_to_set);

    dphdc::HDMatrix associative_memory = test_matrix.reduceToLabelsBundle(created_labels);

    std::vector<std::vector<bool>> associative_memory_vector = associative_memory.getVectors();

    std::vector<std::vector<bool>> answer = {{true, true, false, false},
                                             {true, true, true,  false}};


    dphdc::HDMatrix created_encoded_test_entries(hd_vector_size, 2, dphdc::vectors_generator::none,
                                                 ACCELERATOR_CMAKE_DPHDC);
    created_encoded_test_entries.setVectors(answer);

    std::vector<std::string> answer_labels = {"0", "1"};

    EXPECT_EQ(associative_memory.queryModel(created_encoded_test_entries, dphdc::distance_method::cosine),
              answer_labels);

// Printing device being used to terminal //
    std::cout << "Used device " << test_matrix.

            getAssociatedAccelerator()

              << " to run test " << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}


TEST(tests_test_model, cosine_test) {
    const int hd_vector_size = 4;
    std::vector<std::string> created_labels = {"0", "1", "0", "1", "0"};
    dphdc::HDMatrix test_matrix(hd_vector_size, 5, dphdc::vectors_generator::none, ACCELERATOR_CMAKE_DPHDC);

    std::vector<std::vector<bool>> vectors_to_set = {{true, true,  true,  false},
                                                     {true, true,  false, false},
                                                     {true, true,  false, false},
                                                     {true, false, true,  false},
                                                     {true, false, false, false}};
    test_matrix.setVectors(vectors_to_set);

    dphdc::HDMatrix associative_memory = test_matrix.reduceToLabelsBundle(created_labels);

    std::vector<std::vector<bool>> associative_memory_vector = associative_memory.getVectors();

    std::vector<std::vector<bool>> answer = {{true, true, false, false},
                                             {true, true, true,  false}};


    dphdc::HDMatrix created_encoded_test_entries(hd_vector_size, 2, dphdc::vectors_generator::none,
                                                 ACCELERATOR_CMAKE_DPHDC);
    created_encoded_test_entries.setVectors(answer);

    std::vector<std::string> test_labels = {"0", "1"};

    EXPECT_EQ(associative_memory.testModel(created_encoded_test_entries, test_labels, dphdc::distance_method::cosine),
              1);

// Printing device being used to terminal //
    std::cout << "Used device " << test_matrix.

            getAssociatedAccelerator()

              << " to run test " << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}

TEST(tests_read_write_data, matrix_write_read_test) {
    const int hd_vector_size = 5;
    const std::string filename_matrix = "test_matrix";
    const std::string filename_associative_memory = "test_associative_memory";
    std::vector<std::string> created_labels = {"0", "1", "010", "1000", "0101"};

    dphdc::HDMatrix memory_matrix(hd_vector_size, 5, dphdc::vectors_generator::random, ACCELERATOR_CMAKE_DPHDC);
    dphdc::HDMatrix memory_associative_memory = memory_matrix.reduceToLabelsBundle(created_labels);

    memory_matrix.storeMatrix(filename_matrix);
    dphdc::HDMatrix disk_matrix(filename_matrix + ".dphdcm", ACCELERATOR_CMAKE_DPHDC);

    memory_associative_memory.storeMatrix(filename_associative_memory);
    dphdc::HDMatrix disk_associative_memory(filename_associative_memory + ".dphdcm", ACCELERATOR_CMAKE_DPHDC);

    EXPECT_EQ(memory_matrix.getVectors(), disk_matrix.getVectors());
    EXPECT_EQ(memory_matrix.getLabels(), disk_matrix.getLabels());

    EXPECT_EQ(memory_associative_memory.getVectors(), disk_associative_memory.getVectors());
    EXPECT_EQ(memory_associative_memory.getLabels(), disk_associative_memory.getLabels());

    EXPECT_THROW(dphdc::HDMatrix test("a", ACCELERATOR_CMAKE_DPHDC), std::invalid_argument);

// Printing device being used to terminal //
    std::cout << "Used device " << memory_matrix.getAssociatedAccelerator() << " to run test "
              << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}

TEST(tests_read_write_data, representation_write_read_test) {
    const int hd_vector_size = 5;
    const std::string filename_representation = "test_representation";
    std::vector<unsigned char> chars_to_represent = {'a', 'b', 'c', 'd', 'e'};

    dphdc::HDRepresentation<unsigned char> memory_representation(hd_vector_size, dphdc::vectors_generator::random,
                                                                 ACCELERATOR_CMAKE_DPHDC, chars_to_represent);

    memory_representation.storeRepresentation(filename_representation);
    dphdc::HDRepresentation<unsigned char> disk_representation(filename_representation + ".dphdcr",
                                                               ACCELERATOR_CMAKE_DPHDC);

    EXPECT_EQ(memory_representation.getVectors(), disk_representation.getVectors());
    EXPECT_EQ(memory_representation.getDataTranslation(), disk_representation.getDataTranslation());


    EXPECT_THROW(dphdc::HDRepresentation<unsigned char> test("a", ACCELERATOR_CMAKE_DPHDC), std::invalid_argument);

// Printing device being used to terminal //
    std::cout << "Used device " << memory_representation.getAssociatedAccelerator() << " to run test "
              << ::testing::UnitTest::GetInstance()->current_test_info()->name() << std::endl;
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}