add_executable(language language.cpp readDataset.cpp readDataset.hpp)
target_link_libraries(language dphdc readExeInputs ResultsHandler)