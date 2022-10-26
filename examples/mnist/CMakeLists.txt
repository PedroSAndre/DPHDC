add_executable(mnist mnist.cpp readDataset.cpp readDataset.hpp)
target_link_libraries(mnist dphdc readExeInputs ResultsHandler)