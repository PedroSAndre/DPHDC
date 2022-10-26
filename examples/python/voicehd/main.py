import sys
import time

from readDataset import readDataset

sys.path.append("build/src/")
import pydphdc

VECTOR_SIZE = 10000


def generateRangeVector() -> list[int]:
    to_return = []

    for i in range(20):
        to_return.append(i)

    return to_return


def getSelector() -> pydphdc.selector:
    if sys.argv[1] == "CPU":
        selector = pydphdc.cpu
    elif sys.argv[1] == "GPU":
        selector = pydphdc.gpu
    elif sys.argv[1] == "CUDA":
        selector = pydphdc.cuda
    else:
        raise ValueError("Selector passed is not valid")
    return selector


def main():
    train_data_labels = readDataset(True)
    test_data_labels = readDataset(False)

    selector = getSelector()

    intensity_representation = pydphdc.HDRepresentationInt(VECTOR_SIZE, pydphdc.full_level, selector,
                                                           generateRangeVector())
    position_vectors = pydphdc.HDMatrix(VECTOR_SIZE, len(train_data_labels[0][0]), pydphdc.random, selector)

    training_start = time.perf_counter()
    associative_memory = intensity_representation.encodeWithXOR(train_data_labels[0],
                                                                position_vectors).reduceToLabelsBundle(
        train_data_labels[1])
    training_stop = time.perf_counter()

    testing_start = time.perf_counter()
    encoded_test_entries = intensity_representation.encodeWithXOR(test_data_labels[0], position_vectors)
    accuracy = associative_memory.testModel(encoded_test_entries, test_data_labels[1],
                                            pydphdc.hamming_distance) * 100
    testing_stop = time.perf_counter()

    print(f"Accuracy:  {accuracy}%")
    print(f"Training took: {training_stop - training_start}s")
    print(f"Testing took: {testing_stop - testing_start}s")
    print(intensity_representation.getAssociatedAccelerator())


if __name__ == "__main__":
    main()
