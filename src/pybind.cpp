#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "HDRepresentation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pydphdc, handle) {
    handle.doc() = "Port of the DPHDC C++ + SYCL library";

    py::enum_<dphdc::permutation::permutation> permutation(handle, "permutation");
    permutation.value("no_permutation", dphdc::permutation::no_permutation);
    permutation.value("shift_right", dphdc::permutation::shift_right);
    permutation.export_values();

    py::enum_<dphdc::vectors_generator::vectors_generator> vectors_generator(handle, "vectors_generator");
    vectors_generator.value("none", dphdc::vectors_generator::none);
    vectors_generator.value("all_true", dphdc::vectors_generator::all_true);
    vectors_generator.value("random", dphdc::vectors_generator::random);
    vectors_generator.value("half_level", dphdc::vectors_generator::half_level);
    vectors_generator.value("full_level", dphdc::vectors_generator::full_level);
    vectors_generator.export_values();

    py::enum_<dphdc::distance_method::distance_method> distance_method(handle, "distance_method");
    distance_method.value("hamming_distance", dphdc::distance_method::hamming_distance);
    distance_method.export_values();

    py::enum_<dphdc::selector> selector(handle, "selector");
    selector.value("cpu", dphdc::cpu);
    selector.value("gpu", dphdc::gpu);
    selector.value("cuda", dphdc::cuda);
    selector.export_values();


    py::class_<dphdc::HDMatrix> hd_matrix_class(handle, "HDMatrix");
    hd_matrix_class.def(py::init<int, int, dphdc::vectors_generator::vectors_generator, dphdc::selector>());
    hd_matrix_class.def(py::init<std::vector<dphdc::HDMatrix> &>());
    hd_matrix_class.def("reduceToLabelsBundle", &dphdc::HDMatrix::reduceToLabelsBundle);
    hd_matrix_class.def("testModel", &dphdc::HDMatrix::testModel);
    hd_matrix_class.def("queryModel", &dphdc::HDMatrix::queryModel);
    hd_matrix_class.def("getVectors", &dphdc::HDMatrix::getVectors);
    hd_matrix_class.def("getAssociatedAccelerator", &dphdc::HDMatrix::getAssociatedAccelerator);
    hd_matrix_class.def("setVectors", &dphdc::HDMatrix::setVectors);
    hd_matrix_class.def("getLabels", &dphdc::HDMatrix::getLabels);

    py::class_<dphdc::HDRepresentation<int>, dphdc::HDMatrix> hd_representation_2(handle, "HDRepresentationInt");
    hd_representation_2.def(
            py::init<int, dphdc::vectors_generator::vectors_generator, dphdc::selector, const std::vector<int> &>());
    hd_representation_2.def(py::init<dphdc::HDMatrix &, const std::vector<int> &>());
    hd_representation_2.def("encodeWithBundle", &dphdc::HDRepresentation<int>::encodeWithBundle);
    hd_representation_2.def("encodeWithXOR",
                            py::overload_cast<const std::vector<std::vector<int>> &, dphdc::HDMatrix &>(
                                    &dphdc::HDRepresentation<int>::encodeWithXOR));
    hd_representation_2.def("encodeWithXOR",
                            py::overload_cast<const std::vector<std::vector<int>> &, dphdc::permutation::permutation>(
                                    &dphdc::HDRepresentation<int>::encodeWithXOR));

    py::class_<dphdc::HDRepresentation<std::string>, dphdc::HDMatrix> hd_representation_18(handle,
                                                                                           "HDRepresentationStr");
    hd_representation_18.def(
            py::init<int, dphdc::vectors_generator::vectors_generator, dphdc::selector, const std::vector<std::string> &>());
    hd_representation_18.def(py::init<dphdc::HDMatrix &, const std::vector<std::string> &>());
    hd_representation_18.def("encodeWithBundle", &dphdc::HDRepresentation<std::string>::encodeWithBundle);
    hd_representation_18.def("encodeWithXOR",
                             py::overload_cast<const std::vector<std::vector<std::string>> &, dphdc::HDMatrix &>(
                                     &dphdc::HDRepresentation<std::string>::encodeWithXOR));
    hd_representation_18.def("encodeWithXOR",
                             py::overload_cast<const std::vector<std::vector<std::string>> &, dphdc::permutation::permutation>(
                                     &dphdc::HDRepresentation<std::string>::encodeWithXOR));
}