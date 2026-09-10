/// @file HDF5Writer.cpp
/// @brief Implementation of the HDF5Writer wrapper.

#include "io/HDF5Writer.hpp"

HDF5Writer::HDF5Writer(const std::string &path)
    : file_(path, HighFive::File::Overwrite) {
}

void HDF5Writer::writeAttribute(const std::string &name, Real value) {
    file_.createAttribute(name, value);
}

void HDF5Writer::writeAttribute(const std::string &name, int value) {
    file_.createAttribute(name, value);
}

void HDF5Writer::writeAttribute(const std::string &name,
                                const std::string &value) {
    file_.createAttribute(name, value);
}

void HDF5Writer::writeDataset(const std::string &name, const Real *data,
                              size_t count) {
    file_.createDataSet<Real>(name, HighFive::DataSpace({count}))
        .write_raw(data);
}

void HDF5Writer::writeDataset2D(const std::string &name, const Real *data,
                                size_t rows, size_t cols) {
    file_.createDataSet<Real>(name, HighFive::DataSpace({rows, cols}))
        .write_raw(data);
}

void HDF5Writer::writeGroupDataset(const std::string &group,
                                   const std::string &name, const Real *data,
                                   size_t count) {
    // Lazily create the group: createGroup throws if it already exists.
    auto grp =
        file_.exist(group) ? file_.getGroup(group) : file_.createGroup(group);
    grp.createDataSet<Real>(name, HighFive::DataSpace({count})).write_raw(data);
}

void HDF5Writer::flush() {
    file_.flush();
}
