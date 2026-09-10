/// @file HDF5Writer.hpp
/// @brief Thin RAII wrapper around a HighFive file opened for writing.
///
/// The wrapper is deliberately primitive: scalar attributes, 1-D/2-D Real
/// datasets and lazily-created group datasets. All field access (device
/// copies, modal-to-nodal projection) lives in the strategy-specific
/// writers. Construction truncates an existing file; destruction closes
/// it via HighFive's own RAII, so a partially written file is cleanly
/// replaced by the next run.

#pragma once

#include <string>

#include <highfive/H5File.hpp>

#include "common/Types.hpp"

/// @brief Minimal RAII wrapper around an HDF5 file opened for writing.
class HDF5Writer {
public:
    /// @brief Open (or create) an HDF5 file, truncating any existing one.
    /// @throws HighFive::FileException on I/O failure.
    explicit HDF5Writer(const std::string &path);

    /// @brief Write a floating-point scalar attribute to the root group.
    void writeAttribute(const std::string &name, Real value);

    /// @brief Write an integer scalar attribute to the root group.
    void writeAttribute(const std::string &name, int value);

    /// @brief Write a string scalar attribute to the root group.
    void writeAttribute(const std::string &name, const std::string &value);

    /// @brief Write a 1-D dataset of \p count Real values (SoA layout).
    void writeDataset(const std::string &name, const Real *data, size_t count);

    /// @brief Write a 2-D row-major dataset of \p rows x \p cols Reals.
    void writeDataset2D(const std::string &name, const Real *data, size_t rows,
                        size_t cols);

    /// @brief Write a 1-D dataset into a subgroup, creating the group on
    ///        first use (used for mesh/x and mesh/y).
    void writeGroupDataset(const std::string &group, const std::string &name,
                           const Real *data, size_t count);

    /// @brief Flush pending writes to disk (destruction closes cleanly).
    void flush();

private:
    HighFive::File file_;
};
