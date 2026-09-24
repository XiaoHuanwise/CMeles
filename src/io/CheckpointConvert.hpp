/// @file CheckpointConvert.hpp
/// @brief Offline conversion of strategy-B checkpoints into strategy-A
///        solution files (host-only: no OCCA device required).

#pragma once

#include <string>

#include "config/Config.hpp"

/// @brief Convert one checkpoint into a quadrature-point solution file.
///
/// Rebuilds the Vandermonde matrix from the mesh.h5 attributes (order,
/// N_q), projects the modal coefficients of every conserved variable onto
/// the quadrature points, derives the primitive variables and writes a
/// file with the same structure as the strategy A output. The
/// visualization outputs (XDMF sidecar / Tecplot PLT) requested by \p
/// format are generated next to it, exactly as in the direct strategy;
/// the sub-grid connectivity is rebuilt combinatorially from N_elem/N_q
/// plus the element_type attribute, so pre-existing mesh.h5 files
/// (without the format attribute) convert unchanged.
///
/// @param meshPath       mesh.h5 written by CheckpointWriter::writeMesh.
/// @param checkpointPath checkpoint_<step>.h5.
/// @param outPath        destination solution_<step>.h5.
/// @param format         output format for the converted files.
/// @return               The physical time of the converted step (read
///                       from the checkpoint's "time" attribute; batch
///                       conversion collects the values for the
///                       temporal-collection index).
/// @throws std::runtime_error on inconsistent metadata or I/O failure.
Real convertCheckpoint(const std::string &meshPath,
                       const std::string &checkpointPath,
                       const std::string &outPath,
                       OutputFormat format = OutputFormat::Hdf5);

/// @brief Read the optional "format" attribute recorded in a mesh.h5.
/// @return The parsed format, or Hdf5 when the attribute is absent
///         (older files).
/// @throws std::invalid_argument when the attribute holds an unknown name.
OutputFormat convertReadMeshFormat(const std::string &meshPath);

/// @brief Read the optional "series" attribute recorded in a mesh.h5
///        ([output] series at run time).
/// @return False only when the attribute is explicitly 0; defaults to
///         true for older files without the attribute.
bool convertReadMeshSeries(const std::string &meshPath);
