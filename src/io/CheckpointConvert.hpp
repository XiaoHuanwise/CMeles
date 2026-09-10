/// @file CheckpointConvert.hpp
/// @brief Offline conversion of strategy-B checkpoints into strategy-A
///        solution files (host-only: no OCCA device required).

#pragma once

#include <string>

/// @brief Convert one checkpoint into a quadrature-point solution file.
///
/// Rebuilds the Vandermonde matrix from the mesh.h5 attributes (order,
/// N_q), projects the modal coefficients of every conserved variable onto
/// the quadrature points, derives the primitive variables and writes a
/// file with the same structure as the strategy A output.
///
/// @param meshPath       mesh.h5 written by CheckpointWriter::writeMesh.
/// @param checkpointPath checkpoint_<step>.h5.
/// @param outPath        destination solution_<step>.h5.
/// @throws std::runtime_error on inconsistent metadata or I/O failure.
void convertCheckpoint(const std::string &meshPath,
                       const std::string &checkpointPath,
                       const std::string &outPath);
