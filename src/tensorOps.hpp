/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file tensorOps.hpp
 */

#pragma once

// Source includes
#include "genericTensorOps.hpp"
#include "fixedSizeSquareMatrixOps.hpp"

namespace LvArray
{

/**
 * @brief Contains operations for operating on compile time sized vectors and matrices.
 * @details LvArray::tensorOps functions accept four differnet types of arguments
 *     -# Scalars
 *     -# Vectors: These can either be a one dimensional c-array such as double[ 3 ] or a one dimensional
 *        LvArray::Array, LvArray::ArrayView, or LvArray::ArraySlice.
 *     -# Matrices: These can either be a two dimensional c-array such as int[ 5 ][ 2 ] or a two dimensional
 *        LvArray::Array, LvArray::ArrayView, or LvArray::ArraySlice.
 *     -# Symmetric matrices: These are represented in Voigt notation as a vector.
 *
 *   Each function takes in the size of the objects as template parameters. As an example to take
 *   the dot product of two vectors of length 3 you'd call
 *     @code LvArray::tensorOps::AiBi< 3 >( x, y ) @endcode
 */
namespace tensorOps
{}

}
