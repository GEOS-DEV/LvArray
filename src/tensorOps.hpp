/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2019, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-746361
 *
 * All rights reserved. See COPYRIGHT for details.
 *
 * This file is part of the GEOSX Simulation Framework.
 *
 * GEOSX is a free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
