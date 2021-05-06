/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file tv_helpers.hpp
 */

#pragma once

// Source includes
#include "../system.hpp"

// TPL include
#include <typeinfo>

/**
 * @brief Contains functions that are used by the implementation of the
 *   totalview c++ view for data inspection within totalview.
 */
namespace totalview
{

/**
 * @brief This function returns a string that may be used as the "type" in a call to
 *        TV_ttf_add_row(). This will either be a single value or an array.
 * @tparam TYPE The type of that requires a format string.
 * @tparam INDEX_TYPE The type of integer that dims is passed in as.
 * @param NDIM  the number of dimensions of the array (0 if scalar)
 * @param dims  The dimensions of the array
 * @return A string for use as "type" in a TV_tff_add_row( name, type, data) call.
 */
template< typename TYPE, typename INDEX_TYPE >
std::string format( int NDIM, INDEX_TYPE const * const dims )
{
  std::string rval = LvArray::system::demangleType< TYPE >();
  for( int i=0; i<NDIM; ++i )
  {
    rval += "["+std::to_string( dims[i] )+"]";
  }
  return rval;
}

}
