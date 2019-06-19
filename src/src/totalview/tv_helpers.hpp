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
 * @file tv_helpers.hpp
 */

#ifndef CORECOMPONENTS_CXX_UTILITIES_SRC_SRC_TOTALVIEW_TV_HELPERS_HPP_
#define CORECOMPONENTS_CXX_UTILITIES_SRC_SRC_TOTALVIEW_TV_HELPERS_HPP_

#include "StringUtilities.hpp"
#include <typeinfo>

/**
 * @namespace This namespace is used to encapsulate functions that are used by the implementation
 *            of the totalview c++ view for data inspection within totalview.
 * @return
 */
namespace totalview
{

/**
 * @brief A function to wrap the stringify and demangling of a type.
 * @tparam The type to be stringified.
 * @return A string holding the name of the type.
 */
template< typename T >
std::string typeName( )
{
  return cxx_utilities::demangle( typeid(T).name() );
}

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
  std::string rval = typeName< TYPE >();
  for( int i=0 ; i<NDIM ; ++i )
  {
    rval += "["+std::to_string( dims[i] )+"]";
  }
  return rval;
}

}


#endif /* CORECOMPONENTS_CXX_UTILITIES_SRC_SRC_TOTALVIEW_TV_HELPERS_HPP_ */
