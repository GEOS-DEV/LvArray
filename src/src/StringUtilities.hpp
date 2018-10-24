/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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

/*
 * StringUtilities.hpp
 *
 *  Created on: Nov 12, 2014
 *      Author: rrsettgast
 */

#ifndef CXX_UTILITIES_STRINGUTILITIES_HPP_
#define CXX_UTILITIES_STRINGUTILITIES_HPP_


#include <string>
#include <vector>
#include <iostream>
#include "Array.hpp"

namespace cxx_utilities
{
std::string demangle( const std::string& name );


template< typename T, int NDIM, typename INDEX_TYPE >
void equateStlVector( LvArray::Array<T, NDIM, INDEX_TYPE> & lhs, std::vector<T> const & rhs )
{
  static_assert(NDIM == 1, "Dimension must be 1");
  GEOS_ERROR_IF(lhs.size() != static_cast<INDEX_TYPE>(rhs.size()), "Size mismatch");
  for( unsigned int a=0 ; a<rhs.size() ; ++a )
  {
    lhs[a] = rhs[a];
  }
}

template<typename T>
void equateStlVector( T & lhs, std::vector<T> const & rhs )
{
  GEOS_ERROR_IF(rhs.size() > 1, "Size should not be greater than 1 but is: " << rhs.size());
  if ( rhs.size() == 1 )
  {
    lhs = rhs[0];
  }
}

}

#endif /* STRINGUTILITIES_HPP_ */
