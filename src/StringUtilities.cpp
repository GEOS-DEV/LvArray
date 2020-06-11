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


#include "StringUtilities.hpp"

// System includes
#include <cxxabi.h>
#include <cstring>
#include <memory>

namespace LvArray
{

///////////////////////////////////////////////////////////////////////////////
std::string demangle( const std::string & name )
{
  int status = -4; // some arbitrary value to eliminate the compiler warning
  char * const demangledName = abi::__cxa_demangle( name.c_str(), nullptr, nullptr, &status );

  std::string const result = (status == 0) ? demangledName : name;

  std::free( demangledName );

  return result;
}

///////////////////////////////////////////////////////////////////////////////
std::string calculateSize( size_t const bytes )
{
  char const * suffix;
  uint shift;
  if( bytes >> 30 != 0 )
  {
    suffix = "GB";
    shift = 30;
  }
  else if( bytes >> 20 != 0 )
  {
    suffix = "MB";
    shift = 20;
  }
  else
  {
    suffix = "KB";
    shift = 10;
  }

  double const units = double( bytes ) / ( 1 << shift );

  char result[10];
  std::snprintf( result, 10, "%.1f %s", units, suffix );
  return result;
}

} // namespace LvArray
