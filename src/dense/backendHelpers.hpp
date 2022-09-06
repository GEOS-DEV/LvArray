#pragma once

#if defined( LVARRAY_USE_MAGMA )
  #include <magma.h>
#endif

/// This macro provide a flexible interface for Fortran naming convention for compiled objects
// #ifdef FORTRAN_MANGLE_NO_UNDERSCORE
#define LVARRAY_LAPACK_FORTRAN_MANGLE( name ) name
// #else
// #define LVARRAY_LAPACK_FORTRAN_MANGLE( name ) name ## _
// #endif