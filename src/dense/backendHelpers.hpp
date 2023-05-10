#pragma once

#include <complex>

/// This macro provide a flexible interface for Fortran naming convention for compiled objects
// #ifdef FORTRAN_MANGLE_NO_UNDERSCORE
#define LVARRAY_LAPACK_FORTRAN_MANGLE( name ) name
// #else
// #define LVARRAY_LAPACK_FORTRAN_MANGLE( name ) name ## _
// #endif

namespace LvArray
{
namespace dense
{

template< typename T >
struct TypeDispatch
{};

template<>
struct TypeDispatch< float >
{
  template< typename F_FLOAT, typename F_DOUBLE, typename F_CFLOAT, typename F_CDOUBLE, typename ... ARGS >
  static constexpr auto dispatch(
    F_FLOAT && fFloat,
    F_DOUBLE &&,
    F_CFLOAT &&,
    F_CDOUBLE &&,
    ARGS && ... args )
  {
    return fFloat( std::forward< ARGS >( args ) ... );
  }
};

template<>
struct TypeDispatch< double >
{
  template< typename F_FLOAT, typename F_DOUBLE, typename F_CFLOAT, typename F_CDOUBLE, typename ... ARGS >
  static constexpr auto dispatch(
    F_FLOAT &&,
    F_DOUBLE && fDouble,
    F_CFLOAT &&,
    F_CDOUBLE &&,
    ARGS && ... args )
  {
    return fDouble( std::forward< ARGS >( args ) ... );
  }
};

template<>
struct TypeDispatch< std::complex< float > >
{
  template< typename F_FLOAT, typename F_DOUBLE, typename F_CFLOAT, typename F_CDOUBLE, typename ... ARGS >
  static constexpr auto dispatch(
    F_FLOAT &&,
    F_DOUBLE &&,
    F_CFLOAT && fCFloat,
    F_CDOUBLE &&,
    ARGS && ... args )
  {
    return fCFloat( std::forward< ARGS >( args ) ... );
  }
};

template<>
struct TypeDispatch< std::complex< double > >
{
  template< typename F_FLOAT, typename F_DOUBLE, typename F_CFLOAT, typename F_CDOUBLE, typename ... ARGS >
  static constexpr auto dispatch(
    F_FLOAT &&,
    F_DOUBLE &&,
    F_CFLOAT &&,
    F_CDOUBLE && fCDouble,
    ARGS && ... args )
  {
    return fCDouble( std::forward< ARGS >( args ) ... );
  }
};

} // namespace dense
} // namespace LvArray
