#pragma once

#include "../Array.hpp"
#include "../ChaiBuffer.hpp"

#include <complex>

namespace LvArray
{
namespace dense
{
namespace internal
{

/**
 * TODO make a complex type and add it to the main LvArray. Make a uniform way of interacting with various complex number implementations.
 */
template< typename T >
struct RealVersion
{
  using Type = T;
};

/**
 *
 */
template< typename T >
struct RealVersion< std::complex< T > >
{
  using Type = T;
};

} // namespace internal

/**
 *
 */
enum class SymmetricMatrixStorageType
{
  UPPER_TRIANGULAR,
  LOWER_TRIANGULAR,
};

/**
 * TODO: move to internal namespace
 */
char const * getOption( SymmetricMatrixStorageType const option );

/**
 *
 */
template< typename T >
using RealVersion = typename internal::RealVersion< T >::Type;


using DenseInt = int;

/**
 *
 */
template< typename T >
struct Matrix
{
  /**
   *
   */
  template< typename INDEX_TYPE  >
  Matrix( ArraySlice< T, 2, 1, INDEX_TYPE > const & slice ):
    nRows{ integerConversion< DenseInt >( slice.size( 1 ) ) },
    nCols{ integerConversion< DenseInt >( slice.size( 0 ) ) },
    stride{ integerConversion< DenseInt >( slice.strides()[ 0 ] ) },
    data{ &slice( 0, 0 ) }
  {}

  /**
   *
   */
  bool isSquare() const
  {
    return nRows == nCols;
  }

  DenseInt const nRows;
  DenseInt const nCols;
  DenseInt const stride;
  T * const data;
};

/**
 *
 */
template< typename T >
struct Vector
{
  template< int USD, typename INDEX_TYPE >
  Vector( ArraySlice< T, 1, USD, INDEX_TYPE > const & slice ):
    n{ integerConversion< DenseInt >( slice.size() ) },
    stride{ integerConversion< DenseInt >( slice.strides()[ 0 ] )  },
    data{ &slice[ 0 ] }
  {}

  DenseInt const n;
  DenseInt const stride;
  T * const data;
};

/**
 * TODO(corbett5): Make this into a virtual heirarchy so we can get rid of ChaiBuffer here.
 * Also add a version that is only for computing sizes so no dynamic allocation needed.
 * When that is done you can get rid of the constructor here.
 */
template< typename T >
struct Workspace
{
  Workspace()
  {}

  Workspace( std::ptrdiff_t initialSize ):
    work( initialSize ),
    rwork( initialSize ),
    iwork( initialSize )
  {}

  Array< T, 1, RAJA::PERM_I, std::ptrdiff_t, ChaiBuffer > work;

  Array< RealVersion< T >, 1, RAJA::PERM_I, std::ptrdiff_t, ChaiBuffer > rwork;

  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, ChaiBuffer > iwork;
};

} // namespace dense
} // namespace LvArray