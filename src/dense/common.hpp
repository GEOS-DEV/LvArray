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

enum class Operation
{
  NO_OP,
  TRANSPOSE,
  ADJOINT,
};

Operation transposeOp( Operation const op );

/**
 *
 */
template< typename T >
using RealVersion = typename internal::RealVersion< T >::Type;

/**
 *
 */
template< typename T >
static constexpr bool IsComplex = !std::is_same< RealVersion< T >, T >::value;

/**
 *
 */
template< typename T, typename U >
static constexpr bool IsComplexT = IsComplex< T > && std::is_same< RealVersion< T >, U >::value;

/**
 *
 */
template< typename T >
struct Matrix
{
  Matrix(
    typeManipulation::CArray< std::ptrdiff_t, 2 > const & sizesIn,
    typeManipulation::CArray< std::ptrdiff_t, 2 > const & stridesIn,
    T * const dataIn ):
    sizes{ sizesIn },
    strides{ stridesIn },
    data{ dataIn }
  {
    LVARRAY_ERROR_IF_LT( sizes[ 0 ], 0 );
    LVARRAY_ERROR_IF_LT( sizes[ 1 ], 0 );
    LVARRAY_ERROR_IF_LT( strides[ 0 ], 0 );
    LVARRAY_ERROR_IF_LT( strides[ 1 ], 0 );
  }

  Matrix( T & value ):
    sizes{ 1, 1 },
    strides{ 1, 1 },
    data{ &value }
  {}

  Matrix( Matrix< std::remove_const_t< T > > const & src ):
    sizes{ src.sizes },
    strides{ src.strides },
    data{ src.data }
  {}

  bool isSquare() const
  { return sizes[0] == sizes[1]; }

  bool isColumnMajor() const
  { return strides[ 0 ] == 1; }

  bool isRowMajor() const
  { return strides[ 1 ] == 1; }

  bool isContiguous() const
  { return isColumnMajor() || isRowMajor(); }

  std::ptrdiff_t nRows() const
  { return sizes[ 0 ]; }

  std::ptrdiff_t nCols() const
  { return sizes[ 1 ]; }

  Matrix transpose() const
  {
    return Matrix( { sizes[ 1 ], sizes[ 0 ] }, { strides[ 1 ], strides[ 0 ] }, data );
  }

  typeManipulation::CArray< std::ptrdiff_t, 2 > sizes;
  typeManipulation::CArray< std::ptrdiff_t, 2 > strides;
  T * data;
};

template< typename T, typename PERM, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
Matrix< T > toMatrix(
  Array< T, 2, PERM, INDEX_TYPE, BUFFER_TYPE > const & array,
  MemorySpace const space,
  bool const touch )
{
  array.move( space, touch );
  return Matrix< T >( array.dimsArray(), array.stridesArray(), array.data() );
}

/**
 *
 */
template< typename T >
struct Vector
{
  template< int USD, typename INDEX_TYPE >
  Vector( ArraySlice< T, 1, USD, INDEX_TYPE > const & slice ):
    size{ integerConversion< std::ptrdiff_t >( slice.size() ) },
    stride{ integerConversion< std::ptrdiff_t >( slice.stride( 0 ) ) },
    data{ slice.data() }
  {}

  Vector( T & value ):
    size{ 1 },
    stride{ 1 },
    data{ &value }
  {}

  std::ptrdiff_t const size;
  std::ptrdiff_t const stride;
  T * const data;
};

/**
 *
 */
template< typename T >
struct Workspace
{
  virtual ~Workspace()
  {};

  virtual Vector< T > work() = 0;

  virtual Vector< T > work2() = 0;

  virtual Vector< T > work3() = 0;

  virtual Vector< RealVersion< T > > rwork() = 0;

  virtual Vector< int > iwork() = 0;

  virtual void resizeWork( MemorySpace const space, std::ptrdiff_t const newSize ) = 0;

  virtual void resizeWork2( MemorySpace const space, std::ptrdiff_t const newSize ) = 0;

  virtual void resizeWork3( MemorySpace const space, std::ptrdiff_t const newSize ) = 0;

  virtual void resizeRWork( MemorySpace const space, std::ptrdiff_t const newSize ) = 0;

  virtual void resizeIWork( MemorySpace const space, std::ptrdiff_t const newSize ) = 0;
};

/**
 *
 */
template< typename T, template< typename > class BUFFER_TYPE >
struct ArrayWorkspace : public Workspace< T >
{
  ArrayWorkspace()
  {
    m_work.setName( "ArrayWorkspace::m_work" );
    m_work2.setName( "ArrayWorkspace::m_work2" );
    m_work3.setName( "ArrayWorkspace::m_work3" );
    m_rwork.setName( "ArrayWorkspace::m_rwork" );
    m_iwork.setName( "ArrayWorkspace::m_iwork" );
  }

  virtual Vector< T > work() override
  { return m_work.toSlice(); }

  virtual Vector< T > work2() override
  { return m_work2.toSlice(); }

  virtual Vector< T > work3() override
  { return m_work3.toSlice(); }

  virtual Vector< RealVersion< T > > rwork() override
  { return m_rwork.toSlice(); }

  virtual Vector< int > iwork() override
  { return m_iwork.toSlice(); }

  virtual void resizeWork( MemorySpace const space, std::ptrdiff_t const newSize ) override
  {
    m_work.resizeWithoutInitializationOrDestruction( space, newSize );
  }

  virtual void resizeWork2( MemorySpace const space, std::ptrdiff_t const newSize ) override
  {
    m_work2.resizeWithoutInitializationOrDestruction( space, newSize );
  }

  virtual void resizeWork3( MemorySpace const space, std::ptrdiff_t const newSize ) override
  {
    m_work3.resizeWithoutInitializationOrDestruction( space, newSize );
  }
 
  virtual void resizeRWork( MemorySpace const space, std::ptrdiff_t const newSize ) override
  {
    m_rwork.resizeWithoutInitializationOrDestruction( space, newSize );
  }

  virtual void resizeIWork( MemorySpace const space, std::ptrdiff_t const newSize ) override
  {
    m_iwork.resizeWithoutInitializationOrDestruction( space, newSize );
  }

private:
  Array< T, 1, RAJA::PERM_I, std::ptrdiff_t, BUFFER_TYPE > m_work;

  Array< T, 1, RAJA::PERM_I, std::ptrdiff_t, BUFFER_TYPE > m_work2;

  Array< T, 1, RAJA::PERM_I, std::ptrdiff_t, BUFFER_TYPE > m_work3;

  Array< RealVersion< T >, 1, RAJA::PERM_I, std::ptrdiff_t, BUFFER_TYPE > m_rwork;

  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, BUFFER_TYPE > m_iwork;
};

/**
 *
 */
template< typename T >
struct OptimalSizeCalculation : public Workspace< T >
{
  OptimalSizeCalculation()
  {}

  virtual Vector< T > work() override
  { return m_work; }

  virtual Vector< T > work2() override
  { return m_work2; }

  virtual Vector< T > work3() override
  { return m_work3; }

  virtual Vector< RealVersion< T > > rwork() override
  { return m_rwork; }

  virtual Vector< int > iwork() override
  { return m_iwork; }

  virtual void resizeWork( MemorySpace const LVARRAY_UNUSED_ARG( space ), std::ptrdiff_t const LVARRAY_UNUSED_ARG( newSize ) ) override
  { LVARRAY_ERROR( "Not supported by OptimalSizeCalculation." ); }

  virtual void resizeWork2( MemorySpace const LVARRAY_UNUSED_ARG( space ), std::ptrdiff_t const LVARRAY_UNUSED_ARG( newSize ) ) override
  { LVARRAY_ERROR( "Not supported by OptimalSizeCalculation." ); }

  virtual void resizeWork3( MemorySpace const LVARRAY_UNUSED_ARG( space ), std::ptrdiff_t const LVARRAY_UNUSED_ARG( newSize ) ) override
  { LVARRAY_ERROR( "Not supported by OptimalSizeCalculation." ); }

  virtual void resizeRWork( MemorySpace const LVARRAY_UNUSED_ARG( space ), std::ptrdiff_t const LVARRAY_UNUSED_ARG( newSize ) ) override
  { LVARRAY_ERROR( "Not supported by OptimalSizeCalculation." ); }

  virtual void resizeIWork( MemorySpace const LVARRAY_UNUSED_ARG( space ), std::ptrdiff_t const LVARRAY_UNUSED_ARG( newSize ) ) override
  { LVARRAY_ERROR( "Not supported by OptimalSizeCalculation." ); }

  std::ptrdiff_t optimalWorkSize() const
  { return static_cast< std::ptrdiff_t >( m_work.real() ); }

  std::ptrdiff_t optimalRWorkSize() const
  { return static_cast< std::ptrdiff_t >( m_rwork ); }

  std::ptrdiff_t optimalIWorkSize() const
  { return m_iwork; }

private:
  T m_work { -1 };

  T m_work2 { -1 };

  T m_work3 { -1 };

  RealVersion< T > m_rwork { -1 };

  int m_iwork { -1 };
};

} // namespace dense
} // namespace LvArray