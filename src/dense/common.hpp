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
enum class BuiltInBackends
{
  LAPACK,
#if defined( LVARRAY_USE_MAGMA )
  MAGMA,
  MAGMA_GPU,
#endif
};

/**
 *
 */
MemorySpace getSpaceForBackend( BuiltInBackends const backend );

/**
 *
 */
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
  Matrix( ArraySlice< T, 2, 0, INDEX_TYPE > const & slice ):
    nRows{ integerConversion< DenseInt >( slice.size( 0 ) ) },
    nCols{ integerConversion< DenseInt >( slice.size( 1 ) ) },
    stride{ integerConversion< DenseInt >( slice.stride( 1 ) ) },
    isColumnMajor{ true },
    data{ slice.data() }
  {}

  /**
   *
   */
  template< typename INDEX_TYPE, int USD >
  Matrix( ArraySlice< T, 1, USD, INDEX_TYPE > const & slice ):
    nRows{ integerConversion< DenseInt >( slice.size( 1 ) ) },
    nCols{ integerConversion< DenseInt >( 1 ) },
    stride{ integerConversion< DenseInt >( slice.stride( 0 ) ) },
    isColumnMajor{ true },
    data{ slice.data() }
  {}

  /**
   *
   */
  Matrix( T & value ):
    nRows{ 1 },
    nCols{ 1 },
    stride{ 1 },
    isColumnMajor{ true },
    data{ &value }
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
  bool const isColumnMajor;
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
    size{ integerConversion< DenseInt >( slice.size() ) },
    stride{ integerConversion< DenseInt >( slice.stride( 0 ) ) },
    data{ slice.data() }
  {}

  Vector( T & value ):
    size{ 1 },
    stride{ 1 },
    data{ &value }
  {}

  DenseInt const size;
  DenseInt const stride;
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

  virtual Vector< DenseInt > iwork() = 0;

  virtual void resizeWork( MemorySpace const space, DenseInt const newSize ) = 0;

  virtual void resizeWork2( MemorySpace const space, DenseInt const newSize ) = 0;

  virtual void resizeWork3( MemorySpace const space, DenseInt const newSize ) = 0;

  virtual void resizeRWork( MemorySpace const space, DenseInt const newSize ) = 0;

  virtual void resizeIWork( MemorySpace const space, DenseInt const newSize ) = 0;
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

  virtual Vector< DenseInt > iwork() override
  { return m_iwork.toSlice(); }

  virtual void resizeWork( MemorySpace const space, DenseInt const newSize ) override
  {
    m_work.resizeWithoutInitializationOrDestruction( space, newSize );
  }

  virtual void resizeWork2( MemorySpace const space, DenseInt const newSize ) override
  {
    m_work2.resizeWithoutInitializationOrDestruction( space, newSize );
  }

  virtual void resizeWork3( MemorySpace const space, DenseInt const newSize ) override
  {
    m_work3.resizeWithoutInitializationOrDestruction( space, newSize );
  }
 
  virtual void resizeRWork( MemorySpace const space, DenseInt const newSize ) override
  {
    m_rwork.resizeWithoutInitializationOrDestruction( space, newSize );
  }

  virtual void resizeIWork( MemorySpace const space, DenseInt const newSize ) override
  {
    m_iwork.resizeWithoutInitializationOrDestruction( space, newSize );
  }

private:
  Array< T, 1, RAJA::PERM_I, DenseInt, BUFFER_TYPE > m_work;

  Array< T, 1, RAJA::PERM_I, DenseInt, BUFFER_TYPE > m_work2;

  Array< T, 1, RAJA::PERM_I, DenseInt, BUFFER_TYPE > m_work3;

  Array< RealVersion< T >, 1, RAJA::PERM_I, DenseInt, BUFFER_TYPE > m_rwork;

  Array< DenseInt, 1, RAJA::PERM_I, DenseInt, BUFFER_TYPE > m_iwork;
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

  virtual void resizeWork( MemorySpace const LVARRAY_UNUSED_ARG( space ), DenseInt const LVARRAY_UNUSED_ARG( newSize ) ) override
  { LVARRAY_ERROR( "Not supported by OptimalSizeCalculation." ); }

  virtual void resizeWork2( MemorySpace const LVARRAY_UNUSED_ARG( space ), DenseInt const LVARRAY_UNUSED_ARG( newSize ) ) override
  { LVARRAY_ERROR( "Not supported by OptimalSizeCalculation." ); }

  virtual void resizeWork3( MemorySpace const LVARRAY_UNUSED_ARG( space ), DenseInt const LVARRAY_UNUSED_ARG( newSize ) ) override
  { LVARRAY_ERROR( "Not supported by OptimalSizeCalculation." ); }

  virtual void resizeRWork( MemorySpace const LVARRAY_UNUSED_ARG( space ), DenseInt const LVARRAY_UNUSED_ARG( newSize ) ) override
  { LVARRAY_ERROR( "Not supported by OptimalSizeCalculation." ); }

  virtual void resizeIWork( MemorySpace const LVARRAY_UNUSED_ARG( space ), DenseInt const LVARRAY_UNUSED_ARG( newSize ) ) override
  { LVARRAY_ERROR( "Not supported by OptimalSizeCalculation." ); }

  DenseInt optimalWorkSize() const
  { return static_cast< DenseInt >( m_work.real() ); }

  DenseInt optimalRWorkSize() const
  { return static_cast< DenseInt >( m_rwork ); }

  DenseInt optimalIWorkSize() const
  { return m_iwork; }

private:
  T m_work { -1 };

  T m_work2 { -1 };

  T m_work3 { -1 };

  RealVersion< T > m_rwork { -1 };

  DenseInt m_iwork { -1 };
};

} // namespace dense
} // namespace LvArray