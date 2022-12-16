#include "qr.hpp"
#include "backendHelpers.hpp"

extern "C"
{

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_SGEQRF LVARRAY_LAPACK_FORTRAN_MANGLE( sgeqrf )
void LVARRAY_SGEQRF( 
  LvArray::dense::DenseInt const * N,
  LvArray::dense::DenseInt const * M,
  float * A,
  LvArray::dense::DenseInt const * LDA,
  float * TAU,
  float * WORK,
  LvArray::dense::DenseInt const * LWORK,
  LvArray::dense::DenseInt * INFO );

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_DGEQRF LVARRAY_LAPACK_FORTRAN_MANGLE( dgeqrf )
void LVARRAY_DGEQRF( 
  LvArray::dense::DenseInt const * N,
  LvArray::dense::DenseInt const * M,
  double * A,
  LvArray::dense::DenseInt const * LDA,
  double * TAU,
  double * WORK,
  LvArray::dense::DenseInt const * LWORK,
  LvArray::dense::DenseInt * INFO );

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_CGEQRF LVARRAY_LAPACK_FORTRAN_MANGLE( cgeqrf )
void LVARRAY_CGEQRF( 
  LvArray::dense::DenseInt const * N,
  LvArray::dense::DenseInt const * M,
  std::complex< float > * A,
  LvArray::dense::DenseInt const * LDA,
  std::complex< float > * TAU,
  std::complex< float > * WORK,
  LvArray::dense::DenseInt const * LWORK,
  LvArray::dense::DenseInt * INFO );

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_ZGEQRF LVARRAY_LAPACK_FORTRAN_MANGLE( zgeqrf )
void LVARRAY_ZGEQRF( 
  LvArray::dense::DenseInt const * N,
  LvArray::dense::DenseInt const * M,
  std::complex< double > * A,
  LvArray::dense::DenseInt const * LDA,
  std::complex< double > * TAU,
  std::complex< double > * WORK,
  LvArray::dense::DenseInt const * LWORK,
  LvArray::dense::DenseInt * INFO );

} // extern "C"

namespace LvArray
{
namespace dense
{

template< typename T >
void geqrf(
    BuiltInBackends const backend,
    Matrix< T > const & A,
    Vector< T > const & tau,
    Workspace< T > & workspace );
{
  LVARRAY_ERROR_IF( !A.isColumnMajor, "The matrix A must be column major." );
  
  DenseInt const N = A.nCols;
  DenseInt const M = A.nRows;
  DenseInt const LDA = A.stride;

  LVARRAY_ERROR_IF_LT( LDA, M );

  LVARRAY_ERROR_IF_LT( tau.size, std::min( M, N ) );

  DenseInt const NRHS = B.nCols;
  DenseInt const LDA = A.stride;
  DenseInt const LDB = B.stride;
  DenseInt INFO = 0;

  if( backend == BuiltInBackends::LAPACK )
  {
    if( std::is_same< T, float >::value )
    {
      LVARRAY_SGESV(
        &N,
        &NRHS,
        reinterpret_cast< float * >( A.data ),
        &LDA,
        pivots.data,
        reinterpret_cast< float * >( B.data ),
        &LDB,
        &INFO );
    }
    if( std::is_same< T, double >::value )
    {
      LVARRAY_DGESV(
        &N,
        &NRHS,
        reinterpret_cast< double * >( A.data ),
        &LDA,
        pivots.data,
        reinterpret_cast< double * >( B.data ),
        &LDB,
        &INFO );
    }
    if( IsComplexT< T, float > )
    {
      LVARRAY_CGESV(
        &N,
        &NRHS,
        reinterpret_cast< std::complex< float > * >( A.data ),
        &LDA,
        pivots.data,
        reinterpret_cast< std::complex< float > * >( B.data ),
        &LDB,
        &INFO );
    }
    if( IsComplexT< T, double > )
    {
      LVARRAY_ZGESV(
        &N,
        &NRHS,
        reinterpret_cast< std::complex< double > * >( A.data ),
        &LDA,
        pivots.data,
        reinterpret_cast< std::complex< double > * >( B.data ),
        &LDB,
        &INFO );
    }
  }
#if defined( LVARRAY_USE_MAGMA )
  else if( backend == BuiltInBackends::MAGMA )
  {
    if( std::is_same< T, float >::value )
    {
      magma_sgesv(
        N,
        NRHS,
        reinterpret_cast< float * >( A.data ),
        LDA,
        pivots.data,
        reinterpret_cast< float * >( B.data ),
        LDB,
        &INFO );
    }
    if( std::is_same< T, double >::value )
    {
      magma_dgesv(
        N,
        NRHS,
        reinterpret_cast< double * >( A.data ),
        LDA,
        pivots.data,
        reinterpret_cast< double * >( B.data ),
        LDB,
        &INFO );
    }
    if( IsComplexT< T, float > )
    {
      magma_cgesv(
        N,
        NRHS,
        reinterpret_cast< magmaFloatComplex * >( A.data ),
        LDA,
        pivots.data,
        reinterpret_cast< magmaFloatComplex * >( B.data ),
        LDB,
        &INFO );
    }
    if( IsComplexT< T, double > )
    {
      magma_zgesv(
        N,
        NRHS,
        reinterpret_cast< magmaDoubleComplex * >( A.data ),
        LDA,
        pivots.data,
        reinterpret_cast< magmaDoubleComplex * >( B.data ),
        LDB,
        &INFO );
    }
  }
  else if( backend == BuiltInBackends::MAGMA_GPU )
  {
    if( std::is_same< T, float >::value )
    {
      magma_sgesv_gpu(
        N,
        NRHS,
        reinterpret_cast< float * >( A.data ),
        LDA,
        pivots.data,
        reinterpret_cast< float * >( B.data ),
        LDB,
        &INFO );
    }
    if( std::is_same< T, double >::value )
    {
      magma_dgesv_gpu(
        N,
        NRHS,
        reinterpret_cast< double * >( A.data ),
        LDA,
        pivots.data,
        reinterpret_cast< double * >( B.data ),
        LDB,
        &INFO );
    }
    if( IsComplexT< T, float > )
    {
      magma_cgesv_gpu(
        N,
        NRHS,
        reinterpret_cast< magmaFloatComplex * >( A.data ),
        LDA,
        pivots.data,
        reinterpret_cast< magmaFloatComplex * >( B.data ),
        LDB,
        &INFO );
    }
    if( IsComplexT< T, double > )
    {
      magma_zgesv_gpu(
        N,
        NRHS,
        reinterpret_cast< magmaDoubleComplex * >( A.data ),
        LDA,
        pivots.data,
        reinterpret_cast< magmaDoubleComplex * >( B.data ),
        LDB,
        &INFO );
    }
  }
#endif
  else
  {
    LVARRAY_ERROR( "Unknown built in backend: " << static_cast< int >( backend ) );
  }

  LVARRAY_ERROR_IF( INFO < 0, "The " << -INFO << "-th argument had an illegal value." );
  LVARRAY_ERROR_IF( INFO > 0, "The factorization has been completed but U( " << INFO - 1 << ", " << INFO - 1 <<
                              " ) is exactly zero so the solution could not be computed." );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template void gesv< float >(
  BuiltInBackends const backend,
  Matrix< float > const & A,
  Matrix< float > const & B,
  Vector< DenseInt > const & pivots );

////////////////////////////////////////////////////////////////////////////////////////////////////
template void gesv< double >(
  BuiltInBackends const backend,
  Matrix< double > const & A,
  Matrix< double > const & B,
  Vector< DenseInt > const & pivots );

////////////////////////////////////////////////////////////////////////////////////////////////////
template void gesv< std::complex< float > >(
  BuiltInBackends const backend,
  Matrix< std::complex< float > > const & A,
  Matrix< std::complex< float > > const & B,
  Vector< DenseInt > const & pivots );

////////////////////////////////////////////////////////////////////////////////////////////////////
template void gesv< std::complex< double > >(
  BuiltInBackends const backend,
  Matrix< std::complex< double > > const & A,
  Matrix< std::complex< double > > const & B,
  Vector< DenseInt > const & pivots );


} // namespace dense
} // namespace LvArray
