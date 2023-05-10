#include "linearSolve.hpp"
#include "backendHelpers.hpp"

extern "C"
{

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_SGESV LVARRAY_LAPACK_FORTRAN_MANGLE( sgesv )
void LVARRAY_SGESV( 
  LvArray::dense::DenseInt const * N,
  LvArray::dense::DenseInt const * NRHS,
  float * A,
  LvArray::dense::DenseInt const * LDA,
  LvArray::dense::DenseInt * IPIV,
  float * B,
  LvArray::dense::DenseInt const * LDB,
  LvArray::dense::DenseInt * INFO );

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_DGESV LVARRAY_LAPACK_FORTRAN_MANGLE( dgesv )
void LVARRAY_DGESV( 
  LvArray::dense::DenseInt const * N,
  LvArray::dense::DenseInt const * NRHS,
  double * A,
  LvArray::dense::DenseInt const * LDA,
  LvArray::dense::DenseInt * IPIV,
  double * B,
  LvArray::dense::DenseInt const * LDB,
  LvArray::dense::DenseInt * INFO );

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_CGESV LVARRAY_LAPACK_FORTRAN_MANGLE( cgesv )
void LVARRAY_CGESV( 
  LvArray::dense::DenseInt const * N,
  LvArray::dense::DenseInt const * NRHS,
  std::complex< float > * A,
  LvArray::dense::DenseInt const * LDA,
  LvArray::dense::DenseInt * IPIV,
  std::complex< float > * B,
  LvArray::dense::DenseInt const * LDB,
  LvArray::dense::DenseInt * INFO );

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_ZGESV LVARRAY_LAPACK_FORTRAN_MANGLE( zgesv )
void LVARRAY_ZGESV( 
  LvArray::dense::DenseInt const * N,
  LvArray::dense::DenseInt const * NRHS,
  std::complex< double > * A,
  LvArray::dense::DenseInt const * LDA,
  LvArray::dense::DenseInt * IPIV,
  std::complex< double > * B,
  LvArray::dense::DenseInt const * LDB,
  LvArray::dense::DenseInt * INFO );

} // extern "C"

namespace LvArray
{
namespace dense
{

template< typename T >
void gesv(
  BuiltInBackends const backend,
  Matrix< T > const & A,
  Matrix< T > const & B,
  Vector< DenseInt > const & pivots )
{
  LVARRAY_ERROR_IF( !A.isSquare(), "The matrix A must be square." );
  LVARRAY_ERROR_IF( !A.isColumnMajor(), "The matrix A must be column major." );

  LVARRAY_ERROR_IF_NE( A.sizes[ 0 ], B.sizes[ 0 ] );
  LVARRAY_ERROR_IF( !B.isColumnMajor(), "The matrix B must be column major." );

  LVARRAY_ERROR_IF_NE( pivots.size, A.sizes[ 0 ] );

  DenseInt const N = A.sizes[ 1 ];
  DenseInt const NRHS = B.sizes[ 1 ];
  DenseInt const LDA = A.strides[ 1 ];
  DenseInt const LDB = B.strides[ 1 ];
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
