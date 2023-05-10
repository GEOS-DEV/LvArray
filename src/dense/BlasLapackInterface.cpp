#include "BlasLapackInterface.hpp"
#include "backendHelpers.hpp"

extern "C"
{

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_SGEMM LVARRAY_LAPACK_FORTRAN_MANGLE( sgemm )
void LVARRAY_SGEMM( 
   char const * TRANSA,
   char const * TRANSB,
   int const * M,
   int const * N,
   int const * K,
   float const * ALPHA,
   float const * A,
   int const * LDA,
   float const * B,
   int const * LDB,
   float const * BETA,
   float * C,
   int const * LDC );

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_DGEMM LVARRAY_LAPACK_FORTRAN_MANGLE( dgemm )
void LVARRAY_DGEMM( 
   char const * TRANSA,
   char const * TRANSB,
   int const * M,
   int const * N,
   int const * K,
   double const * ALPHA,
   double const * A,
   int const * LDA,
   double const * B,
   int const * LDB,
   double const * BETA,
   double * C,
   int const * LDC );
  
////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_CGEMM LVARRAY_LAPACK_FORTRAN_MANGLE( cgemm )
void LVARRAY_CGEMM( 
   char const * TRANSA,
   char const * TRANSB,
   int const * M,
   int const * N,
   int const * K,
   std::complex< float > const * ALPHA,
   std::complex< float > const * A,
   int const * LDA,
   std::complex< float > const * B,
   int const * LDB,
   std::complex< float > const * BETA,
   std::complex< float > * C,
   int const * LDC );

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_ZGEMM LVARRAY_LAPACK_FORTRAN_MANGLE( zgemm )
void LVARRAY_ZGEMM( 
   char const * TRANSA,
   char const * TRANSB,
   int const * M,
   int const * N,
   int const * K,
   std::complex< double > const * ALPHA,
   std::complex< double > const * A,
   int const * LDA,
   std::complex< double > const * B,
   int const * LDB,
   std::complex< double > const * BETA,
   std::complex< double > * C,
   int const * LDC );

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_SGESV LVARRAY_LAPACK_FORTRAN_MANGLE( sgesv )
void LVARRAY_SGESV( 
  int const * N,
  int const * NRHS,
  float * A,
  int const * LDA,
  int * IPIV,
  float * B,
  int const * LDB,
  int * INFO );

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_DGESV LVARRAY_LAPACK_FORTRAN_MANGLE( dgesv )
void LVARRAY_DGESV( 
  int const * N,
  int const * NRHS,
  double * A,
  int const * LDA,
  int * IPIV,
  double * B,
  int const * LDB,
  int * INFO );

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_CGESV LVARRAY_LAPACK_FORTRAN_MANGLE( cgesv )
void LVARRAY_CGESV( 
  int const * N,
  int const * NRHS,
  std::complex< float > * A,
  int const * LDA,
  int * IPIV,
  std::complex< float > * B,
  int const * LDB,
  int * INFO );

////////////////////////////////////////////////////////////////////////////////////////////////////
#define LVARRAY_ZGESV LVARRAY_LAPACK_FORTRAN_MANGLE( zgesv )
void LVARRAY_ZGESV( 
  int const * N,
  int const * NRHS,
  std::complex< double > * A,
  int const * LDA,
  int * IPIV,
  std::complex< double > * B,
  int const * LDB,
  int * INFO );

} // extern "C"

namespace LvArray
{
namespace dense
{

char toLapackChar( Operation const op )
{
  if( op == Operation::NO_OP ) return 'N';
  if( op == Operation::TRANSPOSE ) return 'T';
  if( op == Operation::ADJOINT ) return 'C';

  LVARRAY_ERROR( "Unknown operation: " << int( op ) );
  return '\0';
}


template< typename T >
void BlasLapackInterface< T >::gemm(
  Operation opA,
  Operation opB,
  T const alpha,
  Matrix< T const > const & A,
  Matrix< T const > const & B,
  T const beta, 
  Matrix< T > const & C )
{
  char const TRANSA = toLapackChar( opA );
  char const TRANSB = toLapackChar( opB );
  int const M = C.sizes[ 0 ];
  int const N = C.sizes[ 1 ];
  int const K = opA == Operation::NO_OP ? A.sizes[ 1 ] : A.sizes[ 0 ];
  int const LDA = std::max( std::ptrdiff_t{ 1 }, A.strides[ 1 ] );
  int const LDB = std::max( std::ptrdiff_t{ 1 }, B.strides[ 1 ] );
  int const LDC = std::max( std::ptrdiff_t{ 1 }, C.strides[ 1 ] );

  TypeDispatch< T >::dispatch( LVARRAY_SGEMM, LVARRAY_DGEMM, LVARRAY_CGEMM, LVARRAY_ZGEMM,
    &TRANSA,
    &TRANSB,
    &M,
    &N,
    &K,
    &alpha,
    A.data,
    &LDA,
    B.data,
    &LDB,
    &beta,
    C.data,
    &LDC );
}


template< typename T >
void BlasLapackInterface< T >::gesv(
  Matrix< T > const & A,
  Matrix< T > const & B,
  Vector< int > const & pivots )
{
  int const N = A.sizes[ 0 ];
  int const NRHS = B.sizes[ 1 ];
  int const LDA = A.strides[ 1 ];
  int const LDB = B.strides[ 1 ];
  int INFO = 0;

  TypeDispatch< T >::dispatch( LVARRAY_SGESV, LVARRAY_DGESV, LVARRAY_CGESV, LVARRAY_ZGESV,
    &N,
    &NRHS,
    A.data,
    &LDA,
    pivots.data,
    B.data,
    &LDB,
    &INFO );
  
  LVARRAY_ERROR_IF( INFO < 0, "The " << -INFO << "-th argument had an illegal value." );
  LVARRAY_ERROR_IF( INFO > 0, "The factorization has been completed but U( " << INFO - 1 << ", " << INFO - 1 <<
                              " ) is exactly zero so the solution could not be computed." );
}

template class BlasLapackInterface< float >;
template class BlasLapackInterface< double >;
template class BlasLapackInterface< std::complex< float > >;
template class BlasLapackInterface< std::complex< double > >;

} // namespace dense
} // namespace LvArray