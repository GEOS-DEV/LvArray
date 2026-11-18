/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Tests for the triple-product tensor operations R = P * B * Q (and transposed variants).

// Source includes
#include "tensorOps.hpp"
#include "Array.hpp"
#include "testUtils.hpp"
#include "output.hpp"
#include "testTensorOpsCommon.hpp"

// TPL includes
#include <gtest/gtest.h>

namespace LvArray
{
namespace testing
{

/**
 * Sizes follow the general operator Rij_eq_PikBklQlj< I, J, K, L >:
 *   P is I x J, B is J x K, Q is K x L, R is I x L.
 * The square-B operators use N = J for the original space (B is N x N) and M = I for the projected
 * space (R is M x M), with P / Q of size N x M (Pki variants) or M x N (Pik variants).
 */
template< typename T_I_J_K_L_POLICY_TUPLE >
class TripleProductTest : public ::testing::Test
{
public:
  using T = std::tuple_element_t< 0, T_I_J_K_L_POLICY_TUPLE >;
  static constexpr std::ptrdiff_t I = std::tuple_element_t< 1, T_I_J_K_L_POLICY_TUPLE > {};
  static constexpr std::ptrdiff_t J = std::tuple_element_t< 2, T_I_J_K_L_POLICY_TUPLE > {};
  static constexpr std::ptrdiff_t K = std::tuple_element_t< 3, T_I_J_K_L_POLICY_TUPLE > {};
  static constexpr std::ptrdiff_t L = std::tuple_element_t< 4, T_I_J_K_L_POLICY_TUPLE > {};
  using POLICY = std::tuple_element_t< 5, T_I_J_K_L_POLICY_TUPLE >;

  static constexpr std::ptrdiff_t N = J;
  static constexpr std::ptrdiff_t M = I;

  // Runs one operator on every combination of destination / operand storage that matters:
  // all C arrays, all PERM_IJK slices, all PERM_KJI slices, and a mixed set.
  // OP( dst, p, b, q ) performs the operation, EXPECTED is the reference result,
  // DST_INIT is the value dst is filled with beforehand (relevant for the Rij_add_* operators).
  #define TRIPLE_PRODUCT_TEST_LAYOUTS( DST_ROWS, DST_COLS, P_ROWS, P_COLS, B_ROWS, B_COLS, Q_ROWS, Q_COLS, Q_SEED, SYM_B, OP ) \
    ArrayT< T, RAJA::PERM_IJK > dst_IJK { 1, DST_ROWS, DST_COLS }; \
    ArrayT< T, RAJA::PERM_KJI > dst_KJI { 1, DST_ROWS, DST_COLS }; \
    ArrayT< T, RAJA::PERM_IJK > p_IJK { 1, P_ROWS, P_COLS }; \
    ArrayT< T, RAJA::PERM_KJI > p_KJI { 1, P_ROWS, P_COLS }; \
    ArrayT< T, RAJA::PERM_IJK > b_IJK { 1, B_ROWS, B_COLS }; \
    ArrayT< T, RAJA::PERM_KJI > b_KJI { 1, B_ROWS, B_COLS }; \
    ArrayT< T, RAJA::PERM_IJK > q_IJK { 1, Q_ROWS, Q_COLS }; \
    ArrayT< T, RAJA::PERM_KJI > q_KJI { 1, Q_ROWS, Q_COLS }; \
    fill( p_IJK.toSlice(), m_pSeed ); fill( p_KJI.toSlice(), m_pSeed ); \
    fill( b_IJK.toSlice(), m_bSeed ); fill( b_KJI.toSlice(), m_bSeed ); \
    if( SYM_B ) { symmetrize< B_ROWS >( b_IJK[ 0 ] ); symmetrize< B_ROWS >( b_KJI[ 0 ] ); } \
    fill( q_IJK.toSlice(), Q_SEED ); fill( q_KJI.toSlice(), Q_SEED ); \
    ArrayViewT< T, 3, 2 > const dstV_IJK = dst_IJK.toView(); \
    ArrayViewT< T, 3, 0 > const dstV_KJI = dst_KJI.toView(); \
    ArrayViewT< T const, 3, 2 > const pV_IJK = p_IJK.toViewConst(); \
    ArrayViewT< T const, 3, 0 > const pV_KJI = p_KJI.toViewConst(); \
    ArrayViewT< T const, 3, 2 > const bV_IJK = b_IJK.toViewConst(); \
    ArrayViewT< T const, 3, 0 > const bV_KJI = b_KJI.toViewConst(); \
    ArrayViewT< T const, 3, 2 > const qV_IJK = q_IJK.toViewConst(); \
    ArrayViewT< T const, 3, 0 > const qV_KJI = q_KJI.toViewConst(); \
    T pL[ P_ROWS ][ P_COLS ]; fill( pL, m_pSeed ); \
    T bL[ B_ROWS ][ B_COLS ]; fill( bL, m_bSeed ); \
    if( SYM_B ) { symmetrize< B_ROWS >( bL ); } \
    T qL[ Q_ROWS ][ Q_COLS ]; fill( qL, Q_SEED ); \
    std::ptrdiff_t const dstSeed = m_dstSeed; \
    forall< POLICY >( 1, [=] LVARRAY_HOST_DEVICE ( int ) \
        { \
          T dstL[ DST_ROWS ][ DST_COLS ]; \
          _RUN( dstL, pL, bL, qL, OP ); \
          _RUN( dstV_IJK[ 0 ], pV_IJK[ 0 ], bV_IJK[ 0 ], qV_IJK[ 0 ], OP ); \
          _RUN( dstV_KJI[ 0 ], pV_KJI[ 0 ], bV_KJI[ 0 ], qV_KJI[ 0 ], OP ); \
          _RUN( dstV_KJI[ 0 ], pL, bV_IJK[ 0 ], qV_KJI[ 0 ], OP ); \
          _RUN( dstL, pV_IJK[ 0 ], bL, qV_KJI[ 0 ], OP ); \
        } )

  #define _RUN( dst, p, b, q, OP ) \
    fill( dst, dstSeed ); \
    OP( dst, p, b, q ); \
    CHECK_EQUALITY_2D( std::remove_reference_t< decltype( expected ) >::rows, \
                       std::remove_reference_t< decltype( expected ) >::cols, dst, expected.data )

  // Copy the upper triangle of the square matrix @p b into its lower triangle.
  template< std::ptrdiff_t SIZE, typename MATRIX >
  static void symmetrize( MATRIX && b )
  {
    for( std::ptrdiff_t k = 0; k < SIZE; ++k )
      for( std::ptrdiff_t l = 0; l < k; ++l )
        b[ k ][ l ] = b[ l ][ k ];
  }

  template< std::ptrdiff_t ROWS, std::ptrdiff_t COLS >
  struct Result
  {
    static constexpr std::ptrdiff_t rows = ROWS;
    static constexpr std::ptrdiff_t cols = COLS;
    T data[ ROWS ][ COLS ];
  };

  // P: I x J, B: J x K, Q: K x L.
  void testPikBklQlj()
  {
    T p[ I ][ J ]; fill( p, m_pSeed );
    T b[ J ][ K ]; fill( b, m_bSeed );
    T q[ K ][ L ]; fill( q, m_qSeed );
    Result< I, L > expected;
    for( std::ptrdiff_t i = 0; i < I; ++i )
      for( std::ptrdiff_t j = 0; j < L; ++j )
      {
        T sum = 0;
        for( std::ptrdiff_t k = 0; k < J; ++k )
          for( std::ptrdiff_t l = 0; l < K; ++l )
            sum += p[ i ][ k ] * b[ k ][ l ] * q[ l ][ j ];
        expected.data[ i ][ j ] = sum;
      }

    #define _OP( dst, p, b, q ) tensorOps::Rij_eq_PikBklQlj< I, J, K, L >( dst, p, b, q )
    TRIPLE_PRODUCT_TEST_LAYOUTS( I, L, I, J, J, K, K, L, m_qSeed, false, _OP );
    #undef _OP
  }

  // P, Q: N x M, B: N x N, R: M x M.
  template< bool ADD, bool SAME_PQ >
  void testPkiBklQlj()
  {
    T p[ N ][ M ]; fill( p, m_pSeed );
    T b[ N ][ N ]; fill( b, m_bSeed );
    T q[ N ][ M ]; fill( q, SAME_PQ ? m_pSeed : m_qSeed );
    T init[ M ][ M ]; fill( init, m_dstSeed );
    Result< M, M > expected;
    for( std::ptrdiff_t i = 0; i < M; ++i )
      for( std::ptrdiff_t j = 0; j < M; ++j )
      {
        T sum = ADD ? init[ i ][ j ] : 0;
        for( std::ptrdiff_t k = 0; k < N; ++k )
          for( std::ptrdiff_t l = 0; l < N; ++l )
            sum += p[ k ][ i ] * b[ k ][ l ] * q[ l ][ j ];
        expected.data[ i ][ j ] = sum;
      }

    if constexpr( SAME_PQ )
    {
      #define _OP( dst, p, b, q ) \
        if constexpr( ADD ) tensorOps::Rij_add_PkiBklPlj< N, M >( dst, p, b ); \
        else tensorOps::Rij_eq_PkiBklPlj< N, M >( dst, p, b )
      TRIPLE_PRODUCT_TEST_LAYOUTS( M, M, N, M, N, N, N, M, m_pSeed, false, _OP );
      #undef _OP
    }
    else
    {
      #define _OP( dst, p, b, q ) \
        if constexpr( ADD ) tensorOps::Rij_add_PkiBklQlj< N, M >( dst, p, b, q ); \
        else tensorOps::Rij_eq_PkiBklQlj< N, M >( dst, p, b, q )
      TRIPLE_PRODUCT_TEST_LAYOUTS( M, M, N, M, N, N, N, M, m_qSeed, false, _OP );
      #undef _OP
    }
  }

  // P, Q: M x N, B: N x N, R: M x M.
  template< bool ADD, bool SAME_PQ >
  void testPikBklQjl()
  {
    T p[ M ][ N ]; fill( p, m_pSeed );
    T b[ N ][ N ]; fill( b, m_bSeed );
    T q[ M ][ N ]; fill( q, SAME_PQ ? m_pSeed : m_qSeed );
    T init[ M ][ M ]; fill( init, m_dstSeed );
    Result< M, M > expected;
    for( std::ptrdiff_t i = 0; i < M; ++i )
      for( std::ptrdiff_t j = 0; j < M; ++j )
      {
        T sum = ADD ? init[ i ][ j ] : 0;
        for( std::ptrdiff_t k = 0; k < N; ++k )
          for( std::ptrdiff_t l = 0; l < N; ++l )
            sum += p[ i ][ k ] * b[ k ][ l ] * q[ j ][ l ];
        expected.data[ i ][ j ] = sum;
      }

    if constexpr( SAME_PQ )
    {
      #define _OP( dst, p, b, q ) \
        if constexpr( ADD ) tensorOps::Rij_add_PikBklPjl< N, M >( dst, p, b ); \
        else tensorOps::Rij_eq_PikBklPjl< N, M >( dst, p, b )
      TRIPLE_PRODUCT_TEST_LAYOUTS( M, M, M, N, N, N, M, N, m_pSeed, false, _OP );
      #undef _OP
    }
    else
    {
      #define _OP( dst, p, b, q ) \
        if constexpr( ADD ) tensorOps::Rij_add_PikBklQjl< N, M >( dst, p, b, q ); \
        else tensorOps::Rij_eq_PikBklQjl< N, M >( dst, p, b, q )
      TRIPLE_PRODUCT_TEST_LAYOUTS( M, M, M, N, N, N, M, N, m_qSeed, false, _OP );
      #undef _OP
    }
  }

  // Symmetric-result variants: B: N x N symmetric, R: M x M dense.
  // PKI: P is N x M and R = P^T B P. Otherwise P is M x N and R = P B P^T.
  template< bool ADD, bool PKI >
  void testSymPBP()
  {
    constexpr std::ptrdiff_t P_ROWS = PKI ? N : M;
    constexpr std::ptrdiff_t P_COLS = PKI ? M : N;
    T p[ P_ROWS ][ P_COLS ]; fill( p, m_pSeed );
    T b[ N ][ N ]; fill( b, m_bSeed ); symmetrize< N >( b );
    T init[ M ][ M ]; fill( init, m_dstSeed ); // not symmetric: the add variants must not assume it
    Result< M, M > expected;
    for( std::ptrdiff_t i = 0; i < M; ++i )
      for( std::ptrdiff_t j = 0; j < M; ++j )
      {
        T sum = ADD ? init[ i ][ j ] : 0;
        for( std::ptrdiff_t k = 0; k < N; ++k )
          for( std::ptrdiff_t l = 0; l < N; ++l )
            sum += PKI ? p[ k ][ i ] * b[ k ][ l ] * p[ l ][ j ]
                       : p[ i ][ k ] * b[ k ][ l ] * p[ j ][ l ];
        expected.data[ i ][ j ] = sum;
      }

    if constexpr( PKI )
    {
      #define _OP( dst, p, b, q ) \
        if constexpr( ADD ) tensorOps::symRij_add_PkiBklPlj< N, M >( dst, p, b ); \
        else tensorOps::symRij_eq_PkiBklPlj< N, M >( dst, p, b )
      TRIPLE_PRODUCT_TEST_LAYOUTS( M, M, N, M, N, N, N, M, m_pSeed, true, _OP );
      #undef _OP
    }
    else
    {
      #define _OP( dst, p, b, q ) \
        if constexpr( ADD ) tensorOps::symRij_add_PikBklPjl< N, M >( dst, p, b ); \
        else tensorOps::symRij_eq_PikBklPjl< N, M >( dst, p, b )
      TRIPLE_PRODUCT_TEST_LAYOUTS( M, M, M, N, N, N, M, N, m_pSeed, true, _OP );
      #undef _OP
    }
  }

  #undef _RUN
  #undef TRIPLE_PRODUCT_TEST_LAYOUTS

private:
  std::ptrdiff_t const m_pSeed = 0;
  std::ptrdiff_t const m_bSeed = 100;
  std::ptrdiff_t const m_qSeed = 200;
  std::ptrdiff_t const m_dstSeed = 300;
};

using TripleProductTestTypes = ::testing::Types<
  std::tuple< double,
              std::integral_constant< int, 2 >,
              std::integral_constant< int, 3 >,
              std::integral_constant< int, 4 >,
              std::integral_constant< int, 5 >,
              serialPolicy >
  , std::tuple< int,
                std::integral_constant< int, 3 >,
                std::integral_constant< int, 4 >,
                std::integral_constant< int, 2 >,
                std::integral_constant< int, 3 >,
                serialPolicy >
  , std::tuple< double,
                std::integral_constant< int, 3 >,
                std::integral_constant< int, 3 >,
                std::integral_constant< int, 3 >,
                std::integral_constant< int, 3 >,
                serialPolicy >

#if ( defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP) ) && defined(LVARRAY_USE_CHAI)
  , std::tuple< double,
                std::integral_constant< int, 2 >,
                std::integral_constant< int, 3 >,
                std::integral_constant< int, 4 >,
                std::integral_constant< int, 5 >,
                parallelDevicePolicy< 32 > >
  , std::tuple< double,
                std::integral_constant< int, 3 >,
                std::integral_constant< int, 3 >,
                std::integral_constant< int, 3 >,
                std::integral_constant< int, 3 >,
                parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( TripleProductTest, TripleProductTestTypes, );

TYPED_TEST( TripleProductTest, Rij_eq_PikBklQlj )
{ this->testPikBklQlj(); }

TYPED_TEST( TripleProductTest, Rij_eq_PkiBklQlj )
{ this->template testPkiBklQlj< false, false >(); }

TYPED_TEST( TripleProductTest, Rij_add_PkiBklQlj )
{ this->template testPkiBklQlj< true, false >(); }

TYPED_TEST( TripleProductTest, Rij_eq_PkiBklPlj )
{ this->template testPkiBklQlj< false, true >(); }

TYPED_TEST( TripleProductTest, Rij_add_PkiBklPlj )
{ this->template testPkiBklQlj< true, true >(); }

TYPED_TEST( TripleProductTest, Rij_eq_PikBklQjl )
{ this->template testPikBklQjl< false, false >(); }

TYPED_TEST( TripleProductTest, Rij_add_PikBklQjl )
{ this->template testPikBklQjl< true, false >(); }

TYPED_TEST( TripleProductTest, Rij_eq_PikBklPjl )
{ this->template testPikBklQjl< false, true >(); }

TYPED_TEST( TripleProductTest, Rij_add_PikBklPjl )
{ this->template testPikBklQjl< true, true >(); }

TYPED_TEST( TripleProductTest, symRij_eq_PkiBklPlj )
{ this->template testSymPBP< false, true >(); }

TYPED_TEST( TripleProductTest, symRij_add_PkiBklPlj )
{ this->template testSymPBP< true, true >(); }

TYPED_TEST( TripleProductTest, symRij_eq_PikBklPjl )
{ this->template testSymPBP< false, false >(); }

TYPED_TEST( TripleProductTest, symRij_add_PikBklPjl )
{ this->template testSymPBP< true, false >(); }

} // namespace testing
} // namespace LvArray
