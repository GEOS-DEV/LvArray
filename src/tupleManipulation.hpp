/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file Layout.hpp
 * @brief Contains the implementation of Layout.
 */

#pragma once

#include "Macros.hpp"
#include "typeManipulation.hpp"

#include "camp/tuple.hpp"

namespace camp
{

template< typename T, int N >
struct seq_inc {};

template< typename T, T ... Is, int N >
struct seq_inc< int_seq< T, Is... >, N >
{
  using type = int_seq< T, Is + N ... >;
};

template< typename T, int N >
using seq_inc_t = typename seq_inc< T, N >::type;

template< typename T, T Lower, T Upper >
using make_range = seq_inc_t< make_int_seq_t< T, Upper - Lower >, Lower >;

template< idx_t Lower, idx_t Upper >
using make_idx_range = make_range< idx_t, Lower, Upper >;

namespace detail
{

template< typename Seq >
struct reverse_seq {};

template< typename T, T ... Is >
struct reverse_seq< int_seq< T, Is... > >
{
  using type = int_seq< T, sizeof...(Is) - Is - 1 ... >;
};

} // namespace detail

template< typename Seq >
using reverse_seq_t = typename detail::reverse_seq< Seq >::type;

template< typename T, T N >
using make_int_rseq_t = reverse_seq_t< make_int_seq_t< T, N > >;

template< idx_t N >
using make_idx_rseq_t = make_int_rseq_t< idx_t, N >;

template< typename T >
using idx_rseq_from_t = reverse_seq_t< idx_seq_from_t< T > >;

template< typename T >
inline constexpr idx_t tsize_v = tuple_size< decay< T > >::value;

}

namespace LvArray
{

/**
 * @brief Make function for tuple
 * @tparam Ts argument type list
 * @param args variadic arguments
 * @return the constructed tuple
 */
template< typename ... Ts >
LVARRAY_HOST_DEVICE constexpr
auto
makeTuple( Ts && ... args )
{
  return camp::make_tuple( LVARRAY_FWD( args )... );
}

namespace internal
{

template< camp::idx_t ... Is, typename T >
LVARRAY_HOST_DEVICE constexpr
auto
makeTupleFromPtr( T const * const vals, camp::idx_seq< Is... > )
{
  return camp::make_tuple( vals[Is]... );
}

} // namespace internal

/**
 * @brief Make a homogeneous tuple from an array of values.
 * @tparam N the number of values
 * @tparam T type of values
 * @param vals pointer ot values
 * @return the constructed tuple
 */
template< camp::idx_t N, typename T >
LVARRAY_HOST_DEVICE constexpr
auto
makeTupleFromPtr( T const * const vals )
{
  return internal::makeTupleFromPtr( vals, camp::make_idx_seq_t< N >{} );
}

namespace tupleManipulation
{

template< typename T >
struct common_type;

template< typename T, typename ... Ts >
struct common_type< camp::tuple< T, Ts... > >
{
  using type = std::common_type_t< value_type_t< T >, value_type_t< Ts >... >;
};

template<>
struct common_type< camp::tuple<> >
{
  using type = int;
};

template< typename T >
using common_type_t = typename common_type< camp::decay< T > >::type;

namespace internal
{

template< typename T, typename U, camp::idx_t ... Is, camp::idx_t ... Js, camp::idx_t ... Ks >
LVARRAY_HOST_DEVICE constexpr
auto
construct( T const & t, U const & v, camp::idx_seq< Is... >, camp::idx_seq< Js... >, camp::idx_seq< Ks... > )
{
  return makeTuple( camp::get< Is >( t )..., (void(Js), v)..., camp::get< Ks >( t )... );
}

} // namespace internal

template< int N, typename U >
LVARRAY_HOST_DEVICE constexpr
auto
repeat( U const & v )
{
  return internal::construct( camp::tuple<>{}, v, camp::idx_seq<>{}, camp::make_idx_seq_t< N >{}, camp::idx_seq<>{} );
}

template< typename T, camp::idx_t ... Is >
LVARRAY_HOST_DEVICE constexpr
auto
select( T const & t, camp::idx_seq< Is... > )
{
  return makeTuple( camp::get< Is >( t ) ... );
}

template< typename T, camp::idx_t ... Is >
LVARRAY_HOST_DEVICE constexpr
auto
selectRefs( T const & t, camp::idx_seq< Is... > )
{
  return camp::tuple< camp::tuple_element_t< Is, T > const & ... >( camp::get< Is >( t ) ... );
}

template< typename T, camp::idx_t ... Is >
LVARRAY_HOST_DEVICE constexpr
auto
selectRefs( T & t, camp::idx_seq< Is... > )
{
  return camp::tuple< camp::tuple_element_t< Is, T > & ... >( camp::get< Is >( t ) ... );
}

template< int N = 1, typename T, typename U >
LVARRAY_HOST_DEVICE constexpr
auto
prepend( T const & t, U const & v )
{
  return internal::construct( t, v, camp::idx_seq<>{}, camp::make_idx_seq_t< N >{}, camp::idx_seq_from_t< T >{} );
}

template< int N = 1, typename T, typename U >
LVARRAY_HOST_DEVICE constexpr
auto
append( T const & t, U const & v )
{
  return internal::construct( t, v, camp::idx_seq_from_t< T >{}, camp::make_idx_seq_t< N >{}, camp::idx_seq<>{} );
}

template< int I, int N = 1, typename T, typename U >
LVARRAY_HOST_DEVICE constexpr
auto
replace( T const & t, U const & v )
{
  return internal::construct( t, v, camp::make_idx_seq_t< I >{}, camp::make_idx_seq_t< N >{}, camp::make_idx_range< I + N, camp::tsize_v< T > >{} );
}

template< int I, int N = 1, typename T, typename U >
LVARRAY_HOST_DEVICE constexpr
auto
insert( T const & t, U const & v )
{
  return internal::construct( t, v, camp::make_idx_seq_t< I >{}, camp::make_idx_seq_t< N >{}, camp::make_idx_range< I, camp::tsize_v< T > >{} );
}

template< int I, int N = 1, typename T >
LVARRAY_HOST_DEVICE constexpr
auto
remove( T const & t )
{
  return internal::construct( t, 0, camp::make_idx_seq_t< I >{}, camp::idx_seq<>{}, camp::make_idx_range< I + N, camp::tsize_v< T > >{} );
}

template< int I, int J, typename T >
LVARRAY_HOST_DEVICE constexpr
auto
take( T const & t )
{
  return select( t, camp::make_idx_range< I, J >{} );
}

template< int N, typename T >
LVARRAY_HOST_DEVICE constexpr
auto
drop_front( T const & t )
{
  return select( t, camp::make_idx_range< N, camp::tsize_v< T > >{} );
}

template< int N, typename T >
LVARRAY_HOST_DEVICE constexpr
auto
drop_back( T const & t )
{
  return select( t, camp::make_idx_range< 0, camp::tsize_v< T > - N >{} );
}

namespace internal
{

// Implementation of general zip() and unzip() are left as an exercise to the reader.

template< typename T, camp::idx_t ... Is >
LVARRAY_HOST_DEVICE constexpr
auto
zip2( T && t, camp::idx_seq< Is... > )
{
  return makeTuple( makeTuple( camp::get< Is >( camp::get< 0 >( LVARRAY_FWD( t ) ) ), camp::get< Is >( camp::get< 1 >( LVARRAY_FWD( t ) ) ) )... );
}

template< typename T, camp::idx_t ... Is >
LVARRAY_HOST_DEVICE constexpr
auto
unzip2( T && t, camp::idx_seq< Is... > )
{
  return makeTuple( makeTuple( camp::get< 0 >( camp::get< Is >( LVARRAY_FWD( t ) ) )... ),
                    makeTuple( camp::get< 1 >( camp::get< Is >( LVARRAY_FWD( t ) ) )... ) );
}

} // namespace internal

template< typename T >
LVARRAY_HOST_DEVICE constexpr
auto
zip2( T && t )
{
  return internal::zip2( LVARRAY_FWD( t ), camp::idx_seq_from_t< camp::tuple_element_t< 0, T > >{} );
}

template< typename T1, typename T2 >
LVARRAY_HOST_DEVICE constexpr
auto
zip2( T1 && t1, T2 && t2 )
{
  static_assert( camp::tsize_v< T1 > == camp::tsize_v< T2 >, "Tuples must have equal sizes." );
  return zip2( makeTuple( LVARRAY_FWD( t1 ), LVARRAY_FWD( t2 ) ) );
}

template< typename T >
LVARRAY_HOST_DEVICE constexpr
auto
unzip2( T && t )
{
  return internal::unzip2( LVARRAY_FWD( t ), camp::idx_seq_from_t< T >{} );
}

namespace internal
{

template< typename T, typename F, typename G, camp::idx_t ... Is >
LVARRAY_HOST_DEVICE constexpr
auto
transformApply( T && t, F && f, G && g, camp::idx_seq< Is... > )
{
  return LVARRAY_FWD( g )( f( camp::get< Is >( LVARRAY_FWD( t ) ) )... );
}

template< typename T1, typename T2, typename F,  typename G, camp::idx_t ... Is >
LVARRAY_HOST_DEVICE constexpr
auto
transformApply2( T1 && t1, T2 && t2, F && f, G && g, camp::idx_seq< Is... > )
{
  return LVARRAY_FWD( g )( f( camp::get< Is >( LVARRAY_FWD( t1 ) ), camp::get< Is >( LVARRAY_FWD( t2 ) ) )... );
}

} // namespace internal

template< typename T, typename F, typename G, typename SEQ = camp::idx_seq_from_t< T > >
LVARRAY_HOST_DEVICE constexpr
auto
transformApply( T && t, F && f, G && g, SEQ s = {} )
{
  return internal::transformApply( LVARRAY_FWD( t ), LVARRAY_FWD( f ), LVARRAY_FWD( g ), s );
}

template< typename T1, typename T2, typename F, typename G, typename SEQ = camp::idx_seq_from_t< T1 > >
LVARRAY_HOST_DEVICE constexpr
auto
transformApply2( T1 && t1, T2 && t2, F && f, G && g, SEQ s = {} )
{
  static_assert( camp::tsize_v< T1 > == camp::tsize_v< T2 >, "Tuples must have equal number of elements" );
  return internal::transformApply2( LVARRAY_FWD( t1 ), LVARRAY_FWD( t2 ), LVARRAY_FWD( f ), LVARRAY_FWD( g ), s );
}

template< typename T, typename F, typename SEQ = camp::idx_seq_from_t< T > >
LVARRAY_HOST_DEVICE constexpr
auto
transform( T && t, F && f, SEQ s = {} )
{
  return transformApply( LVARRAY_FWD( t ), LVARRAY_FWD( f ), []( auto && ... vs ) { return makeTuple( LVARRAY_FWD( vs )... ); }, s );
}

template< typename T1, typename T2, typename F, typename SEQ = camp::idx_seq_from_t< T1 > >
LVARRAY_HOST_DEVICE constexpr
auto
transform2( T1 && t, T2 && t2, F && f, SEQ s = {}  )
{
  return transformApply2( LVARRAY_FWD( t ), LVARRAY_FWD( t2 ), LVARRAY_FWD( f ), []( auto && ... vs ) { return makeTuple( LVARRAY_FWD( vs )... ); }, s );
}

template< typename T, typename G, typename SEQ = camp::idx_seq_from_t< T > >
LVARRAY_HOST_DEVICE constexpr
auto
apply( T && t, G && g, SEQ s = {} )
{
  return transformApply( LVARRAY_FWD( t ), []( auto && v ) -> decltype(auto) { return LVARRAY_FWD( v ); }, LVARRAY_FWD( g ), s );
}

template< typename T, typename F, typename SEQ = camp::idx_seq_from_t< T > >
LVARRAY_HOST_DEVICE constexpr
void
forEach( T && t, F && f, SEQ s = {} )
{
  tupleManipulation::apply( LVARRAY_FWD( t ), [&f]( auto && ... vs ){ ( f( LVARRAY_FWD( vs ) ), ... ); }, s );
}

template< typename T1, typename T2, typename F, typename SEQ = camp::idx_seq_from_t< T1 > >
LVARRAY_HOST_DEVICE constexpr
void
forEach2( T1 && t1, T2 && t2, F && f, SEQ s = {} )
{
  tupleManipulation::transform2( LVARRAY_FWD( t1 ), LVARRAY_FWD( t2 ), [&f]( auto && v1, auto && v2 ) { f( LVARRAY_FWD( v1 ), LVARRAY_FWD( v2 ) ); return _t{}; }, s );
}

namespace internal
{

template< typename T, typename U, typename F >
LVARRAY_HOST_DEVICE constexpr
decltype(auto)
fold( T &&, U && init, F &&, camp::idx_seq<> )
{
  return LVARRAY_FWD( init );
}

template< typename T, typename U, typename F, camp::idx_t I, camp::idx_t ... Is >
LVARRAY_HOST_DEVICE constexpr
auto
fold( T && t, U && init, F && f, camp::idx_seq< I, Is... > )
{
  return fold( t, f( LVARRAY_FWD( init ), camp::get< I >( LVARRAY_FWD( t ) ) ), f, camp::idx_seq< Is... >{} );
}

} // namespace internal

template< typename T, typename U, typename F >
LVARRAY_HOST_DEVICE constexpr
auto
foldLeft( T && t, U && init, F && f )
{
  return internal::fold( LVARRAY_FWD( t ), LVARRAY_FWD( init ), LVARRAY_FWD( f ), camp::idx_seq_from_t< T >{} );
}

template< typename T, typename U, typename F >
LVARRAY_HOST_DEVICE constexpr
auto
foldRight( T && t, U && init, F && f )
{
  return internal::fold( LVARRAY_FWD( t ), LVARRAY_FWD( init ), [&f]( auto && x, auto && y ){ return f( LVARRAY_FWD( y ), LVARRAY_FWD( x ) ); }, camp::idx_rseq_from_t< T >{} );
}

template< typename T, typename F >
LVARRAY_HOST_DEVICE constexpr
auto
allOf( T && t, F && f )
{
  return transformApply( LVARRAY_FWD( t ), LVARRAY_FWD( f ), []( auto && ... vs ){ return ( _t{} && ... && vs ); } );
}

template< typename T1, typename T2, typename F, typename SEQ = camp::idx_seq_from_t< T1 > >
LVARRAY_HOST_DEVICE constexpr
auto
allOf2( T1 && t1, T2 && t2, F && f, SEQ s = {} )
{
  return transformApply2( LVARRAY_FWD( t1 ), LVARRAY_FWD( t2 ), LVARRAY_FWD( f ), []( auto && ... vs ){ return ( _t{} && ... && vs ); }, s );
}

template< typename T, typename F >
LVARRAY_HOST_DEVICE constexpr
auto
anyOf( T && t, F && f )
{
  return transformApply( LVARRAY_FWD( t ), LVARRAY_FWD( f ), []( auto && ... vs ){ return ( _f{} || ... || vs ); } );
}

template< typename T1, typename T2, typename F, typename SEQ = camp::idx_seq_from_t< T1 > >
LVARRAY_HOST_DEVICE constexpr
auto
anyOf2( T1 && t1, T2 && t2, F && f, SEQ s = {} )
{
  return transformApply2( LVARRAY_FWD( t1 ), LVARRAY_FWD( t2 ), LVARRAY_FWD( f ), []( auto && ... vs ){ return ( _f{} || ... || vs ); }, s );
}

namespace internal
{

template< typename T, camp::idx_t... TI, typename U, camp::idx_t... UI >
LVARRAY_HOST_DEVICE constexpr
auto
concat( T && t, camp::idx_seq< TI... >, U && u, camp::idx_seq< UI... > )
{
  return makeTuple( camp::get< TI >( LVARRAY_FWD( t ) )..., camp::get< UI >( LVARRAY_FWD( u ) )... );
}

}

template< typename T, typename ... Ts >
LVARRAY_HOST_DEVICE constexpr
auto
concat( T && t, Ts && ... ts )
{
  if constexpr( sizeof...( Ts ) == 0 )
  {
    return LVARRAY_FWD( t );
  }
  else
  {
    // This doesn't compile under nvcc 12.1:
    // return camp::tuple_cat_pair( LVARRAY_FWD( t ), concat( LVARRAY_FWD( ts )... ) );
    auto u = concat( LVARRAY_FWD( ts )... );
    return internal::concat( LVARRAY_FWD( t ), camp::idx_seq_from_t< T >{}, LVARRAY_FWD( u ), camp::idx_seq_from_t< decltype( u ) >{} );
  }
}

template< typename T, typename P >
LVARRAY_HOST_DEVICE constexpr
auto
filter( T && t, P && p )
{
  return transformApply( LVARRAY_FWD( t ), LVARRAY_FWD( p ), []( auto && ... vs ){ return concat( LVARRAY_FWD( vs )... ); } );
}

template< typename T1, typename T2, typename P >
LVARRAY_HOST_DEVICE constexpr
auto
filter( T1 && t1, T2 && t2, P && p )
{
  return transformApply2( LVARRAY_FWD( t1 ), LVARRAY_FWD( t2 ), LVARRAY_FWD( p ), []( auto && ... vs ){ return concat( LVARRAY_FWD( vs )... ); } );
}

namespace internal
{

template< camp::idx_t I, typename T, typename P >
LVARRAY_HOST_DEVICE constexpr
camp::idx_t
find( T const & t, P && p )
{
  if constexpr( I == camp::tsize_v< T > )
  {
    return I;
  }
  else
  {
    if constexpr( decltype( p( camp::get< I >( t ) ) )::value )
    {
      return I;
    }
    else
    {
      return find< I + 1 >( t, p );
    }
  }
}

}

template< typename T, typename P >
LVARRAY_HOST_DEVICE constexpr
camp::idx_t
find( T && t, P && p )
{
  if constexpr( camp::tsize_v< T > == 0 )
  {
    return 1;
  }
  else
  {
    return internal::find< 0 >( LVARRAY_FWD( t ), LVARRAY_FWD( p ) );
  }
}

template< typename T >
LVARRAY_HOST_DEVICE constexpr
auto
getValue( T const & t, int const idx )
{
  // TODO: reconsider Array/Slice interface and remove this garbage function if no longer needed
  static_assert( camp::tsize_v< T > > 0, "Tuple must not be empty." );
  common_type_t< T > ret = camp::get< 0 >( t );
  tupleManipulation::forEach( t, [idx, curr=0, &ret]( auto const & e ) mutable { if( idx == curr++ ) ret = e; } );
  return ret;
}

namespace internal
{

template< typename T, camp::idx_t ... Is >
LVARRAY_HOST_DEVICE constexpr
typeManipulation::CArray< common_type_t< T >, camp::tsize_v< T > >
toArray( T && t, camp::idx_seq< Is... > )
{
  return { camp::get< Is >( LVARRAY_FWD( t ) )... };
}

template< typename T, int N, camp::idx_t ... Is >
LVARRAY_HOST_DEVICE constexpr
auto
fromArray( typeManipulation::CArray< T, N > const & arr, camp::idx_seq< Is... > )
{
  return makeTuple( arr[Is]... );
}

} // namespace internal

template< typename T >
LVARRAY_HOST_DEVICE constexpr
auto
toArray( T && t )
{
  return internal::toArray( LVARRAY_FWD( t ), camp::idx_seq_from_t< T >{} );
}

template< typename T, int N >
LVARRAY_HOST_DEVICE constexpr
auto
fromArray( typeManipulation::CArray< T, N > const & arr )
{
  return internal::fromArray( arr, camp::make_idx_seq_t< N >{} );
}

#define LVARRAY_MAKE_CMP_STRUCT( NAME, OP ) \
struct NAME \
{ \
  template< typename T, typename U > \
  LVARRAY_HOST_DEVICE constexpr \
  auto operator()( T const & lhs, U const & rhs ) \
  { \
    return lhs OP rhs; \
  } \
}

LVARRAY_MAKE_CMP_STRUCT( cmp_lt, <  );
LVARRAY_MAKE_CMP_STRUCT( cmp_le, <= );
LVARRAY_MAKE_CMP_STRUCT( cmp_gt, >  );
LVARRAY_MAKE_CMP_STRUCT( cmp_ge, >= );
LVARRAY_MAKE_CMP_STRUCT( cmp_eq, == );
LVARRAY_MAKE_CMP_STRUCT( cmp_ne, != );

#undef LVARRAY_MAKE_CMP_STRUCT

#define LVARRAY_MAKE_BOP_STRUCT( NAME, OP ) \
struct NAME \
{ \
  template< typename T, typename U > \
  LVARRAY_HOST_DEVICE constexpr \
  auto operator()( T const & lhs, U const & rhs ) \
  { \
    return lhs OP rhs; \
  } \
}

LVARRAY_MAKE_BOP_STRUCT( op_plus,  + );
LVARRAY_MAKE_BOP_STRUCT( op_minus, - );
LVARRAY_MAKE_BOP_STRUCT( op_mult,  * );
LVARRAY_MAKE_BOP_STRUCT( op_div,   / );
LVARRAY_MAKE_BOP_STRUCT( op_rem,   % );

#undef LVARRAY_MAKE_BOP_STRUCT

} // namespace tupleManipulation

} // namespace LvArray
