/*
 * tv_helpers.hpp
 *
 *  Created on: Jun 8, 2019
 *      Author: settgast
 */

#ifndef CORECOMPONENTS_CXX_UTILITIES_SRC_SRC_TOTALVIEW_TV_HELPERS_HPP_
#define CORECOMPONENTS_CXX_UTILITIES_SRC_SRC_TOTALVIEW_TV_HELPERS_HPP_

#include "StringUtilities.hpp"
#include<typeinfo>

namespace totalview
{

template< typename T >
std::string typeName( )
{
  return cxx_utilities::demangle( typeid(T).name() );
}

template< typename TYPE, typename INDEX_TYPE>
std::string format( int NDIM, INDEX_TYPE const * const dims )
{
  std::string rval = typeName<TYPE>();
  for( int i=0 ; i<NDIM; ++i )
  {
    rval += "["+std::to_string(dims[i])+"]";
  }
  return rval;
}

template< int NDIM, typename INDEX_TYPE  >
void tvformat( char * type, char const * name, INDEX_TYPE const * dims );

template<> inline void tvformat<1,int>( char * type, char const * name, int const * dims )                      { sprintf( type, "%s[%d]", name, dims[0] ); }
template<> inline void tvformat<1,long int>( char * type, char const * name, long int const * dims )            { sprintf( type, "%s[%ld]", name, dims[0] ); }
template<> inline void tvformat<1,long long int>( char * type, char const * name, long long int const * dims )  { sprintf( type, "%s[%lld]", name, dims[0] ); }

template<> inline void tvformat<2,int>( char * type, char const * name, int const * dims )                      { sprintf( type, "%s[%d][%d]", name, dims[0], dims[1] ); }
template<> inline void tvformat<2,long int>( char * type, char const * name, long int const * dims )            { sprintf( type, "%s[%ld][%ld]", name, dims[0], dims[1] ); }
template<> inline void tvformat<2,long long int>( char * type, char const * name, long long int const * dims )  { sprintf( type, "%s[%lld][%lld]", name, dims[0], dims[1] ); }

template<> inline void tvformat<3,int>( char * type, char const * name, int const * dims )                      { sprintf( type, "%s[%d][%d][%d]", name, dims[0], dims[1], dims[2] ); }
template<> inline void tvformat<3,long int>( char * type, char const * name, long int const * dims )            { sprintf( type, "%s[%ld][%ld][%ld]", name, dims[0], dims[1], dims[2] ); }
template<> inline void tvformat<3,long long int>( char * type, char const * name, long long int const * dims )  { sprintf( type, "%s[%lld][%lld][%lld]", name, dims[0], dims[1], dims[2] ); }

template<> inline void tvformat<4,int>( char * type, char const * name, int const * dims )                      { sprintf( type, "%s[%d][%d][%d][%d]", name, dims[0], dims[1], dims[2], dims[3] ); }
template<> inline void tvformat<4,long int>( char * type, char const * name, long int const * dims )            { sprintf( type, "%s[%ld][%ld][%ld][%ld]", name, dims[0], dims[1], dims[2], dims[3] ); }
template<> inline void tvformat<4,long long int>( char * type, char const * name, long long int const * dims )  { sprintf( type, "%s[%lld][%lld][%lld][%lld]", name, dims[0], dims[1], dims[2], dims[3] ); }

template<> inline void tvformat<5,int>( char * type, char const * name, int const * dims )                      { sprintf( type, "%s[%d][%d][%d][%d][%d]", name, dims[0], dims[1], dims[2], dims[3], dims[4] ); }
template<> inline void tvformat<5,long int>( char * type, char const * name, long int const * dims )            { sprintf( type, "%s[%ld][%ld][%ld][%ld][%ld]", name, dims[0], dims[1], dims[2], dims[3], dims[4] ); }
template<> inline void tvformat<5,long long int>( char * type, char const * name, long long int const * dims )  { sprintf( type, "%s[%lld][%lld][%lld][%lld][%lld]", name, dims[0], dims[1], dims[2], dims[3], dims[4] ); }

}


#endif /* CORECOMPONENTS_CXX_UTILITIES_SRC_SRC_TOTALVIEW_TV_HELPERS_HPP_ */
