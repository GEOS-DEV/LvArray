/*
 * IntegerConversion.hpp
 *
 *  Created on: Nov 20, 2018
 *      Author: settgast
 */

#ifndef INTEGERCONVERSION_HPP_
#define INTEGERCONVERSION_HPP_

#include "StringUtilities.hpp"
#include "Logger.hpp"

/**
 * @file IntegerConversion.hpp
 */

/**
 * @brief function to take an unsigned integer and convert to a signed integer
 * @tparam RTYPE the type of signed integer to convert into
 * @tparam T     the type of the unsigned integer provided
 * @param input  the integer to convert
 * @return the converted integer
 *
 * This function takes in an unsigned integer and converts it to a signed integer. There is a check
 * to make sure that no data is lost as a result of this conversion.
 */
template< typename RTYPE, typename T >
inline typename std::enable_if< std::is_unsigned<T>::value && std::is_signed<RTYPE>::value,
                                RTYPE >::type
integer_conversion( T input )
{
  static_assert( std::numeric_limits<T>::is_integer, "input is not an integer type" );
  static_assert( std::numeric_limits<RTYPE>::is_integer, "requested conversion is not an integer type" );

  GEOS_ERROR_IF( input > std::numeric_limits<RTYPE>::max(),
                       "conversion of \"("
                       <<cxx_utilities::demangle( typeid(T).name() )
                       <<")"<<input<<"\" to ("
                       <<cxx_utilities::demangle( typeid(RTYPE).name() )
                       <<") loses information! ("<<input<<">"
                       <<std::numeric_limits<RTYPE>::max()<<")" );

  return static_cast<RTYPE>(input);
}

/**
 * @brief function to take a signed integer and convert to an unsigned integer
 * @tparam RTYPE the type of unsigned integer to convert into
 * @tparam T     the type of signed the integer provided
 * @param input  the integer value to convert
 * @return the converted integer
 *
 * This function takes in a signed integer and converts it to an unsigned integer. There is a check
 * to make sure that no data is lost as a result of this conversion.
 */
template< typename RTYPE, typename T >
inline typename std::enable_if< std::is_signed<T>::value && std::is_unsigned<RTYPE>::value,
                                RTYPE >::type
integer_conversion( T input )
{
  static_assert( std::numeric_limits<T>::is_integer, "input is not an integer type" );
  static_assert( std::numeric_limits<RTYPE>::is_integer, "requested conversion is not an integer type" );

  GEOS_ERROR_IF( input < 0,
                       "conversion of integer \"("
                       <<cxx_utilities::demangle( typeid(T).name() )
                       <<")"<<input<<"\" to type ("
                       <<cxx_utilities::demangle( typeid(RTYPE).name() )
                       <<") loses information! ("<<input<<"<0)" );

  GEOS_ERROR_IF( static_cast<typename std::make_unsigned<T>::type>(input) > std::numeric_limits<RTYPE>::max(),
                       "conversion of integer \"("
                       <<cxx_utilities::demangle( typeid(T).name() )
                       <<")"<<input<<"\" to type ("
                       <<cxx_utilities::demangle( typeid(RTYPE).name() )
                       <<") loses information! ("<<input<<">"
                       <<std::numeric_limits<RTYPE>::max()<<")" );

  return static_cast<RTYPE>(input);
}


/**
 * @brief function to take an integer and convert to an integer with the same signedness
 * @tparam RTYPE the type of integer to convert into
 * @tparam T     the type of the integer provided
 * @param input  the integer value to convert
 * @return the converted integer
 *
 * This function takes in an integer and converts it to an integer of different type. There is a check
 * to make sure that no data is lost as a result of this conversion.
 */
template< typename RTYPE, typename T >
inline typename std::enable_if< ( std::is_signed<T>::value && std::is_signed<RTYPE>::value ) ||
                                ( std::is_unsigned<T>::value && std::is_unsigned<RTYPE>::value ),
                                RTYPE >::type
integer_conversion( T input )
{
  static_assert( std::numeric_limits<T>::is_integer, "input is not an integer type" );
  static_assert( std::numeric_limits<RTYPE>::is_integer, "requested conversion is not an integer type" );

  if( input > std::numeric_limits<RTYPE>::max() ||
      input < std::numeric_limits<RTYPE>::lowest() )
  {
    abort();
  }

  GEOS_ERROR_IF( input > std::numeric_limits<RTYPE>::max(),
                       "conversion of integer \"("
                       <<cxx_utilities::demangle( typeid(T).name() )
                       <<")"<<input<<"\" to type "
                       <<cxx_utilities::demangle( typeid(RTYPE).name() )
                       <<" loses information! ("<<input<<">"
                       <<std::numeric_limits<RTYPE>::max()<<")" );

  GEOS_ERROR_IF( input < std::numeric_limits<RTYPE>::lowest(),
                       "conversion of integer \"("
                       <<cxx_utilities::demangle( typeid(T).name() )
                       <<")"<<input<<"\" to type ("
                       <<cxx_utilities::demangle( typeid(RTYPE).name() )
                       <<") loses information! ("<<input<<"<"
                       <<std::numeric_limits<RTYPE>::lowest()<<")" );


  return static_cast<RTYPE>(input);
}



#endif /* SRC_SRC_INTEGERCONVERSION_HPP_ */
