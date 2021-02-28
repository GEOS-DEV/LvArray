#pragma once

constexpr auto simpleTemplatesPath = __FILE__;

template< int N >
int add( int const m )
{ return m + N; }

template< typename T >
void squareAll( T * const output, T const * const input, int const numValues )
{
  for( int i = 0; i < numValues; ++i )
  {
    output[ i ] = input[ i ] * input[ i ];
  }
}
