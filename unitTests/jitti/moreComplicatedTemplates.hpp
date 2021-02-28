#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <sstream>


constexpr auto moreComplicatedTemplatesPath = __FILE__;

template< int N >
std::string addToString( std::string const & m )
{ return m + std::to_string( N ); }

template< typename KEY, typename VALUE >
VALUE & staticMapAccess( KEY const & key, VALUE const & value )
{
  static std::unordered_map< KEY, VALUE > s_map;
  auto const iter = s_map.find( key );
  if ( iter == s_map.end() )
  {
    return s_map.insert( std::make_pair( key, value ) ).first->second;
  }

  return iter->second;
}

struct Base
{
  virtual ~Base() = default;
  virtual std::string getValueString() const = 0;
};

template< typename T >
struct Derived : public Base
{
  Derived( T const & value ):
    m_value( value )
  {}

  virtual std::string getValueString() const override
  {
    std::ostringstream oss;
    oss << m_value;
    return oss.str();
  }

  T const m_value;
};

template< typename T >
std::unique_ptr< Base > factory( T const & value )
{
  return std::make_unique< Derived< T > >( value );
}
