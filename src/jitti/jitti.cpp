// Source includes
#include "jitti.hpp"

// System includes
#include <dlfcn.h>

namespace jitti
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::ostream & operator<<( std::ostream & os, SymbolTable const & table )
{
  for ( auto const & kvPair : table )
  {
    os << kvPair.first << ": " << "{ \"" << kvPair.second.first << "\", " <<
          LvArray::demangle( kvPair.second.second.name() ) << " }\n";
  }

  return os;
}

int monthStringToInt( std::string const & month )
{
  if ( month == "Jan" )
  { return 1; }
  if ( month == "Feb" )
  { return 2; }
  if ( month == "Mar" )
  { return 3; }
  if ( month == "Apr" )
  { return 4; }
  if ( month == "May" )
  { return 5; }
  if ( month == "Jun" )
  { return 6; }
  if ( month == "Jul" )
  { return 7; }
  if ( month == "Aug" )
  { return 8; }
  if ( month == "Sep" )
  { return 9; }
  if ( month == "Oct" )
  { return 10; }
  if ( month == "Nov" )
  { return 11; }
  if ( month == "Dec" )
  { return 12; }

  LVARRAY_ERROR( "Uncrecognized month: " << month );
  return -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::array< int, 6 > getCompileTime()
{
  std::string const monthString( __DATE__, 3 );

  int const month = monthStringToInt( monthString );
  int const day = std::atoi( static_cast< char const * >( __DATE__ ) + 4 );
  int const year = std::atoi( static_cast< char const * >( __DATE__ ) + 7 );

  int const hour = std::atoi( __TIME__ );
  int const minute = std::atoi( static_cast< char const * >( __TIME__ ) + 3 );
  int const second = std::atoi( static_cast< char const * >( __TIME__ ) + 6 );

  LVARRAY_LOG_VAR( year );
  LVARRAY_LOG_VAR( month );
  LVARRAY_LOG_VAR( day );
  LVARRAY_LOG_VAR( hour );
  LVARRAY_LOG_VAR( minute );
  LVARRAY_LOG_VAR( second );

  return { year, month, day, hour, minute, second };
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
DynamicLibrary::DynamicLibrary( std::string const & path ):
  m_handle( dlopen( path.data(), RTLD_LAZY ) )
{}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
DynamicLibrary::DynamicLibrary( DynamicLibrary && src ) :
  m_handle( src.m_handle )
{
  src.m_handle = nullptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
DynamicLibrary::~DynamicLibrary()
{
  if ( m_handle != nullptr )
  { LVARRAY_ERROR_IF( dlclose( m_handle ), dlerror() ); }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void * DynamicLibrary::getSymbol( char const * const name ) const
{
  LVARRAY_ERROR_IF( m_handle == nullptr, dlerror() );

  void * const symbolAddress = dlsym( m_handle, name );

  char const * const error = dlerror();

  LVARRAY_ERROR_IF( error != nullptr, "Could not find the symbol: " << name << "\n" << error );

  return symbolAddress;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TypedDynamicLibrary::TypedDynamicLibrary( DynamicLibrary && dl ):
  DynamicLibrary( std::move( dl ) ),
  m_symbols( getSymbolTable() )
{}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
SymbolTable TypedDynamicLibrary::getSymbolTable()
{
  SymbolTable const * (*symbols)() =
    reinterpret_cast< SymbolTable const * (*)() >( DynamicLibrary::getSymbol( "getExportedSymbols") );

  char const * const error = dlerror();

  LVARRAY_ERROR_IF( error != nullptr, "Could not find the symbols table!\n" << error );

  return *symbols();
}

} // namespace jitti
