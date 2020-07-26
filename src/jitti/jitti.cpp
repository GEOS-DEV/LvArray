// Source includes
#include "jitti.hpp"

// System includes
#include <dlfcn.h>

namespace jitti
{

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
