#include "Function.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static std::ostream & operator<<( std::ostream & os, jitti::SymbolTable const & table )
{
  for ( auto const & kvPair : table )
  {
    os << kvPair.first << ": " << "{ \"" << kvPair.second.first << "\", " <<
          LvArray::system::demangle( kvPair.second.second.name() ) << " }\n";
  }

  return os;
}

namespace jitti
{
namespace internal
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TypedDynamicLibrary::TypedDynamicLibrary( std::string const & path ):
  m_handle( dlopen( path.c_str(), RTLD_LAZY ) )
{
  LVARRAY_ERROR_IF( m_handle == nullptr, "dlopen( " << path << " ) failed." );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TypedDynamicLibrary::TypedDynamicLibrary( TypedDynamicLibrary && src ):
  m_handle( src.m_handle )
{
  src.m_handle = nullptr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TypedDynamicLibrary::~TypedDynamicLibrary()
{
  if ( m_handle != nullptr )
  { LVARRAY_ERROR_IF( dlclose( m_handle ), dlerror() ); }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
SymbolTable const & TypedDynamicLibrary::getSymbolTable() const
{
  LVARRAY_ERROR_IF( m_handle == nullptr, "Error" );

  void * const symbolAddress = dlsym( m_handle, "getExportedSymbols" );

  char const * const error = dlerror();

  LVARRAY_ERROR_IF( error != nullptr,
                    "Expected a function named 'getExportedSymbols' but none exists\n" <<
                    "The error from dlerror() is: " << error );

  LVARRAY_ERROR_IF( symbolAddress == nullptr,
                    "Expected a function named 'getExportedSymbols' but none exists." );

  SymbolTable const * (*symbols)() =
    reinterpret_cast< SymbolTable const * (*)() >( symbolAddress );

  return *symbols();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void * TypedDynamicLibrary::getSymbolCheckType( std::string const & name, std::type_index const expectedType ) const
{
  SymbolTable const & symbols = getSymbolTable();
  auto const iter = symbols.find( name );
  LVARRAY_ERROR_IF( iter == symbols.end(),
                    "Symbol \"" << name << "\" not found in the exported symbol table.\n" <<
                    "table contains:\n" << symbols );

  std::pair< void *, std::type_index > const & value = iter->second;

  LVARRAY_ERROR_IF( value.second != expectedType,
                    "Symbol \"" << name << "\" found but it has type " <<
                    LvArray::system::demangle( value.second.name() ) << " not " <<
                    LvArray::system::demangle( expectedType.name() ) );

  return  value.first;
}

} // namespace internal
} // namespace jitti
