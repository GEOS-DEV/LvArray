// Source includes
#include "jitti.hpp"

// System includes
#include <dlfcn.h>
#include <regex>

namespace jitti
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::ostream & operator<<( std::ostream & os, SymbolTable const & table )
{
  for ( auto const & kvPair : table )
  {
    os << kvPair.first << ": " << "{ \"" << kvPair.second.first << "\", " <<
          LvArray::system::demangle( kvPair.second.second.name() ) << " }\n";
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
DynamicLibrary::DynamicLibrary( char const * const path ):
  m_handle( dlopen( path, RTLD_LAZY ) )
{
  LVARRAY_ERROR_IF( m_handle == nullptr, "dlopen( " << path << " ) failed." );
}

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
  LVARRAY_ERROR_IF( symbolAddress == nullptr,
                    "Could not find the symbol: " << name << " but dlerror didn't report anything." );

  return symbolAddress;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
SymbolTable getSymbolTable( DynamicLibrary const & dl )
{
  SymbolTable const * (*symbols)() =
    reinterpret_cast< SymbolTable const * (*)() >( dl.getSymbol( "getExportedSymbols" ) );

  return *symbols();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TypedDynamicLibrary::TypedDynamicLibrary( char const * const path ):
  DynamicLibrary( path ),
  m_symbols( getSymbolTable( *this ) )
{}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TypedDynamicLibrary::TypedDynamicLibrary( DynamicLibrary && src ):
  DynamicLibrary( std::move( src ) ),
  m_symbols( getSymbolTable( *this ) )
{}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TypedDynamicLibrary::TypedDynamicLibrary( TypedDynamicLibrary && src ):
  DynamicLibrary( std::move( src ) ),
  m_symbols( std::move( src.m_symbols ) )
{
  src.m_symbols.clear();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void * TypedDynamicLibrary::getSymbolCheckType( char const * const name, std::type_index const expectedType ) const
{
  auto const iter = m_symbols.find( name );
  LVARRAY_ERROR_IF( iter == m_symbols.end(),
                    "Symbol \"" << name << "\" not found in the exported symbol table.\n" <<
                    "table contains:\n" << m_symbols );

  std::pair< void *, std::type_index > const & value = iter->second;

  LVARRAY_ERROR_IF( value.second != expectedType,
                    "Symbol \"" << name << "\" found but it has type " <<
                    LvArray::system::demangle( value.second.name() ) << " not " <<
                    LvArray::system::demangle( expectedType.name() ) );

  return  value.first;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Compiler::Compiler( std::string const & compileCommand,
                    bool const cuda,
                    std::string const & linker,
                    std::string const & linkArgs ):
  m_compileCommand( compileCommand ),
  m_cuda( cuda ),
  m_linker( linker ),
  m_linkArgs( linkArgs )
{}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
DynamicLibrary Compiler::createDynamicLibrary( std::string const & filePath,
                                               std::vector< std::string > const & defines,
                                               std::string const & outputObject,
                                               std::string const & outputLibrary ) const
{
  std::string compileCommand = m_compileCommand;
  if( m_cuda )
  {
    compileCommand += " -Xcompiler";
  }

  compileCommand += " -fPIC -c " + filePath + " -o " + outputObject;

  std::string joinedDefines;
  for ( std::string const & define : defines )
  { joinedDefines += " -D " + define; }

  // You need to escape commas for NVCC.
  if( m_cuda )
  {
    compileCommand += std::regex_replace( joinedDefines, std::regex( "," ), "\\," );
  }
  else
  {
    compileCommand += joinedDefines;
  }

  LVARRAY_LOG_VAR( compileCommand );
  LVARRAY_ERROR_IF( std::system( compileCommand.data() ) != 0, compileCommand );

  std::string linkCommand = m_linker + " -shared " + outputObject + " " + m_linkArgs + " -o " + outputLibrary;

  LVARRAY_LOG_VAR( linkCommand );
  LVARRAY_ERROR_IF( std::system( linkCommand.data() ) != 0, linkCommand );

  return DynamicLibrary( outputLibrary );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TypedDynamicLibrary TemplateCompiler::instantiateTemplate( std::string const & templateFunction,
                                                           std::string const & templateParams,
                                                           std::string const & headerFile,
                                                           std::string const & outputObject,
                                                           std::string const & outputLibrary ) const
{
  std::string const currentFile = __FILE__;
  std::string const sourceFile = currentFile.substr( 0, currentFile.size() - ( sizeof( "jitti.cpp" ) - 1 ) )
                                 + "templateSource.cpp";

  std::vector< std::string > defines = {
    "JITTI_TEMPLATE_HEADER_FILE='\"" + headerFile + "\"'",
    "JITTI_TEMPLATE_FUNCTION=" + templateFunction,
    "JITTI_TEMPLATE_PARAMS=\"" + templateParams + "\"",
    "JITTI_TEMPLATE_PARAMS_STRING='\"" + templateParams + "\"'"
  };

  return Compiler::createDynamicLibrary( sourceFile, defines, outputObject, outputLibrary );
}

} // namespace jitti
