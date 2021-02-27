// Source includes
#include "jitti.hpp"

// System includes
#include <regex>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

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

namespace internal
{

int monthStringToInt( char const * const month )
{
  if ( strncmp( month, "Jan", 3 ) == 0 )
  { return 0; }
  if ( strncmp( month, "Feb", 3 ) == 0 )
  { return 1; }
  if ( strncmp( month, "Mar", 3 ) == 0 )
  { return 2; }
  if ( strncmp( month, "Apr", 3 ) == 0 )
  { return 3; }
  if ( strncmp( month, "May", 3 ) == 0 )
  { return 4; }
  if ( strncmp( month, "Jun", 3 ) == 0 )
  { return 5; }
  if ( strncmp( month, "Jul", 3 ) == 0 )
  { return 6; }
  if ( strncmp( month, "Aug", 3 ) == 0 )
  { return 7; }
  if ( strncmp( month, "Sep", 3 ) == 0 )
  { return 8; }
  if ( strncmp( month, "Oct", 3 ) == 0 )
  { return 9; }
  if ( strncmp( month, "Nov", 3 ) == 0 )
  { return 10; }
  if ( strncmp( month, "Dec", 3 ) == 0 )
  { return 11; }

  LVARRAY_ERROR( "Uncrecognized month: " << month );
  return -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
time_t getCompileTime( char const * const date, char const * const time )
{
  struct tm dateTime {};
  dateTime.tm_mon = monthStringToInt( date );
  dateTime.tm_mday = std::atoi( date + 4 );
  dateTime.tm_year = std::atoi( date + 7 );

  dateTime.tm_hour = std::atoi( time );
  dateTime.tm_min = std::atoi( time + 3 );
  dateTime.tm_sec = std::atoi( time + 6 );

  dateTime.tm_isdst = -1;

  return mktime( &dateTime );
}

}

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

namespace internal
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TypedDynamicLibrary compileTemplate( std::string const & compileCommand,
                                     bool const compilerIsNVCC,
                                     std::string const & linker,
                                     std::string const & linkArgs,
                                     std::string const & templateFunction,
                                     std::string const & templateParams,
                                     std::string const & headerFile,
                                     std::string const & outputObject,
                                     std::string const & outputLibrary )
{
  std::string const currentFile = __FILE__;
  std::string const sourceFile = currentFile.substr( 0, currentFile.size() - ( sizeof( "jitti.cpp" ) - 1 ) )
                                 + "templateSource.cpp"; 

  std::vector< std::string > const defines = {
    "JITTI_TEMPLATE_HEADER_FILE='\"" + headerFile + "\"'",
    "JITTI_TEMPLATE_FUNCTION=" + templateFunction,
    "JITTI_TEMPLATE_PARAMS=\"" + templateParams + "\"",
    "JITTI_TEMPLATE_PARAMS_STRING='\"" + templateParams + "\"'"
  };

  std::string fullCompileCommand = compileCommand;
  if( compilerIsNVCC )
  {
    fullCompileCommand += " -Xcompiler";
  }

  fullCompileCommand += " -fPIC -c " + sourceFile + " -o " + outputObject;

  std::string joinedDefines;
  for ( std::string const & define : defines )
  { joinedDefines += " -D " + define; }

  // You need to escape commas for NVCC.
  if( compilerIsNVCC )
  {
    fullCompileCommand += std::regex_replace( joinedDefines, std::regex( "," ), "\\," );
  }
  else
  {
    fullCompileCommand += joinedDefines;
  }

  LVARRAY_LOG( "\nCompiling " << sourceFile );
  LVARRAY_LOG_VAR( fullCompileCommand );
  LVARRAY_ERROR_IF( std::system( fullCompileCommand.c_str() ) != 0, fullCompileCommand );

  std::string linkCommand = linker + " -shared " + outputObject + " " + linkArgs + " -o " + outputLibrary;

  LVARRAY_LOG( "\nLinking " << outputObject );
  LVARRAY_LOG_VAR( linkCommand );
  LVARRAY_ERROR_IF( std::system( linkCommand.c_str() ) != 0, linkCommand );

  return TypedDynamicLibrary( outputLibrary );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void readDirectoryFiles( time_t const time,
                         std::string const & directory,
                         std::unordered_map< std::string, std::string > & libraries )
{
  DIR * const dirp = opendir( directory.c_str() );
  LVARRAY_ERROR_IF( dirp == nullptr, "Could not open '" << directory << "' as a directory." );

  struct dirent * dp;
  while( ( dp = readdir( dirp ) ) != nullptr )
  {
    std::string const fileName = dp->d_name;
    if( fileName.size() > 3 && fileName.substr( fileName.size() - 3 ) == ".so" )
    {
      std::string const filePath = directory + "/" + fileName;

      struct stat fileInfo;
      LVARRAY_ERROR_IF( stat( filePath.c_str(), &fileInfo ) != 0, "Error stat'ing '" << filePath << "'" );

      if( fileInfo.st_mtime > time )
      {
        auto const result = libraries.emplace( std::make_pair( std::move( fileName ), std::move( filePath ) ) );
        LVARRAY_ERROR_IF( !result.second, "A library with the name '" << fileName << "' already exists." );
      }
    }
  }

  closedir( dirp );
}

} // namespace internal

} // namespace jitti
