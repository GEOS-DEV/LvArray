/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "system.hpp"
#include "Macros.hpp"

// System includes
#include <map>
#include <sstream>
#include <iostream>
#include <signal.h>
#include <fenv.h>
#include <cxxabi.h>
#include <string.h>

#if defined(__x86_64__)
  #include <pmmintrin.h>
#endif

#include <dlfcn.h>
#include <unwind.h>

#if defined( LVARRAY_ADDR2LINE_EXEC )
  #include <unistd.h>
  #include <sys/wait.h>
#endif

/**
 * @struct UnwindState
 * @brief Holds info used in unwindCallback.
 * @note Adapted from https://github.com/boostorg/stacktrace
 */
struct UnwindState
{
  /// The number of frames left to skip.
  std::size_t framesToSkip;

  /// A pointer to the current frame.
  void * * current;

  /// A pointer to the final frame.
  void * * end;
};

/**
 * @brief Callback used with _Unwind_Backtrace.
 * @param context
 * @param arg The UnwindState.
 * @note Adapted from https://github.com/boostorg/stacktrace
 */
static _Unwind_Reason_Code unwindCallback( _Unwind_Context * const context, void * const arg )
{
  // Note: do not write `::_Unwind_GetIP` because it is a macro on some platforms.
  // Use `_Unwind_GetIP` instead!
  UnwindState * const state = static_cast< UnwindState * >(arg);
  if( state->framesToSkip )
  {
    --state->framesToSkip;
    return _Unwind_GetIP( context ) ? _URC_NO_REASON : _URC_END_OF_STACK;
  }

  *state->current = reinterpret_cast< void * >( _Unwind_GetIP( context ) );

  ++state->current;
  if( !*(state->current - 1) || state->current == state->end )
  {
    return _URC_END_OF_STACK;
  }

  return _URC_NO_REASON;
}

/**
 * @brief Populate @p frames with the stack return addresses.
 * @param frames A pointer to the buffer to fill, must have length at least @p maxFrames.
 * @param maxFrames The maximum number of frames to collect.
 * @param skip The number of initial frames to skip.
 * @note Adapted from https://github.com/boostorg/stacktrace
 */
static std::size_t collect( void * * const frames, std::size_t const maxFrames, std::size_t const skip )
{
  std::size_t frames_count = 0;
  if( !maxFrames )
  {
    return frames_count;
  }

  UnwindState state = { skip + 1, frames, frames + maxFrames };
  _Unwind_Backtrace( &unwindCallback, &state );
  frames_count = state.current - frames;

  if( frames_count && frames[frames_count - 1] == 0 )
  {
    --frames_count;
  }

  return frames_count;
}

/**
 * @brief Return the demangled name of the function at @p address.
 * @param address The address of the function gotten from the stack frame.
 * @return the demangled name of the function at @p address.
 */
static std::string getFunctionNameFromFrame( void const * const address )
{
  Dl_info dli;
  const bool dl_ok = dladdr( address, &dli );
  if( dl_ok )
  {
    if( dli.dli_sname )
    {
      return LvArray::system::demangle( dli.dli_sname );
    }

    return dli.dli_fname;
  }

  return "Unknown";
}

#if defined( LVARRAY_ADDR2LINE_EXEC )

/**
 * @brief Return @c true iff @p path is an absolute path.
 * @param path The file path to inspect.
 * @return @c true iff @p path is an absolute path.
 */
static constexpr bool isAbsPath( char const * path )
{ return *path != '\0' && ( *path == ':' || *path == '/' || isAbsPath( path + 1 ) ); }

/**
 * @class UnwindState
 * @brief Used to fork a subprocess that executes @c LVARRAY_ADDR2LINE_EXEC and get the results.
 * @note Adapted from https://github.com/boostorg/stacktrace
 */
class Addr2LinePipe
{
public:

  /**
   * @brief Constructor.
   * @param flag The flag(s) to pass to addr2line.
   * @param execPath The path to the executable the address is from, usually the current executable.
   * @param addr The address to query.
   */
  Addr2LinePipe( char const * const flag, char const * const execPath, char const * const addr ):
    m_file( nullptr ),
    m_pid( 0 )
  {
    int pdes[ 2 ];
    char prog_name[] = STRINGIZE( LVARRAY_ADDR2LINE_EXEC );
    static_assert( isAbsPath( STRINGIZE( LVARRAY_ADDR2LINE_EXEC ) ),
                   "LVARRAY_ADDR2LINE_EXEC = " STRINGIZE( LVARRAY_ADDR2LINE_EXEC ) );

    char * argp[] = {
      prog_name,
      const_cast< char * >( flag ),
      const_cast< char * >( execPath ),
      const_cast< char * >( addr ),
      0
    };

    if( pipe( pdes ) < 0 )
    { return; }

    m_pid = fork();
    switch( m_pid )
    {
      case -1:
      {
        // Failed...
        close( pdes[ 0 ] );
        close( pdes[ 1 ] );
        return;

      }
      case 0:
        // We are the child.
        close( STDERR_FILENO );
        close( pdes[ 0 ] );
        if( pdes[ 1 ] != STDOUT_FILENO )
        { dup2( pdes[ 1 ], STDOUT_FILENO ); }

        // Do not use `execlp()`, `execvp()`, and `execvpe()` here!
        // `exec*p*` functions are vulnerable to PATH variable evaluation attacks.
        execv( prog_name, argp );
        _exit( 127 );
    }

    m_file = fdopen( pdes[ 0 ], "r" );
    close( pdes[ 1 ] );
  }

  /**
   * @brief User defined conversion to a file pointer.
   * @return A file pointer to the output of the addr2line execution.
   */
  operator FILE *() const
  { return m_file; }

  /// Destructor
  ~Addr2LinePipe()
  {
    if( m_file )
    {
      fclose( m_file );
      int pstat = 0;
      kill( m_pid, SIGKILL );
      waitpid( m_pid, &pstat, 0 );
    }
  }

private:
  /// A file pointer to the output of the addr2line execution.
  FILE * m_file;

  /// The process ID of the forked child process.
  pid_t m_pid;
};

/**
 * @brief Return the result of calling @c addr2line @p flag @c pathToCurrentExecutable @p addr.
 * @param flag The flag to pass to addr2line.
 * @param addr The address to pass to addr2line.
 * @return The result of calling @c addr2line @p flag @c pathToCurrentExecutable @p addr.
 */
static std::string addr2line( const char * flag, const void * addr )
{
  std::string res;

  Dl_info dli;
  if( dladdr( addr, &dli ) )
  {
    res = dli.dli_fname;
  }
  else
  {
    res.resize( 16 );
    int rlin_size = readlink( "/proc/self/exe", &res[ 0 ], res.size() - 1 );
    while( rlin_size == static_cast< int >( res.size() - 1 ) )
    {
      res.resize( res.size() * 4 );
      rlin_size = readlink( "/proc/self/exe", &res[ 0 ], res.size() - 1 );
    }
    if( rlin_size == -1 )
    {
      res.clear();
      return res;
    }
    res.resize( rlin_size );
  }

  std::ostringstream oss;
  oss << addr;

  Addr2LinePipe p( flag, res.c_str(), oss.str().c_str() );
  res.clear();

  if( !p )
  {
    return res;
  }

  char data[ 32 ];
  while( !feof( p ) )
  {
    if( fgets( data, sizeof( data ), p ) )
    {
      res += data;
    }
    else
    {
      break;
    }
  }

  // Trimming
  while( !res.empty() && ( res[ res.size() - 1 ] == '\n' || res[ res.size() - 1 ] == '\r' ) )
  {
    res.erase( res.size() - 1 );
  }

  return res;
}

#endif

/**
 * @brief Return the source location of @p address iff LVARRAY_ADDR2LINE_EXEC is defined.
 * @param address The address to get the source location of.
 * @return The source location of @p address iff LVARRAY_ADDR2LINE_EXEC is defined.
 */
static std::string getSourceLocationFromFrame( void const * const address )
{
  #if defined( LVARRAY_ADDR2LINE_EXEC )
  std::string const source_line = addr2line( "-Cpe", address );
  if( !source_line.empty() && source_line[0] != '?' )
  {
    return source_line;
  }
  #else
  LVARRAY_UNUSED_VARIABLE( address );
  #endif

  return "";
}

/**
 * @brief Return a string representing the current floating point exception.
 * @return A string representing the current floating point exception.
 */
static std::string getFpeDetails()
{
  std::ostringstream oss;
  int const fpe = fetestexcept( FE_ALL_EXCEPT );

  oss << "Floating point exception:";

  if( fpe & FE_DIVBYZERO )
  {
    oss << " Division by zero;";
  }
  if( fpe & FE_INEXACT )
  {
    oss << " Inexact result;";
  }
  if( fpe & FE_INVALID )
  {
    oss << " Invalid argument;";
  }
  if( fpe & FE_OVERFLOW )
  {
    oss << " Overflow;";
  }
  if( fpe & FE_UNDERFLOW )
  {
    oss << " Underflow;";
  }

  return oss.str();
}

namespace LvArray
{
namespace system
{

/// An alias for a function that takes an int and returns nothing.
using handle_type = void ( * )( int );

/// A map containing the initial signal handlers.
static std::map< int, handle_type > initialHandler;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string stackTrace( bool const location )
{
  constexpr int MAX_FRAMES = 25;
  void * array[ MAX_FRAMES ];

  std::size_t const size = collect( array, MAX_FRAMES, 1 );

  std::ostringstream oss;
  oss << "\n** StackTrace of " << size - 1 << " frames **\n";
  for( std::size_t i = 0; i < size; ++i )
  {
    oss << "Frame " << i << ": " << getFunctionNameFromFrame( array[ i ] );

    if( location )
    {
      oss << " " << getSourceLocationFromFrame( array[ i ] );
    }

    oss << "\n";
  }

  oss << "=====\n";

  return oss.str();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string demangle( char const * const name )
{
  if( name == nullptr )
  {
    return "";
  }

  int status = -4; // some arbitrary value to eliminate the compiler warning
  char * const demangledName = abi::__cxa_demangle( name, nullptr, nullptr, &status );

  std::string const result = (status == 0) ? demangledName : name;

  std::free( demangledName );

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string calculateSize( size_t const bytes )
{
  char const * suffix;
  uint shift;
  if( bytes >> 30 != 0 )
  {
    suffix = "GB";
    shift = 30;
  }
  else if( bytes >> 20 != 0 )
  {
    suffix = "MB";
    shift = 20;
  }
  else if( bytes >> 10 != 0 )
  {
    suffix = "KB";
    shift = 10;
  }
  else
  {
    suffix = "B";
    shift = 0;
  }

  double const units = double( bytes ) / ( 1 << shift );

  char result[10];
  std::snprintf( result, 10, "%.1f %s", units, suffix );
  return result;
}

/**
 * @brief A static pointer to the error handler.
 * @note When using a std::function directly there was an exit time error, by not deallocating it we get around it.
 */
std::function< void() > * s_errorHandler = nullptr;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void setErrorHandler( std::function< void() > const & handler )
{
  LVARRAY_ERROR_IF( handler == nullptr, "Error handler cannot be null." );
  if( s_errorHandler != nullptr )
  {
    delete s_errorHandler;
  }

  s_errorHandler = new std::function< void() >( handler );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void callErrorHandler()
{
  if( s_errorHandler == nullptr || *s_errorHandler == nullptr )
  {
    return std::abort();
  }

  (*s_errorHandler)();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void stackTraceHandler( int const sig, bool const exit )
{
  std::ostringstream oss;

  if( sig >= 0 && sig < NSIG )
  {
    // sys_signame not available on linux, so just print the code; strsignal is POSIX
    oss << "Received signal " << sig << ": " << strsignal( sig ) << "\n";

    if( sig == SIGFPE )
    {
      oss << getFpeDetails() << "\n";
    }
  }

  oss << stackTrace( true ) << std::endl;
  std::cout << oss.str();

  if( exit )
  {
    // An infinite loop was encountered when an FPE was received. Resetting the handlers didn't
    // fix it because they would just recurse. This does.
    setSignalHandling( nullptr );
    callErrorHandler();
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void setSignalHandling( void (* handler)( int ) )
{
  initialHandler[SIGHUP] = signal( SIGHUP, handler );
  initialHandler[SIGINT] = signal( SIGINT, handler );
  initialHandler[SIGQUIT] = signal( SIGQUIT, handler );
  initialHandler[SIGILL] = signal( SIGILL, handler );
  initialHandler[SIGTRAP] = signal( SIGTRAP, handler );
  initialHandler[SIGABRT] = signal( SIGABRT, handler );
#if  (defined(_POSIX_C_SOURCE) && !defined(_DARWIN_C_SOURCE))
  initialHandler[SIGPOLL] = signal( SIGPOLL, handler );
#else
  initialHandler[SIGIOT] = signal( SIGIOT, handler );
  initialHandler[SIGEMT] = signal( SIGEMT, handler );
#endif
  initialHandler[SIGFPE] = signal( SIGFPE, handler );
  initialHandler[SIGKILL] = signal( SIGKILL, handler );
  initialHandler[SIGBUS] = signal( SIGBUS, handler );
  initialHandler[SIGSEGV] = signal( SIGSEGV, handler );
  initialHandler[SIGSYS] = signal( SIGSYS, handler );
  initialHandler[SIGPIPE] = signal( SIGPIPE, handler );
  initialHandler[SIGTERM] = signal( SIGTERM, handler );

  return;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void resetSignalHandling()
{
  for( auto a : initialHandler )
  {
    signal( a.first, a.second );
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int getDefaultFloatingPointExceptions()
{
  return ( FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID );
}

#if defined(__APPLE__) && defined(__MACH__)&& !defined(__x86_64__)
static void
fpe_signal_handler( int sig, siginfo_t *sip, void *scp )
{
  LVARRAY_UNUSED_VARIABLE( sig );
  LVARRAY_UNUSED_VARIABLE( scp );

  int fe_code = sip->si_code;

  printf( "In signal handler : " );

  if( fe_code == ILL_ILLTRP )
    printf( "Illegal trap detected\n" );
  else
    printf( "Code detected : %d\n", fe_code );

  abort();
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int enableFloatingPointExceptions( int const exceptions )
{
#if defined(__APPLE__) && defined(__MACH__)
#if !defined(__x86_64__)

  LVARRAY_UNUSED_VARIABLE( exceptions );

  fenv_t env;
  fegetenv( &env );

  env.__fpcr = env.__fpcr | __fpcr_trap_invalid;
  fesetenv( &env );

  struct sigaction act;
  act.sa_sigaction = fpe_signal_handler;
  sigemptyset ( &act.sa_mask );
  act.sa_flags = SA_SIGINFO;
  sigaction( SIGILL, &act, NULL );
  return 0;
#else
  // Public domain polyfill for feenableexcept on OS X
  // http://www-personal.umich.edu/~williams/archive/computation/fe-handling-example.c
  static fenv_t fenv;
  int const newExcepts = exceptions & FE_ALL_EXCEPT;

  if( fegetenv( &fenv ))
  {
    return -1;
  }
  // all previous masks
  int const oldExcepts = fenv.__control & FE_ALL_EXCEPT;

  // unmask
  fenv.__control &= ~newExcepts;
  fenv.__mxcsr   &= ~(newExcepts << 7);

  return fesetenv( &fenv ) ? -1 : oldExcepts;
#endif
#else
  int const oldExceptions = feenableexcept( exceptions );
  LVARRAY_ERROR_IF_EQ( oldExceptions, -1 );
  return oldExceptions;
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int disableFloatingPointExceptions( int const exceptions )
{
#if defined(__APPLE__) && defined(__MACH__)
#if !defined(__x86_64__)
  LVARRAY_UNUSED_VARIABLE( exceptions );
  return 0;
#else
  // Public domain polyfill for feenableexcept on OS X
  // http://www-personal.umich.edu/~williams/archive/computation/fe-handling-example.c
  static fenv_t fenv;
  int const newExcepts = exceptions & FE_ALL_EXCEPT;

  if( fegetenv( &fenv ))
  {
    return -1;
  }
  // all previous masks
  int const oldExcepts = ~( fenv.__control & FE_ALL_EXCEPT );

  // mask
  fenv.__control |= newExcepts;
  fenv.__mxcsr   |= newExcepts << 7;

  return fesetenv( &fenv ) ? -1 : oldExcepts;
#endif
#else
  int const oldExceptions = fedisableexcept( exceptions );
  LVARRAY_ERROR_IF_EQ( oldExceptions, -1 );
  return oldExceptions;
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void setFPE()
{
#if defined(__APPLE__) && defined(__MACH__)
#if !defined(__x86_64__)

#else
  fesetenv( FE_DFL_DISABLE_SSE_DENORMS_ENV );
#endif
#elif defined(__x86_64__)
  _MM_SET_FLUSH_ZERO_MODE( _MM_FLUSH_ZERO_ON );
  _MM_SET_DENORMALS_ZERO_MODE( _MM_DENORMALS_ZERO_ON );
#endif
#if defined(__x86_64__)
  enableFloatingPointExceptions( getDefaultFloatingPointExceptions() );
#endif
}

} // namespace system
} // namespace LvArray
