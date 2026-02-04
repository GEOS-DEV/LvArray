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

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <xmmintrin.h>   // or <immintrin.h>
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

void signalHandler( int sig, siginfo_t * info, void * /*ucontext*/ )
{
  std::ostringstream oss;

  if( sig >= 0 && sig < NSIG )
  {
    oss << "Received signal " << sig << ": " << strsignal( sig ) << "\n";

    if( sig == SIGFPE )
    {
      if( info )
      {
        oss << "  SIGFPE si_code = " << info->si_code << " ";

        switch( info->si_code )
        {
          case FPE_FLTDIV: oss << "(floating divide by zero)\n"; break;
          case FPE_FLTOVF: oss << "(floating overflow)\n"; break;
          case FPE_FLTUND: oss << "(floating underflow)\n"; break;
          case FPE_FLTINV: oss << "(floating invalid operation)\n"; break;
          case FPE_FLTRES: oss << "(floating inexact)\n"; break;
          case FPE_INTDIV: oss << "(integer divide by zero)\n"; break;
          case FPE_INTOVF: oss << "(integer overflow)\n"; break;
          default:         oss << "(other)\n"; break;
        }
      }
    }
  }

  oss << stackTrace( true ) << std::endl;
  std::cerr << oss.str();

  std::_Exit( 1 );
}


static struct sigaction g_oldAction[NSIG];

void setSignalHandling( void (* handler)( int, siginfo_t * info, void * ) )
{
  struct sigaction sa;
  sigemptyset( &sa.sa_mask );
  sa.sa_sigaction = handler;
  sa.sa_flags = SA_SIGINFO;

  auto install = [&]( int sig )
  {
    sigaction( sig, &sa, &g_oldAction[sig] );
  };

  install( SIGHUP );
  install( SIGINT );
  install( SIGQUIT );
  install( SIGILL );
  install( SIGTRAP );
  install( SIGABRT );
  install( SIGFPE );
  install( SIGBUS );
  install( SIGSEGV );
  install( SIGSYS );
  install( SIGPIPE );
  install( SIGTERM );
  // Do NOT try SIGKILL/SIGSTOP: they can’t be caught.
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int getDefaultFloatingPointExceptions()
{
  return ( FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID );
}

unsigned long long int translateFloatingPointException( unsigned long long int const exception )
{
  unsigned long long int result = 0;
#if defined(__APPLE__) && defined(__MACH__) // if apple
  if( exception & FE_INEXACT )
  {
    result |= __fpcr_trap_inexact;
  }
  if( exception & FE_UNDERFLOW )
  {
    result |= __fpcr_trap_underflow;
  }
  if( exception & FE_OVERFLOW )
  {
    result |= __fpcr_trap_overflow;
  }
  if( exception & FE_DIVBYZERO )
  {
    result |= __fpcr_trap_divbyzero;
  }
  if( exception & FE_INVALID )
  {
    result |= __fpcr_trap_invalid;
  }

#if defined(__arm__) || defined(__arm64__) // if apple arm
#elif defined(__x86_64__) // if apple x86_64
#else // if apple but not arm or x86_64
  std::cerr<< "LvArray::system::translateFloatingPointException() not implemented for this architecture" << std::endl;
#endif


#else // if not apple
#if defined(__x86_64__)
  result = exception;
#endif
#endif

return result;
}

#if defined(__APPLE__) && defined(__MACH__)&& !defined(__x86_64__)
static void
fpe_signal_handler( int sig, siginfo_t *sip, void *scp )
{
  LVARRAY_UNUSED_VARIABLE( sig );
  LVARRAY_UNUSED_VARIABLE( scp );

  int fe_code = sip->si_code;

  if( fe_code == ILL_ILLTRP )
    printf( "Illegal trap detected. If you see this you have a FPE, but Apple Silicon doesn't provide data on which FPE has occured.\n" );
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

   unsigned long long int const exceptionMasks = translateFloatingPointException( exceptions );

  fenv_t env;
  fegetenv( &env );

//  std::cout<<std::hex<<"env.__fpcr = " << env.__fpcr << std::endl;
  env.__fpcr = env.__fpcr | exceptionMasks ;
//  std::cout<<std::hex<<"env.__fpcr = " << env.__fpcr << std::endl;

  fesetenv( &env );

  struct sigaction act;
  act.sa_sigaction = fpe_signal_handler;
  sigemptyset ( &act.sa_mask );
  act.sa_flags = SA_SIGINFO;
  sigaction( SIGFPE, &act, NULL );
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

static void enableFlushDenormalsToZero()
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

    // x86/x86-64: MXCSR control, via SSE intrinsics
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    #ifdef _MM_DENORMALS_ZERO_ON
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    #endif

#elif defined(__aarch64__)

    // AArch64: control FPCR (GCC/Clang builtins)
    unsigned long fpcr = __builtin_aarch64_get_fpcr();
    // FZ bit is bit 24 in FPCR (flush-to-zero)
    fpcr |= (1ul << 24);
    __builtin_aarch64_set_fpcr(fpcr);

#elif defined(__arm__) && !defined(__aarch64__)

    // 32-bit ARM with VFP/NEON: FPSCR control
    unsigned int fpscr;
    asm volatile("vmrs %0, fpscr" : "=r"(fpscr));
    // FZ bit is also bit 24 in FPSCR
    fpscr |= (1u << 24);
    asm volatile("vmsr fpscr, %0" : : "r"(fpscr));

#else
    std::cout<< "LvArray::system::enableFlushDenormalsToZero() did not work "<<std::endl;
    // Unknown or unsupported architecture: no-op.
    // Could add a runtime warning or static_assert behind a config macro.
    (void)0;

#endif
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void setFPE()
{
enableFloatingPointExceptions( getDefaultFloatingPointExceptions() );
enableFlushDenormalsToZero();
}

} // namespace system
} // namespace LvArray
