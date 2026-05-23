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

/*
 * Implementation map:
 * 1) Low-level stack frame collection and symbol/source resolution.
 * 2) Public stack trace and type demangling helpers.
 * 3) Error and signal handling (including FPE signal decoding).
 * 4) Cross-platform floating-point environment control:
 *    - enable/disable traps,
 *    - translate FE_* masks where platform internals differ,
 *    - flush denormals/subnormals to zero.
 */

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
  // "-Cpe" asks addr2line for demangled symbol + file:line + inlined call info.
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

// Snapshot of handlers that were active before LvArray installs its own handlers.
// We keep these so resetSignalHandling() can restore previous process behavior.
static struct sigaction g_oldAction[NSIG];
static bool g_oldActionSet[NSIG] = {};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string stackTrace( bool const location )
{
  constexpr int MAX_FRAMES = 25;
  void * array[ MAX_FRAMES ];

  // Skip this helper frame so frame 0 is the caller of stackTrace().
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

/**
 * @brief Default signal handler: print diagnostics and exit.
 * @param sig The signal received.
 * @param info Additional signal information (si_code identifies the fault).
 * @param ucontext Platform-specific user context (unused).
 * @note Uses non-async-signal-safe functions (ostringstream, strsignal, cerr)
 *       for actionable diagnostics. This is a deliberate best-effort tradeoff;
 *       if the process is in a state where these fail, the _Exit below will
 *       still terminate cleanly.
 */
void signalHandler( int sig, siginfo_t * info, void * ucontext )
{
  LVARRAY_UNUSED_VARIABLE( ucontext );
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
          // POSIX SIGFPE subcodes let us report the concrete floating-point fault.
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
    else if( sig == SIGILL && info )
    {
#if defined(__APPLE__) && defined(__MACH__) && defined(__aarch64__)
      if( info->si_code == ILL_ILLTRP )
      {
        // Apple arm64 may report FP traps as SIGILL/ILL_ILLTRP instead of SIGFPE.
        // In that mode the subtype is not exposed, so we emit the best available text.
        oss << "  SIGILL si_code = " << info->si_code
            << " (possible floating-point trap; subtype unavailable on this platform)\n";
      }
      else
#endif
      {
        oss << "  SIGILL si_code = " << info->si_code << "\n";
      }
    }
  }

  oss << stackTrace( true ) << std::endl;
  std::cerr << oss.str();

  std::_Exit( ( sig > 0 ) ? ( 128 + sig ) : EXIT_FAILURE );
}

/// Signals that LvArray installs handlers for.
/// Do NOT include SIGKILL or SIGSTOP — they cannot be caught.
static constexpr int handledSignals[] = {
#ifdef SIGHUP
  SIGHUP,
#endif
#ifdef SIGINT
  SIGINT,
#endif
#ifdef SIGQUIT
  SIGQUIT,
#endif
#ifdef SIGILL
  SIGILL,
#endif
#ifdef SIGTRAP
  SIGTRAP,
#endif
#ifdef SIGABRT
  SIGABRT,
#endif
#ifdef SIGFPE
  SIGFPE,
#endif
#ifdef SIGBUS
  SIGBUS,
#endif
#ifdef SIGSEGV
  SIGSEGV,
#endif
#ifdef SIGSYS
  SIGSYS,
#endif
#ifdef SIGPIPE
  SIGPIPE,
#endif
#ifdef SIGTERM
  SIGTERM,
#endif
};

void setSignalHandling( void (* handler)( int, siginfo_t * info, void * ) )
{
  struct sigaction sa;
  memset( &sa, 0, sizeof( sa ) );
  sigemptyset( &sa.sa_mask );
  if( handler == nullptr )
  {
    sa.sa_handler = SIG_DFL;
    sa.sa_flags = 0;
  }
  else
  {
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO;
  }

  for( int sig : handledSignals )
  {
    if( sig <= 0 || sig >= NSIG )
    {
      continue;
    }

    if( g_oldActionSet[sig] )
    {
      // We already cached the original action for this signal, so do not overwrite it.
      sigaction( sig, &sa, nullptr );
    }
    else if( sigaction( sig, &sa, &g_oldAction[sig] ) == 0 )
    {
      // First install: capture previous action for future resetSignalHandling().
      g_oldActionSet[sig] = true;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void resetSignalHandling()
{
  for( int sig : handledSignals )
  {
    if( sig <= 0 || sig >= NSIG || !g_oldActionSet[sig] )
    {
      continue;
    }

    sigaction( sig, &g_oldAction[sig], nullptr );
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int getDefaultFloatingPointExceptions()
{
  // These are the "hard" numerical faults that should typically stop execution.
  return ( FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID );
}

/// Translate standard FE_* masks to platform-specific fpcr trap bits.
/// This is only needed on Apple arm64 where fpcr uses a different bit layout.
/// Linux (x86 and aarch64) uses feenableexcept/fedisableexcept directly.
#if defined(__APPLE__) && defined(__MACH__) && !defined(__x86_64__)
static unsigned long long int translateFloatingPointException( unsigned long long int const exception )
{
  // Darwin arm64 stores trap masks in fpcr-specific bits (__fpcr_trap_*),
  // so FE_* must be translated before writing env.__fpcr.
  unsigned long long int result = 0;
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
  return result;
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int enableFloatingPointExceptions( int const exceptions )
{
#if defined(__APPLE__) && defined(__MACH__)
#if !defined(__x86_64__)

  // Apple arm64 path: manipulate fpcr trap bits via fenv_t.
  unsigned long long int const exceptionMasks = translateFloatingPointException( exceptions );

  fenv_t env;
  if( fegetenv( &env ))
  {
    return -1;
  }

  env.__fpcr |= exceptionMasks;
  return fesetenv( &env ) ? -1 : 0;
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
  // Linux/glibc path: use native API.
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
  // Apple arm64 path mirrors enableFloatingPointExceptions():
  // read fpcr, clear selected trap bits, and return previous FE_* state.
  unsigned long long int const exceptionMasks = translateFloatingPointException( exceptions );

  fenv_t env;
  if( fegetenv( &env ))
  {
    return -1;
  }

  int oldExcepts = 0;
  if( env.__fpcr & __fpcr_trap_inexact ) oldExcepts |= FE_INEXACT;
  if( env.__fpcr & __fpcr_trap_underflow ) oldExcepts |= FE_UNDERFLOW;
  if( env.__fpcr & __fpcr_trap_overflow ) oldExcepts |= FE_OVERFLOW;
  if( env.__fpcr & __fpcr_trap_divbyzero ) oldExcepts |= FE_DIVBYZERO;
  if( env.__fpcr & __fpcr_trap_invalid ) oldExcepts |= FE_INVALID;

  env.__fpcr &= ~exceptionMasks;
  return fesetenv( &env ) ? -1 : oldExcepts;
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
  // Linux/glibc path: use native API.
  int const oldExceptions = fedisableexcept( exceptions );
  LVARRAY_ERROR_IF_EQ( oldExceptions, -1 );
  return oldExceptions;
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int queryEnabledFloatingPointExceptions()
{
#if defined(__APPLE__) && defined(__MACH__)
#if !defined(__x86_64__)
  // Apple arm64: read FPCR trap-enable bits and translate back to FE_*.
  // If the underlying FPU is trapless, the bits we previously wrote will read
  // back as 0 here and the caller can warn the user.
  fenv_t env;
  if( fegetenv( &env ) )
  {
    return 0;
  }

  int enabled = 0;
  if( env.__fpcr & __fpcr_trap_inexact )   enabled |= FE_INEXACT;
  if( env.__fpcr & __fpcr_trap_underflow ) enabled |= FE_UNDERFLOW;
  if( env.__fpcr & __fpcr_trap_overflow )  enabled |= FE_OVERFLOW;
  if( env.__fpcr & __fpcr_trap_divbyzero ) enabled |= FE_DIVBYZERO;
  if( env.__fpcr & __fpcr_trap_invalid )   enabled |= FE_INVALID;
  return enabled;
#else
  // Apple x86: a bit SET in fenv.__control means the exception is masked
  // (disabled). The enabled set is the bitwise inverse, restricted to
  // FE_ALL_EXCEPT.
  fenv_t env;
  if( fegetenv( &env ) )
  {
    return 0;
  }
  return ( ~env.__control ) & FE_ALL_EXCEPT;
#endif
#else
  // Linux/glibc: native API. fegetexcept returns -1 on error; treat as none.
  int const enabled = fegetexcept();
  return ( enabled < 0 ) ? 0 : enabled;
#endif
}

static void enableFlushDenormalsToZero()
{
  // Flushing denormals prevents very slow subnormal arithmetic paths on many CPUs.
  // The mechanism is architecture-specific, but intent is the same.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

  // x86/x86-64: MXCSR control, via SSE intrinsics.
  _MM_SET_FLUSH_ZERO_MODE( _MM_FLUSH_ZERO_ON );
  #ifdef _MM_DENORMALS_ZERO_ON
  _MM_SET_DENORMALS_ZERO_MODE( _MM_DENORMALS_ZERO_ON );
  #endif

#elif defined(__aarch64__)
  // AArch64: control FPCR directly.
  unsigned long long fpcr = 0;
  asm volatile( "mrs %0, fpcr" : "=r"( fpcr ) );
  fpcr |= ( 1ULL << 24 ); // FZ bit.
  asm volatile( "msr fpcr, %0" : : "r"( fpcr ) );

#elif defined(__arm__) && !defined(__aarch64__)

  // 32-bit ARM with VFP/NEON: FPSCR control.
  unsigned int fpscr;
  asm volatile( "vmrs %0, fpscr" : "=r"( fpscr ) );
  fpscr |= ( 1u << 24 ); // FZ bit.
  asm volatile( "vmsr fpscr, %0" : : "r"( fpscr ) );

#else
  // Unknown or unsupported architecture: no-op.
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
