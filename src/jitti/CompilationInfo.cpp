#include "CompilationInfo.hpp"
#include "../Macros.hpp"

#include <cstring>
#include <time.h>

static int monthStringToInt( char const * const month )
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

namespace jitti
{

namespace internal
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
time_t getCompileTime( char const * const date, char const * const time )
{
  struct tm dateTime {};
  dateTime.tm_mon = monthStringToInt( date );
  dateTime.tm_mday = std::atoi( date + 4 );
  dateTime.tm_year = std::atoi( date + 7 ) - 1900;

  dateTime.tm_hour = std::atoi( time );
  dateTime.tm_min = std::atoi( time + 3 );
  dateTime.tm_sec = std::atoi( time + 6 );

  dateTime.tm_isdst = -1;

  return mktime( &dateTime );
}

} // namespace internal

} // namespace jitti
