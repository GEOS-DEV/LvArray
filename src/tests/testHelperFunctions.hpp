

#include <fenv.h>
#include <xmmintrin.h>
#include <cmath>
#include <float.h>
// API coverage tests
// Each test should be documented with the interface functions being tested

namespace testFloatingPointExceptionsHelpers
{
void func3(double divisor);

void func2(double divisor);

void func1(double divisor);

void func0(double divisor);

void testStackTrace(double divisor);

void show_fe_exceptions(void);

double uf_test(double x, double denominator);

double of_test( double x, double y );

double invalid_test( double x );
}

