
#ifdef USE_CUDA

#define CUDA_TEST(X, Y)              \
  static void cuda_test_##X##Y();    \
  TEST(X, Y) { cuda_test_##X##Y(); } \
  static void cuda_test_##X##Y()

#endif

struct Tensor
{
  double x, y, z;

  LVARRAY_HOST_DEVICE Tensor():
    x(), y(), z()
  {}

  LVARRAY_HOST_DEVICE explicit Tensor( double val ):
    x( 3 * val ), y( 3 * val + 1 ), z( 3 * val + 2 )
  {}

  LVARRAY_HOST_DEVICE Tensor( const Tensor & from) :
    x(from.x),
    y(from.y),
    z(from.z)
  {}

  LVARRAY_HOST_DEVICE Tensor& operator=( const Tensor& other )
  {
    x = other.x;
    y = other.y;
    z = other.z;
    return *this;
  }

  LVARRAY_HOST_DEVICE Tensor& operator*=( const Tensor& other )
  {
    x *= other.x;
    y *= other.y;
    z *= other.z;
    return *this;
  }

  LVARRAY_HOST_DEVICE Tensor operator*( const Tensor& other ) const
  {
    Tensor result = *this;
    result *= other;
    return result;
  }

  bool operator==( const Tensor& other ) const
  {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    return x == other.x && y == other.y && z == other.z;
#pragma GCC diagnostic pop
  }
};

std::ostream& operator<<(std::ostream& stream, const Tensor & t )
{
  stream << "(" << t.x << ", " << t.y << ", " << t.z << ")";
  return stream;
}
