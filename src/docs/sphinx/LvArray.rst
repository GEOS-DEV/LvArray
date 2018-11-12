###############################################################################
LvArray Overview
###############################################################################


Goals
==================
The goal of LvArray is to provide a portable (CPU-GPU) and fast implementation of a 
multi-dimensional array. 

Language
==================
LvArray is written in standard c++14.

Array Classes
=================================
LvArray is split up into three classes that form a simple hierarchy.

   .. code-block:: c

      template< typename T, int NDIM, typename INDEX_TYPE = std::int_fast32_t >
      class ArraySlice

The class "ArraySlice" is the base class, contains pointers to the data, dims, and strides, and provides the operator[] interface.
The operator[] interface allows for range checking to be turned on and off via the cmake variable "ENABLE_ARRAY_BOUNDS_CHECK". 
The singlular goal of the ArraySlice class is to provide the mechanism for fast operator[] indexing, thus it contains no data or functions that are not required to perform this task.
When BoundsChecking is off, a 1D ArraySlice may be treated as a pointer to the data.

   .. code-block:: c

      template< typename T, int NDIM, typename INDEX_TYPE = std::int_fast32_t >
      class ArrayView : public ArraySlice<T, NDIM, INDEX_TYPE >

The class ArrayView inherits its operator[] interface from array slice. 
ArrayView also provides a fortran-like operator() interface where the indices can be specified in a comma-delimited manner.
In addition, ArrayView provides inspection functions such as "size()" to return the ranges of the array.
The main feature of ArrayView is that, it actually holds dim and stride data in a regular c-array, and holds the array data in a ChaiVector.
Thus when an ArrayView copy constructor is invoked, the underlying CHAI::ManagedArray copy constructor is invoked leading to a potential movement of data between host and device spaces. 
For example, if an ArrayView is captured-by-value in a lambda function, and that lambda function is executed on device, the ArrayView is guaranteed to be valid for the device memory space.
All accessors (operator()[]) are const, thus they may be called from an "ArrayView const"

   .. code-block:: c

      template< typename T, int NDIM, typename INDEX_TYPE >
      class Array : public ArrayView<T, NDIM, INDEX_TYPE>

The class Array inherits all functionality from ArrayView (and thus ArraySlice).
The key features of Array is that it provides allocation capablities, and it does NOT retain the copy constructor behavior of ArrayView.
Instead, the Array copy constructor will invoke a deep copy.


Usage
======================
The usage of Array is similar to that of a std::vector. You may declare it, and allocate data, and use operator[] to access data.
One distinct interface deviation from a std::vector is the use of const. 
In the Array classes, the type is either const or non-const and the type defines the access. 
This differs from std::vector in that a "std::vector<double> const" may not modify the contents, where an "Array<double> const" may modify its contents, but an "Array<double const>" may not.
There are user defined conversions to allow for easy casting between an Array to an "Array to const".

   .. code-block:: c

    Array1d< int, 1 > v(100);                      // Allocates an array of length 100.
    ArrayView< int, 1 > & vView = v;               // Defines an ArrayView to the data for the Array v.
    ArrayView< int const > const & vViewConst = v; // Defines an ArrayView to the data for the Array v that may not modify the values of v.

Bounds Checking
======================
Bounds checking is controlled by the cmake varaible "ENABLE_ARRAY_BOUNDS_CHECK". By default, ENABLE_ARRAY_BOUNDS_CHECK is OFF for Debug builds, and ON for Release builds.


