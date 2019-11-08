###############################################################################
LvArray Overview
###############################################################################

Goals
==================
The goal of ``LvArray`` is to provide an optimized portable implementation of a multi-dimensional array. 

Language
==================
``LvArray`` is written in standard c++14.

Features
==================
* Multidimensional slicing via ``operator[]``.
* Multidimensional access via the fortran-like ``operator()``.
* Interface simlar to ``std::vector`` for one dimensional arrays.
* Optional dependency on *CHAI* which enables automatic memory movement to the device (GPU).
* Performance similar to a restrict pointer.

Configurations
==================
* define ``USE_ARRAY_BOUNDS_CHECK`` to enable bounds checking of all array accesses, by default this is on in debug builds and off in release builds.

Array Classes
=================================
``LvArray`` is split up into four classes: ``ChaiVector``, ``Array``, ``ArrayView`` and ``ArraySlice``.
``ChaiVector`` adds a ``std::vector`` like interface to a ``chai::ManagedArray`` or a raw pointer when building without ``CHAI``.
``Array`` inherits from ``ArrayView`` and ``ArraySlice`` is the object returned by the slicing ``operator[]``.

The ``ChaiVector`` Class
---------------------------------
   .. code-block:: c++

      template < typename T >
      class ChaiVector
   
where ``T`` is the storage type.
``ChaiVector`` provides many of the same methods as ``std::vector``, the main difference is that although it can allocate data it does not take ownership and the ``free`` method must be called to do deallocation.
Similarly the copy constructor and ``operator=`` both do shallow copies of the data so there can be multiple ``ChaiVector`` pointing to the same data.
When building without *CHAI* the copy constructor and ``operator=`` both do the same thing, however when using *CHAI* the copy constructor calls the ``chai::ManagedArray`` copy constructor which can trigger movement to a new memory space.

The ``ArraySlice`` Class
---------------------------------
   .. code-block:: c++

      template< typename T, int NDIM, typename INDEX_TYPE = std::int_fast32_t >
      class ArraySlice

where ``T`` is the storage type, ``NDIM`` is the number of dimensions and ``INDEX_TYPE`` is the type used to index into the array.
The ``ArraySlice`` does not define any methods other than the slicing ``operator[]`` which returns an ``ArraySlice<T, NDIM - 1, INDEX_TYPE>`` when ``NDIM > 1`` and a reference to ``T`` when ``NDIM == 1``, a conversion operator to convert from ``ArraySlice<T, NDIM, INDEX_TYPE>`` to ``ArraySlice<T const, NDIM, INDEX_TYPE>``` and a conversion operator to ``T *``.
Because of this interface choice a 1D `ArraySlice` is interchangable with a pointer when ``USE_ARRAY_BOUNDS_CHECK`` is not defined.

The ``ArrayView`` Class
---------------------------------

      template< typename T, int NDIM, typename INDEX_TYPE = std::ptrdiff_t >
      class ArrayView

The ``ArrayView`` class is intended as a view into an ``Array`` and therefore does not own the data.
Like the ``ChaiVector`` both the copy constructor and ``operator=`` do a shallow copy and the copy constructor may trigger movement to a new memory space.
The ``ArrayView::operator[]`` is identical to the ``ArraySlice::operator[]``.
`ArrayView`` also defines an ``operator()`` which allows direct fortran-like access to the data.
In addition to those methods the ``ArrayView`` provides a ``size`` method as well as conversion operators to ``ArraySlice<T, NDIM, INDEX_TYPE>``, ``ArraySlice<T const, NDIM, INDEX_TYPE>``, ``ArrayView<T const, NDIM, INDEX_TYPE>`` and ``T *``.

The ``Array`` Class
---------------------------------
   .. code-block:: c++

      template< typename T, int NDIM, typename INDEX_TYPE = std::ptrdiff_t >
      class Array : public ArrayView< T, NDIM, INDEX_TYPE >
      
Unlike the ``ChaiVector`` and the ``ArrayView`` the ``Array`` class actually owns the data, and therefore you cannot have more than one `Array` pointing to the same data.
In addition to the methods inherited from ``ArrayView`` the ``Array`` class provides ``resize`` and ``reserve`` methods.
When ``NDIM == 1`` the ``Array`` class also defines the methods ``push_back``, ``pop_back``, ``insert`` and ``erase``.
When building with *CHAI* the ``Array`` also defines a ``move`` method which allows you to manually move the data between memory spaces.

Usage
======================
The usage of Array is similar to that of a std::vector. You may declare it, and allocate data, and use operator[] to access data.
One distinct interface deviation from a std::vector is the use of ``const``. 
In the ``Array`` classes, the type ``T`` is either const or non-const and the type defines the access. 
This differs from ``std::vector`` in that a ``std::vector<double> const`` may not modify the contents or do any reallocation, where as an ``Array<double> const`` may modify its contents but cannot do reallocation. An ``Array<double const> const`` can do neither.
A result of this is that all the methods in ``ArrayView`` are ``const`` since the ``ArrayView`` does no allocation.
There are user defined conversions to allow for easy casting between an Array to an "Array to const".

   .. code-block:: c++

    Array< int, 1 > v(100);                      // Allocates an array of length 100.
    Array< int const, 1 > & vConst = v;          // Defines a reference to the Array v that may not modify the values.
    ArrayView< int, 1 > const & vView = v;         // Defines an ArrayView to the data for the Array v.
    ArrayView< int const > const & vViewConst = v; // Defines an ArrayView to the data for the Array v that may not modify the values.
    
If you would like to use an ``Array`` inside a *RAJA* style device kernel you need capture an ``ArrayView`` by value in the lambda capture list. For example

   .. code-block:: c++
      
      Array< double, 1 > v(100);
      /* Do things with v on the host. */
      
      ArrayView< double, 1 > const & vView = v;
      
      /* Do things with vView on device. */
      forall( gpu(), 0, 100,
        [=] __device__ ( int i )
        {
          vView[ i ] *= 2;
        }
      );
       
If after launching the device kernel you would like to use the ``Array`` ``v`` again on the host you must either manually move it

  .. code-block:: c++
      
      v.move(chai::CPU);
      /* Do things with v or vView on the host. */
      
or capture it in a host kernel

  .. code-block:: c++
      
      forall( sequential(), 0, N,
         [=]( int i )
         {
            /* Do things with vView on the host. */
         }
      );
