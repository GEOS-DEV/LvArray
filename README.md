# LvArray
Description
-----------
The LvArray project is a collection of array classes for use in high-performance
simulation software.
LvArray provides:
 1. Multi-dimensional arrays with permutable data layout.
 2. Sorted Arrays are similar to [std::set](https://en.cppreference.com/w/cpp/container/set) but are contigous in memory.
 3. ArrayOfArrays provide a 2-dimensional array with variable sized second dimension.
 4. ArrayOfSets provide a 2-dimensional array with variable sized second dimension that are sorted.
 5. CRSMatrix for storage of sparse matricies.
All components of LvArray provide smart management of data motion between memory
spaces on systems with hetergeneous memory (e.g. CPU/GPU) through lambda copy
semantics in [RAJA](https://github.com/LLNL/RAJA), similar to the implementation of the [CHAI::ManagedArray](https://github.com/LLNL/CHAI)
 

Documentation
---------------------

Full documenation is hosted at [readthedocs](https://lvarray.readthedocs.io/en/latest/)

Authors
-------
See [Github](https://github.com/GEOSX/cxx-utilities/graphs/contributors)

Release
-------

Copyright (c) 2018, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.

All rights reserved.

Unlimited Open Source - LGPL 2.1 License

For release details and restrictions, please read the LICENSE.txt file.
It is also linked here:
[LICENSE](./LICENSE) 
`LLNL-Code-746361`  `OCEC-18-021`

Next release will be BSD.