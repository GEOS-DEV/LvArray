.. ##
.. ## Copyright (c) 2021, Lawrence Livermore National Security, LLC
.. ## and LvArray project contributors. See the LICENCE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

###############################################################################
Development Aids (Coming soon)
###############################################################################

Coming soon: documentation for
  
  - ``LvArray::arrayManipulation``
  - ``LvArray::sortedArrayManipulation``
  - ``LvArray::bufferManipulation``
  - Creating a new buffer type

Debugging helpers
=================

TotalView
---------

GDB
---

``LvArray`` comes with a collection of custom `GDB pretty printers <https://sourceware.org/gdb/onlinedocs/gdb/Pretty-Printing.html>`_
for buffer and array classes that simplify inspection of data in a debug session by adding extra member(s) to the object display
that "views" the data in a convenient format recognizable by GDB and IDEs. This eliminates the need for users of these classes
to understand the particular class structure and spend time obtaining and casting the actual data pointer to the right type.

.. note::

   In order to allow GDB to load the pretty printer script (``scripts/gdb-printers.py``), the following lines must be added to ``.gdbinit``
   (with the proper path substituted):

   .. code-block:: none

      add-auto-load-safe-path /path/to/LvArray/
      directory /path/to/LvArray/

   The first line allows GDB to load python scripts located under ``LvArray`` source tree.
   The second line adds ``LvArray`` source tree to the source path used by GDB to lookup file names, which allows it to find the
   pretty printer script by relative path (which is how it is referenced from the compiled binary).
   The ``.gdbinit`` file may be located under the user's home directory or current working directory where GDB is invoked.

The following pretty printers are available:

  - For buffer types (``StackBuffer``, ``MallocBuffer`` and ``ChaiBuffer``) the extra member ``gdb_view`` is a C-array of the appropriate type
    and size equal to the buffer size.
  - For multidimensional arrays and slices (``ArraySlice``, ``ArrayView`` and ``Array``) the extra member ``gdb_view`` is a multidimensional
    C-array with the sizes equal to current runtime size of the array or slice. This only works correctly for arrays with default
    (unpermuted) layout.
  - For ``ArrayOfArrays`` (and consequently all of its descendants) each contained sub-array is added as a separate child that is
    again a C-array of size equal to that of the sub-array.

Example below demonstrates the difference between pretty printer output and raw output:

.. code-block:: none

   (gdb) p dofNumber
   $1 = const geos::arrayView1d & of size [10] = {gdb_view = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45}}
   (gdb) p /r dofNumber
   $2 = (const geos::arrayView1d &) @0x7ffcda2a8ab0: {static NDIM = 1, static USD = 0, m_dims = {data = {10}}, m_strides = {data = {1}}, m_dataBuffer = {static hasShallowCopy = <optimized out>, m_pointer = 0x55bec1de4860, m_capacity = 10, m_pointerRecord = 0x55bec1dfa5c0}}

This is how the variable is viewed in a debugging session in CLion IDE:

.. code-block:: none

   dofNumber = {const geos::arrayView1d &}
    gdb_view = {const long long [10]}
     [0] = {const long long} 0
     [1] = {const long long} 5
     [2] = {const long long} 10
     [3] = {const long long} 15
     [4] = {const long long} 20
     [5] = {const long long} 25
     [6] = {const long long} 30
     [7] = {const long long} 35
     [8] = {const long long} 40
     [9] = {const long long} 45
    NDIM = {const int} 1
    USD = {const int} 0
    m_dims = {LvArray::typeManipulation::CArray<long, 1>}
    m_strides = {LvArray::typeManipulation::CArray<long, 1>}
    m_dataBuffer = {LvArray::ChaiBuffer<long long const>}

.. warning::

   GDB printers are for host-only data. Attempting to display an array or buffer whose active pointer is a device pointer may have
   a range of outcomes, from incorrect data being displayed, to debugger crashes. The printer script is yet to be tested with ``cuda-gdb``.
