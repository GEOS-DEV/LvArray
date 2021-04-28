.. ##
.. ## Copyright (c) 2021, Lawrence Livermore National Security, LLC
.. ## and LvArray project contributors. See the LICENCE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

##############
Buffer Classes
##############

The buffer classes are the backbone of every LvArray class. A buffer class is responsible for allocating, reallocating and de-allocating a chunk of memory as well as moving it between memory spaces. A buffer is not responsible for managing the lifetime of the objects in their allocation. In general buffer classes have shallow copy semantics and do not de-allocate their allocations upon destruction. Buffer classes implement the copy and move constructors as well as the copy and move assignment operators. They also have a default constructor that leaves them in an uninitialized state. In general it is only safe to assign to an uninitialized buffer although different buffer implementations may allow other operations. To construct an initialized buffer pass a dummy boolean argument, this value of the parameter is not important and it only exists to differentiate it from the default constructor. Once created an initialized buffer must be free'd, either directly or though one of its copies. There are currently three buffer implementations: ``LvArray::MallocBuffer``, ``LvArray::ChaiBuffer`` and ``LvArray::StackBuffer``.

``LvArray::MallocBuffer``
-------------------------
As you might have guessed ``LvArray::MallocBuffer`` uses ``malloc`` and ``free`` to handle its allocation. Copying a ``LvArray::MallocBuffer`` does not copy the allocation. The allocation of a ``LvArray::MallocBuffer`` lives exclusively on the host and as such it will abort the program if you try to move it to or touch it in any space other than ``MemorySpace::host``.

.. literalinclude:: ../../examples/exampleBuffers.cpp
  :language: c++
  :start-after: // Sphinx start after MallocBuffer
  :end-before: // Sphinx end before MallocBuffer

*[Source: examples/exampleBuffers.cpp]*

``LvArray::ChaiBuffer``
-----------------------
``LvArray::ChaiBuffer`` uses `CHAI <https://github.com/LLNL/CHAI>`_ to manage an allocation which can exist on both the host and device, it functions similarly to the ``chai::ManagedArray``. Like the ``LvArray::MallocBuffer`` copying a ``LvArray::ChaiBuffer`` via the assignment operators or the move constructor do not copy the allocation. The unique feature of the ``LvArray::ChaBuffer`` is that when it is copy constructed if the CHAI execution space is set it will move its allocation to the appropriate space creating an allocation there if it did not already exist.

.. literalinclude:: ../../examples/exampleBuffers.cpp
  :language: c++
  :start-after: // Sphinx start after ChaiBuffer captureOnDevice
  :end-before: // Sphinx end before ChaiBuffer captureOnDevice

*[Source: examples/exampleBuffers.cpp]*

In order to prevent unnecessary memory motion if the type contained in the ``LvArray::ChaiBuffer`` is ``const`` then the data is not touched in any space it is moved to.

.. literalinclude:: ../../examples/exampleBuffers.cpp
  :language: c++
  :start-after: // Sphinx start after ChaiBuffer captureOnDeviceConst
  :end-before: // Sphinx end before ChaiBuffer captureOnDeviceConst

*[Source: examples/exampleBuffers.cpp]*

``LvArray::ChaiBuffer`` supports explicit movement and touching as well via the methods ``move`` and ``registerTouch``.

Whenever a ``LvArray::ChaiBuffer`` is moved between memory spaces it will print the size of the allocation, the type of the buffer and the name. Both the name and the type can be set with the ``setName`` method. If this behavior is not desired it can be disabled with ``chai::ArrayManager::getInstance()->disableCallbacks()``.

.. literalinclude:: ../../examples/exampleBuffers.cpp
  :language: c++
  :start-after: // Sphinx start after ChaiBuffer setName
  :end-before: // Sphinx end before ChaiBuffer setName

*[Source: examples/exampleBuffers.cpp]*

Output

::

  Moved    4.0 KB to the DEVICE: LvArray::ChaiBuffer<int> 
  Moved    4.0 KB to the HOST  : LvArray::ChaiBuffer<int> my_buffer
  Moved    4.0 KB to the DEVICE: double my_buffer_with_a_nonsensical_type

``LvArray::StackBuffer``
------------------------
The ``LvArray::StackBuffer`` is unique among the buffer classes because it wraps a c-array of objects whose size is fixed at compile time. It is so named because if you declare a ``LvArray::StackBuffer`` on the stack its allocation will also live on the stack. Unlike the other buffer classes by nature copying a ``LvArray::StackBuffer`` is a deep copy, furthermore a ``LvArray::StackBuffer`` can only contain trivially destructible types, so no putting a ``std::string`` in one. If you try to grow the allocation beyond the fixed size it will abort the program.

.. literalinclude:: ../../examples/exampleBuffers.cpp
  :language: c++
  :start-after: // Sphinx start after ChaiBuffer StackBuffer
  :end-before: // Sphinx end before ChaiBuffer StackBuffer

*[Source: examples/exampleBuffers.cpp]*

Doxygen
-------
- `LvArray::MallocBuffer <doxygen/html/class_lv_array_1_1_malloc_buffer.html>`_
- `LvArray::ChaiBuffer <doxygen/html/class_lv_array_1_1_chai_buffer.html>`_
- `LvArray::StackBuffer <doxygen/html/class_lv_array_1_1_stack_buffer.html>`_
