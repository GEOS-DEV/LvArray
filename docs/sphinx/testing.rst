.. ##
.. ## Copyright (c) 2021, Lawrence Livermore National Security, LLC
.. ## and LvArray project contributors. See the LICENCE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

###############################################################################
Testing
###############################################################################

Testing is a crucial component of writing quality software and the nature LvArray lends itself nicely to unit tests.

Building and Running the Tests
------------------------------
Tests are built by default, to disable the tests set the CMake variable ``ENABLE_TESTS`` to ``OFF``. The tests are output in the ``tests`` folder of the build directory.

To run all the tests run ``make test`` in the build directory. To run a specific set of tests that match the regular expression ``REGEX`` run ``ctest -V -R REGEX``, to run just ``testCRSMatrix`` run ``./tests/testCRSMatrix``. LvArray uses `Google Test`_ for the testing framework and each test accepts a number of command line arguments.

::

    > ./tests/testCRSMatrix --help
    This program contains tests written using Google Test. You can use the
    following command line flags to control its behavior:

    Test Selection:
      --gtest_list_tests
          List the names of all tests instead of running them. The name of
          TEST(Foo, Bar) is "Foo.Bar".
      --gtest_filter=POSTIVE_PATTERNS[-NEGATIVE_PATTERNS]
          Run only the tests whose name matches one of the positive patterns but
          none of the negative patterns. '?' matches any single character; '*'
          matches any substring; ':' separates two patterns.
      --gtest_also_run_disabled_tests
          Run all disabled tests too.

    Test Execution:
      --gtest_repeat=[COUNT]
          Run the tests repeatedly; use a negative count to repeat forever.
      --gtest_shuffle
          Randomize tests' orders on every iteration.
      --gtest_random_seed=[NUMBER]
          Random number seed to use for shuffling test orders (between 1 and
          99999, or 0 to use a seed based on the current time).

    Test Output:
      --gtest_color=(yes|no|auto)
          Enable/disable colored output. The default is auto.
      --gtest_print_time=0
          Don't print the elapsed time of each test.
      --gtest_output=(json|xml)[:DIRECTORY_PATH/|:FILE_PATH]
          Generate a JSON or XML report in the given directory or with the given
          file name. FILE_PATH defaults to test_detail.xml.
      --gtest_stream_result_to=HOST:PORT
          Stream test results to the given server.

    Assertion Behavior:
      --gtest_death_test_style=(fast|threadsafe)
          Set the default death test style.
      --gtest_break_on_failure
          Turn assertion failures into debugger break-points.
      --gtest_throw_on_failure
          Turn assertion failures into C++ exceptions for use by an external
          test framework.
      --gtest_catch_exceptions=0
          Do not report exceptions as test failures. Instead, allow them
          to crash the program or throw a pop-up (on Windows).

  Except for --gtest_list_tests, you can alternatively set the corresponding
  environment variable of a flag (all letters in upper-case). For example, to
  disable colored text output, you can either specify --gtest_color=no or set
  the GTEST_COLOR environment variable to no.

  For more information, please read the Google Test documentation at
  https://github.com/google/googletest/. If you find a bug in Google Test
  (not one in your own code or tests), please report it to
  <googletestframework@googlegroups.com>.

The most useful of these is ``gtest_filter`` which lets you run a subset of tests in the file, this can be very useful when running a test through a debugger.

Test structure
--------------
The source for all the tests are all located in the ``unitTests`` directory, each tests consists of a ``cpp`` file whose name begins with ``test`` followed by the name of the class or namespace that is tested. For example the tests for ``CRSMatrix`` and ``CRSMatrixView`` are in ``unitTests/testCRSMatrix.cpp`` and the tests for ``sortedArrayManipulation`` are in ``unitTests/testSortedArrayManipulation.cpp``. 

.. note::
  The tests for ``LvArray::Array``, ``LvArray::ArrayView`` and ``LvArray::tensorOps`` are spread across multiple ``cpp`` files in order to speed up compilation on multithreaded systems.

Adding a New Test
-----------------
Any time new functionality is added it should be tested. Before writing any test code it is highly recommended you familiarize yourself with the Google Test framework, see the `Google Test primer`_ and `Google Test advanced`_ documentation.

As an example say you add a new class ``Foo``

.. literalinclude:: ../../examples/Foo.hpp
   :language: c++
   :start-after: // Sphinx start after Foo
   :end-before: // Sphinx end before Foo

*[Source: examples/Foo.hpp]*

You'll also want to create a file ``unitTests/testFoo.cpp`` and add it to ``unitTests/CMakeLists.txt``. A basic set of tests might look something like this

.. literalinclude:: ../../examples/exampleTestFoo.cpp
   :language: c++
   :start-after: // Sphinx start after Foo
   :end-before: // Sphinx end before Foo

*[Source: examples/exampleTestFoo.cpp]*

.. note::
  These tests aren't very thorough. They don't test any of the implicit constructors and operators that the compiler defines such as the copy constructor and the move assignment operator and ``get`` and ``set`` are only tested with a single value.

Now you decide you want to generalize ``Foo`` to support types other than ``int`` so you define the template class ``FooTemplate``

.. literalinclude:: ../../examples/Foo.hpp
   :language: c++
   :start-after: // Sphinx start after FooTemplate
   :end-before: // Sphinx end before FooTemplate

*[Source: examples/Foo.hpp]*

Naturally you should test more than just a single instantiation of FooTemplate so you modify your tests as such

.. literalinclude:: ../../examples/exampleTestFoo.cpp
   :language: c++
   :start-after: // Sphinx start after FooTemplate
   :end-before: // Sphinx end before FooTemplate

*[Source: examples/exampleTestFoo.cpp]*

In this example the code duplication isn't too bad because the tests are simple and only two instantiations are being tested. But using this style to write the tests for ``LvArray::Array`` which has five different template arguments would be unmaintainable. Luckily Google Test has an excellent solution: `typed tests`_. Using typed tests the tests can be restructured as

.. literalinclude:: ../../examples/exampleTestFoo.cpp
   :language: c++
   :start-after: // Sphinx start after FooTemplateTest
   :end-before: // Sphinx end before FooTemplateTest

*[Source: examples/exampleTestFoo.cpp]*

The benefits of using typed tests are many. In addition to the reduction in code duplication it makes it easy to run the tests associated with a single instantiation via ``gtest_filter`` and it lets you quickly add and remove types.  Almost every test in LvArray is built using typed tests.

.. note::
  When modifying a typed tests the compilation errors can be particularly painful to parse because usually an error in one instantiation means there will be errors in every instantiation. To decrease the verbosity you can simply limit the types used to instantiate the tests. For instance in the example above instead of testing both ``int`` and ``double`` comment out the ``, double`` and fix the ``int`` instantiation first.

One of the limitations of typed tests is that the class that gtest instantiates can only have a single template parameter and that parameter must be a type (not a value or a template). To get around this the type you pass in can be a ``std::pair`` or ``std::tuple`` when you need more than one type. For example the class ``CRSMatrixViewTest`` is defined as

.. literalinclude:: ../../unitTests/testCRSMatrix.cpp
   :language: c++
   :start-after: // Sphinx start after CRSMatrixViewTest
   :end-before: // Sphinx end before CRSMatrixViewTest

*[Source: unitTests/testCRSMatrix.cpp]*

where ``CRS_MATRIX_POLICY_PAIR`` is intended to be a ``std::pair`` where the first type is the ``CRSMatrix`` type to test and the second type is the RAJA policy to use. It is instantiated as follows

.. literalinclude:: ../../unitTests/testCRSMatrix.cpp
   :language: c++
   :start-after: // Sphinx start after CRSMatrixViewTestTypes
   :end-before: // Sphinx end before CRSMatrixViewTestTypes

*[Source: unitTests/testCRSMatrix.cpp]*

Another hurdle in writing typed tests is writing them in such a way that they compile for all the types. For example ``FooTemplate< std::string >`` is a perfectly valid instantiation but ``FooTemplateTest< std::string >`` is not because ``FooTemplate< std::string > foo( 3 )`` is invalid. You get an error like the following

::

  ../examples/exampleTestFoo.cpp:85:22: error: no matching constructor for initialization of 'FooTemplate<std::basic_string<char> >'
    FooTemplate< T > foo( 5 );
                     ^    ~

However instantiating with ``std::string`` is very important for many LvArray classes because it behaves very differently from the built in types. For that reason ``unitTests/testUtils.hpp`` defines a class ``TestString`` which wraps a ``std::string`` and ``Tensor`` which wraps a ``double[ 3 ]`` both of which have constructors from integers.

Best practices
--------------
  - Whenever possible use typed tests.
  - Whenever possible do not write CUDA (or OpenMP) specific tests. Instead write tests a typed test that is templated on the RAJA policy and use a typed test to instantiate it with the appropriate policies.
  - When linking to gtest it is not necessary to include the ``main`` function in the executable because if it is not there ``gtest`` will link in its own ``main``. However you should include ``main`` in each test file to ease debugging. Furthermore if the executable needs some setup or cleanup such as initializing MPI it should be done in main. Note that while it is certainly possible to write tests which take command line arguments it is discouraged because then ``./tests/testThatTakesCommandLineArguments`` no longer works.
  - For commonly called functions define a macro which first calls ``SCOPED_TRACE`` and then the the function. This helps illuminate exactly where errors are occurring.
  - Prefer the ``EXPECT_`` family of macros to the ``ASSERT_`` family. 
  - Use the most specific ``EXPECT_`` macro applicable. So don't do ``EXPECT_TRUE( bar() == 5 )`` instead use ``EXPECT_EQ( bar(), 5 )``

.. _`Google Test`: https://github.com/google/googletest/tree/306f3754a71d6d1ac644681d3544d06744914228
.. _`Google Test primer`: https://github.com/google/googletest/blob/306f3754a71d6d1ac644681d3544d06744914228/googletest/docs/primer.md
.. _`Google Test advanced`: https://github.com/google/googletest/blob/306f3754a71d6d1ac644681d3544d06744914228/googletest/docs/advanced.md
.. _`typed tests`: https://github.com/google/googletest/blob/306f3754a71d6d1ac644681d3544d06744914228/googletest/docs/advanced.md#typed-tests
