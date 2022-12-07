set( pythonTests
     testPyArray.cpp
     testPySortedArray.cpp
     testPyCRSMatrix.cpp
     testPythonScalars.cpp
     testPythonListOfStrings.cpp
     testPyCallback.cpp
     testPyArrayOfArrays.cpp
     testPyArrayOfSets.cpp
    )

foreach(test ${pythonTests})
    get_filename_component( test_name ${test} NAME_WE )

    blt_add_library( NAME ${test_name}
                     SOURCES ${test}
                     DEPENDS_ON pylvarray ${lvarray_dependencies}
                     SHARED TRUE
                     CLEAR_PREFIX TRUE )

    target_include_directories( ${test_name} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../src )


    configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/${test_name}Driver.py ${CMAKE_CURRENT_BINARY_DIR}/${test_name}Driver.py COPYONLY )

    add_test( NAME ${test_name}
              COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/${test_name}Driver.py )

    set_tests_properties( ${test_name}
                          PROPERTIES ENVIRONMENT PYTHONPATH=${CMAKE_BINARY_DIR}/lib )

endforeach()
