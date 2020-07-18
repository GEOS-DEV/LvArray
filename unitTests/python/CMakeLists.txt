set( pythonTests
     testPythonArray.cpp
     testPythonSortedArray.cpp
    )

foreach(test ${pythonTests})
    get_filename_component( test_name ${test} NAME_WE )

    blt_add_library( NAME ${test_name}
                     SOURCES ${test} ${test_name}.c
                     HEADERS ${test_name}.h
                     DEPENDS_ON lvarray Python3::Python Python3::NumPy
                     SHARED TRUE
                     CLEAR_PREFIX TRUE
                    )

    target_include_directories( ${test_name} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../src )

    add_test( NAME ${test_name}
              COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/${test_name}Driver.py )

    set_tests_properties( ${test_name}
                          PROPERTIES ENVIRONMENT PYTHONPATH=${CMAKE_BINARY_DIR}/lib )

endforeach()
