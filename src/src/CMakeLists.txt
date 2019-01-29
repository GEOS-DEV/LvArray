set(cxx-utilities_headers
    Array.hpp
    ArraySlice.hpp
    ArrayUtilities.hpp
    ArrayView.hpp
    ChaiVector.hpp
    helpers.hpp
    IntegerConversion.hpp
    Logger.hpp
    Macros.hpp
    ObjectCatalog.hpp 
    SetFPE.hpp
    SetSignalHandling.hpp
    SFINAE_Macros.hpp
    SortedArray.hpp
    StackArrayWrapper.hpp
    stackTrace.hpp
    StringUtilities.hpp )

set(cxx-utilities_sources
    Logger.cpp
    SetFPE.cpp
    SetSignalHandling.cpp
    stackTrace.cpp
    StringUtilities.cpp )


message("adding cxx-utilities library")
blt_add_library( NAME             cxx-utilities
                 SOURCES          ${cxx-utilities_sources}
                 HEADERS          ${cxx-utilities_headers}
                 HEADERS_OUTPUT_SUBDIR cxx-utilities
                 DEPENDS_ON       ${cxx_dependencies}
                 )



target_include_directories( cxx-utilities PUBLIC ${CMAKE_CURRENT_LIST_DIR} )
target_include_directories( cxx-utilities PUBLIC ${CMAKE_BINARY_DIR}/include )

cxxutils_add_code_checks(PREFIX cxx-utilities )

