file(GLOB children RELATIVE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/*)
message("children = ${children}")


#set(dirlist "")
foreach(child ${children})
  if(IS_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${child})
    if( NOT ( ${child} STREQUAL "cmake" ) )
      blt_add_component(COMPONENT_NAME ${child} DEFAULT_STATE ON)      
    endif()
  endif()
endforeach()
                   
if (ENABLE_DOCS)
  add_subdirectory(docs)
endif()
