#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "httplib::httplib" for configuration "Release"
set_property(TARGET httplib::httplib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(httplib::httplib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcpp-httplib.a"
  )

list(APPEND _cmake_import_check_targets httplib::httplib )
list(APPEND _cmake_import_check_files_for_httplib::httplib "${_IMPORT_PREFIX}/lib/libcpp-httplib.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
