cmake_minimum_required(VERSION 3.13.4)

project(FEniCS_demo)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/FEniCS_demo")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Ben Zwick (UWA)")
set(EXTENSION_DESCRIPTION "This is an example of a FEniCS extension")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/FEniCS_demo.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/FEniCS_demo/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(FEniCS_demo_module)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
