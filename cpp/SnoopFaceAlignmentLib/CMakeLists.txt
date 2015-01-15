cmake_minimum_required(VERSION 2.8)

get_project_name_from_path(project_name ${CMAKE_CURRENT_SOURCE_DIR})

Project(${project_name})

include_directories(${lib})

set(${project_name}_VERSION_MAJOR 1)
set(${project_name}_VERSION_MINOR 0)
set(${project_name}_VERSION_PATCHLEVEL 0)

get_version(${project_name} ${CMAKE_CURRENT_SOURCE_DIR})


set(SOURCEFILES 
        SDM.cc
        Hog.cc
        ForestBasedRegression.cc
        Alignment.cc
 )
 
 
import_static_lib(opencv_calib3d "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(opencv_contrib "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(opencv_core "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(opencv_features2d "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(opencv_flann "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(opencv_gpu "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(opencv_highgui "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(opencv_imgproc "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(opencv_legacy "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(opencv_ml "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(opencv_nonfree "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(opencv_objdetect "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(opencv_photo "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(opencv_stitching "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(opencv_superres "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(opencv_ts "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(opencv_video "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(opencv_videostab "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(swresample "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(swscale "${CMAKE_INSTALL_PREFIX}/lib")

import_static_lib(libjasper "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(IlmImf "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(libpng "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(libtiff "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(libjpeg "${CMAKE_INSTALL_PREFIX}/lib")

import_static_lib(avcodec "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(avformat "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(avutil "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(avdevice "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(avfilter "${CMAKE_INSTALL_PREFIX}/lib")

#import_shared_lib(rexparameters "${CMAKE_INSTALL_PREFIX}/lib")
#import_shared_lib(log4cpp "${CMAKE_INSTALL_PREFIX}/lib")
#import_shared_lib(rexversion "${CMAKE_INSTALL_PREFIX}/lib")
import_shared_lib(clmTracker "${CMAKE_INSTALL_PREFIX}/lib")

import_static_lib(bz2 "${CMAKE_INSTALL_PREFIX}/lib")

import_static_lib(rexversion "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(rextools "${CMAKE_INSTALL_PREFIX}/lib")
import_static_lib(rexerrors "${CMAKE_INSTALL_PREFIX}/lib")


set(LINKED_LIBS rextoolsStatic rexversionStatic rexerrorsStatic boost_random opencv_coreStatic opencv_highguiStatic opencv_imgprocStatic swscaleStatic libjasperStatic IlmImfStatic libpngStatic libtiffStatic libjpegStatic avcodecStatic avutilStatic avformatStatic avdeviceStatic avfilterStatic pthread z clmTracker bz2Static)

#IF(${USE_GTK})
    set(LINKED_LIBS ${LINKED_LIBS} gobject-2.0 gtk-x11-2.0 gnome-2 glib-2.0 gdk-x11-2.0)
#ENDIF()


add_my_library(STATIC INSTALL TARGET ${project_name} 
                SOURCES ${SOURCEFILES}  
                LIBS -Wl,--start-group  ${LINKED_LIBS}  -Wl,--end-group
                HEADERS SnoopFaceDesc.h)