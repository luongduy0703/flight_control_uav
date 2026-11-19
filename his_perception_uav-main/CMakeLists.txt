# uav_perception_pi (C++): s1 detect -> s2 segment -> s3 select -> s4 fusion chay tren Pi 5 (xem CPP_RULES.md).
#   Pi:  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2   (tu nhan Pi 4 / Pi 5)
#   PC (khong ncnn / libcamera, chi test logic): cmake -S . -B build -DUAVP_WITH_NCNN=OFF -DUAVP_WITH_LIBCAMERA=OFF
cmake_minimum_required(VERSION 3.20)
project(uavp_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(UAVP_WITH_NCNN "Model NCNN (s1, s2): can cho uavp_offline, uavp_real_time" ON)
option(UAVP_WITH_LIBCAMERA "Camera Pi qua libcamera: can cho uavp_real_time" ON)
option(UAVP_BUILD_TESTS "Unit test + parity test (GTest)" ON)
# CPU dich: auto = doc /proc/device-tree/model (Pi 4 -> cortex-a72, Pi 5 -> cortex-a76); none = khong -mcpu.
set(UAVP_MCPU "auto" CACHE STRING "auto | cortex-a72 (Pi 4) | cortex-a76 (Pi 5) | none")

# Canh bao (CPP_RULES.md muc 2, 7): -Wconversion bao chuyen kieu mat du lieu, -Wswitch bao quen gia tri enum.
set(UAVP_WARNINGS -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Wswitch -Wnon-virtual-dtor)
# KHONG -ffast-math (CPP_RULES.md muc 6: giu kiem tra NaN trong fusion).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(_uavp_mcpu "${UAVP_MCPU}")
  if(_uavp_mcpu STREQUAL "auto")
    set(_uavp_mcpu "none")
    if(EXISTS "/proc/device-tree/model")
      file(READ "/proc/device-tree/model" _uavp_model)
      if(_uavp_model MATCHES "Raspberry Pi 5")
        set(_uavp_mcpu "cortex-a76")
      elseif(_uavp_model MATCHES "Raspberry Pi 4")
        set(_uavp_mcpu "cortex-a72")
      endif()
    endif()
  endif()
  if(NOT _uavp_mcpu STREQUAL "none")
    add_compile_options(-mcpu=${_uavp_mcpu})  # sai CPU (vd a76 chay tren Pi 4) -> "Illegal instruction"
  endif()
  message(STATUS "uavp: -mcpu=${_uavp_mcpu}")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT UAVP_LTO OUTPUT UAVP_LTO_MSG)
  if(UAVP_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)  # LTO: inline giua cac file .cpp
  endif()
endif()

# Moi module OpenCV co san: aruco nam trong objdetect (OpenCV >= 4.7, Pi Trixie 4.10) hoac contrib aruco (4.6)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)
# Ten target yaml-cpp: 0.8+ (Pi OS Trixie, Ubuntu 24.04) = yaml-cpp::yaml-cpp; 0.7 (Ubuntu 22.04, Bookworm) = yaml-cpp
if(TARGET yaml-cpp::yaml-cpp)
  set(UAVP_YAML_LIB yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
  set(UAVP_YAML_LIB yaml-cpp)
else()
  set(UAVP_YAML_LIB ${YAML_CPP_LIBRARIES})
endif()
include_directories(SYSTEM ${YAML_CPP_INCLUDE_DIR})  # yaml-cpp cu: target khong mang duong dan include

# --- Thu vien loi: khong phu thuoc ncnn / libcamera (test duoc tren PC) ---------------------------------------
# Moi stage mot thu muc (src/stages/s1..s4); tracking: khoa + bam vat sau s3 (tu the FC, optical flow, Kalman);
# common: config, log, box, file ket qua; sources: LD06, MAVLink (Pixhawk), replay; support: ArUco, ve anh,
# theo doi phan cung, phim terminal, gioi han nhan CPU.
add_library(uavp_core STATIC
  src/common/config.cpp src/common/log.cpp src/common/boxes.cpp src/common/run_files.cpp src/common/report.cpp
  src/common/tensor_io.cpp
  src/stages/s1_detect/letterbox.cpp src/stages/s1_detect/tiling.cpp src/stages/s1_detect/detector.cpp
  src/stages/s2_segment/mask_shape.cpp src/stages/s2_segment/segmenter.cpp
  src/stages/s3_select/tracker.cpp src/stages/s3_select/measure.cpp src/stages/s3_select/decide.cpp
  src/stages/s3_select/rank.cpp
  src/stages/s4_fusion/lidar.cpp src/stages/s4_fusion/plane.cpp src/stages/s4_fusion/fusion.cpp
  src/tracking/world_pose.cpp src/tracking/roi_kalman.cpp src/tracking/roi_flow.cpp src/tracking/locked_tracker.cpp
  src/pipeline/session.cpp src/pipeline/start_gate.cpp
  src/sources/serial_port.cpp src/sources/ld06.cpp src/sources/ld06_buffer.cpp src/sources/mavlink.cpp
  src/sources/fc_shm.cpp
  src/sources/replay_source.cpp
  src/support/aruco_check.cpp src/support/draw.cpp src/support/hw_monitor.cpp src/support/hw_monitor_thread.cpp
  src/support/terminal_keys.cpp src/support/cpu_affinity.cpp src/support/remote_view.cpp)
target_include_directories(uavp_core PUBLIC include)
target_include_directories(uavp_core SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})  # canh bao cua OpenCV khong tinh
target_compile_options(uavp_core PRIVATE ${UAVP_WARNINGS})
target_link_libraries(uavp_core PUBLIC ${OpenCV_LIBS} ${UAVP_YAML_LIB} Threads::Threads)

# --- Bridge Pixhawk -> RAM (khong can ncnn / libcamera): dich vu systemd uavp-fc-bridge ---------------------
add_executable(uavp_fc_bridge apps/fc_bridge.cpp)
target_compile_options(uavp_fc_bridge PRIVATE ${UAVP_WARNINGS})
target_link_libraries(uavp_fc_bridge PRIVATE uavp_core)

add_executable(uavp_fc_check apps/fc_check.cpp)  # kiem tra Pixhawk -> Pi (tan so tin, do cao, lenh may tinh)
target_compile_options(uavp_fc_check PRIVATE ${UAVP_WARNINGS})
target_link_libraries(uavp_fc_check PRIVATE uavp_core)

# --- Model NCNN -------------------------------------------------------------------------------------------------
if(UAVP_WITH_NCNN)
  find_package(ncnn REQUIRED)  # Pi: build ncnn tu source (GUIDE_CPP.md), -Dncnn_DIR=<prefix>/lib/cmake/ncnn
  add_library(uavp_ncnn STATIC src/model/ncnn_model.cpp)
  target_compile_options(uavp_ncnn PRIVATE ${UAVP_WARNINGS})
  target_link_libraries(uavp_ncnn PUBLIC uavp_core ncnn)

  add_executable(uavp_offline apps/offline.cpp)
  target_compile_options(uavp_offline PRIVATE ${UAVP_WARNINGS})
  target_link_libraries(uavp_offline PRIVATE uavp_ncnn)

  add_executable(uavp_bench_model apps/bench_model.cpp)  # do s1 / s2 tren Pi, fp16 storage bat / tat
  target_compile_options(uavp_bench_model PRIVATE ${UAVP_WARNINGS})
  target_link_libraries(uavp_bench_model PRIVATE uavp_ncnn)
endif()

# --- Camera libcamera + ung dung thoi gian thuc --------------------------------------------------------------
if(UAVP_WITH_LIBCAMERA)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBCAMERA REQUIRED IMPORTED_TARGET libcamera)
  add_library(uavp_camera STATIC src/sources/pi_camera.cpp)
  target_compile_options(uavp_camera PRIVATE ${UAVP_WARNINGS})
  target_link_libraries(uavp_camera PUBLIC uavp_core PkgConfig::LIBCAMERA)

  add_executable(uavp_sensor_check apps/sensor_check.cpp)  # test nhanh camera + LiDAR + Pixhawk (khong model)
  target_compile_options(uavp_sensor_check PRIVATE ${UAVP_WARNINGS})
  target_link_libraries(uavp_sensor_check PRIVATE uavp_camera)
endif()
if(UAVP_WITH_NCNN AND UAVP_WITH_LIBCAMERA)
  # 2 chuong trinh dung chung apps/live_app.cpp: chi 4 stage (do thoi gian / sai so) | ca chuong trinh + Pixhawk
  add_library(uavp_live STATIC apps/live_app.cpp)
  target_compile_options(uavp_live PRIVATE ${UAVP_WARNINGS})
  target_link_libraries(uavp_live PUBLIC uavp_ncnn uavp_camera)
  foreach(app real_time real_time_perception)
    add_executable(uavp_${app} apps/${app}.cpp)
    target_compile_options(uavp_${app} PRIVATE ${UAVP_WARNINGS})
    target_link_libraries(uavp_${app} PRIVATE uavp_live)
  endforeach()
endif()

# --- Test ------------------------------------------------------------------------------------------------------
if(UAVP_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_executable(uavp_tests
    tests/test_boxes.cpp tests/test_detect.cpp tests/test_segment.cpp tests/test_select.cpp
    tests/test_fusion.cpp tests/test_ld06.cpp tests/test_hw_monitor.cpp tests/test_config.cpp
    tests/test_mavlink.cpp tests/test_tracking.cpp tests/test_remote.cpp tests/test_start.cpp tests/test_parity.cpp)
  target_compile_options(uavp_tests PRIVATE ${UAVP_WARNINGS})
  target_link_libraries(uavp_tests PRIVATE uavp_core GTest::gtest_main)
  # thu muc goc (configs/, models/) va du lieu parity (tools/dump_parity.py)
  target_compile_definitions(uavp_tests PRIVATE UAVP_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
                                               UAVP_PARITY_DIR="${CMAKE_CURRENT_SOURCE_DIR}/parity_data")
  include(GoogleTest)
  gtest_discover_tests(uavp_tests)
endif()
