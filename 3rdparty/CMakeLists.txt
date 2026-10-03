set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(cpuinfo)

# The Windows guest red-zone patcher needs a decoder/encoder and a small
# trampoline assembler. Pin the same revisions used by shadPS4's implementation.
include(FetchContent)

set(XBYAK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
	xbyak
	GIT_REPOSITORY https://github.com/herumi/xbyak.git
	GIT_TAG 44a72f369268f7d552650891b296693e91db86bb
	GIT_SHALLOW FALSE
)

set(ZYDIS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_DOXYGEN OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
	zydis
	GIT_REPOSITORY https://github.com/zyantific/zydis.git
	GIT_TAG 120e0e705f8e3b507dc49377ac2879979f0d545c
	GIT_SUBMODULES_RECURSE TRUE
	GIT_SHALLOW FALSE
)

FetchContent_MakeAvailable(xbyak zydis)

# ZArchive provides random-access reads for compressed game dumps. Build only
# the reader/writer library and its zstd dependency; the command-line tool is
# not part of KytyPS5's runtime distribution.
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZSTD_LEGACY_SUPPORT OFF CACHE BOOL "" FORCE)
set(ZSTD_MULTITHREAD_SUPPORT OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
	zstd
	GIT_REPOSITORY https://github.com/facebook/zstd.git
	GIT_TAG f8745da6ff1ad1e7bab384bd1f9d742439278e99
	GIT_SHALLOW FALSE
	SOURCE_SUBDIR build/cmake
)
FetchContent_Declare(
	zarchive_source
	GIT_REPOSITORY https://github.com/Exzap/ZArchive.git
	GIT_TAG d2c717730092c7bf8cbb033b12fd4001b7c4d932
	GIT_SHALLOW FALSE
	PATCH_COMMAND "${GIT_EXECUTABLE}" apply --ignore-whitespace
		"${CMAKE_CURRENT_SOURCE_DIR}/patches/zarchive-reader.patch"
	SOURCE_SUBDIR kyty-library-only
)
FetchContent_MakeAvailable(zstd zarchive_source)

add_library(zarchive STATIC
	"${zarchive_source_SOURCE_DIR}/src/zarchivereader.cpp"
	"${zarchive_source_SOURCE_DIR}/src/zarchivewriter.cpp"
	"${zarchive_source_SOURCE_DIR}/src/sha_256.c"
)
add_library(ZArchive::zarchive ALIAS zarchive)
target_include_directories(zarchive SYSTEM PUBLIC "${zarchive_source_SOURCE_DIR}/include")
target_include_directories(zarchive PRIVATE "${zstd_SOURCE_DIR}/lib")
target_link_libraries(zarchive PRIVATE libzstd_static)
target_compile_features(zarchive PRIVATE cxx_std_20)
if(MSVC)
	target_compile_options(zarchive PRIVATE /W0)
else()
	target_compile_options(zarchive PRIVATE -w)
endif()

set(SDL_SHARED OFF CACHE BOOL "" FORCE)
if(APPLE)
	# SDL's built-in iconv fallback does not support CP932.
	set(SDL_SYSTEM_ICONV ON CACHE BOOL "Use system iconv for CP932 conversion" FORCE)
endif()
add_subdirectory(SDL3)

add_library(Vulkan-Headers INTERFACE)
add_library(Vulkan::Headers ALIAS Vulkan-Headers)
target_include_directories(Vulkan-Headers INTERFACE
	"${CMAKE_CURRENT_SOURCE_DIR}/Vulkan-Headers/include"
)

add_library(imgui STATIC
	imgui/imgui.cpp
	imgui/imgui_draw.cpp
	imgui/imgui_tables.cpp
	imgui/imgui_widgets.cpp
	imgui/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/imgui"
	"${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends"
)
target_compile_definitions(imgui PRIVATE IMGUI_IMPL_VULKAN_NO_PROTOTYPES)
target_link_libraries(imgui PUBLIC Vulkan::Headers)
if(MSVC)
	target_compile_options(imgui PRIVATE /W0)
else()
	target_compile_options(imgui PRIVATE -w)
endif()

set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "" FORCE)
set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
set(SPIRV-Headers_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SPIRV-Headers" CACHE PATH "" FORCE)
add_subdirectory(SPIRV-Tools)

add_library(spirv-tools ALIAS SPIRV-Tools-static)
add_library(spirv-tools-opt ALIAS SPIRV-Tools-opt)

add_subdirectory(xxHash/build/cmake)

set(FMT_DOC OFF CACHE BOOL "" FORCE)
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
set(FMT_MODULE OFF CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory(fmt EXCLUDE_FROM_ALL)

set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
add_subdirectory(nlohmann_json EXCLUDE_FROM_ALL)

set(SPDLOG_BUILD_ALL OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_WARNINGS OFF CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
set(SPDLOG_NO_EXCEPTIONS ON CACHE BOOL "" FORCE)
if(WIN32)
	set(SPDLOG_WCHAR_FILENAMES ON CACHE BOOL "" FORCE)
endif()
add_subdirectory(spdlog EXCLUDE_FROM_ALL)

if (NOT TARGET FFmpeg::ffmpeg)
	# On macOS the target arch is driven by CMAKE_OSX_ARCHITECTURES, not the host
	# CMAKE_SYSTEM_PROCESSOR (which reports arm64 even for an x86_64/Rosetta build).
	if(APPLE AND CMAKE_OSX_ARCHITECTURES)
		if(CMAKE_OSX_ARCHITECTURES MATCHES "^(x86_64)$")
			set(ARCHITECTURE x86_64)
		elseif(CMAKE_OSX_ARCHITECTURES MATCHES "^(arm64)$")
			set(ARCHITECTURE arm64)
		else()
			set(ARCHITECTURE x86_64 arm64) # universal
		endif()
	elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
		set(ARCHITECTURE arm64)
	else()
		set(ARCHITECTURE x86_64)
	endif()
	# Pre-fetch the FFmpeg prebuilt archive with retries and a fallback release.
	# The ffmpeg-core submodule downloads the same URL with a single file(DOWNLOAD)
	# attempt, so any transient HTTP failure (curl code 22 from
	# release-assets.githubusercontent.com) aborts the whole configure step.
	# The canonical host stays KytyPS5/ext-ffmpeg-core: neither Inori/Kyty nor
	# Senaxx/KytyPS5 publish these ffmpeg-*.zip assets, so switching the host
	# would turn a transient failure into a permanent 404. The fallback below is
	# therefore the previous known-good ext-ffmpeg-core release tag, not another
	# host. A successful pre-fetch (hash-verified against that release's
	# SHA256SUMS, i.e. the EXPECTED_HASH equivalent) makes the submodule skip its
	# own download: the submodule keys both the zip path and the extract dir off
	# its own HEAD, so fallback content intentionally lands under the pinned
	# name; headers and libs stay consistent because both come from the same
	# verified zip. A failed pre-fetch only warns and leaves the submodule
	# behaviour unchanged.
	if(WIN32 AND NOT MINGW)
		set(KYTY_FFMPEG_OS_PART "ffmpeg-windows")
	elseif(APPLE)
		set(KYTY_FFMPEG_OS_PART "ffmpeg-macos")
	elseif(UNIX)
		set(KYTY_FFMPEG_OS_PART "ffmpeg-linux")
	elseif(MINGW)
		set(KYTY_FFMPEG_OS_PART "ffmpeg-mingw")
	endif()
	if(ARCHITECTURE STREQUAL "x86_64")
		set(KYTY_FFMPEG_ARCH_PART "x64.zip")
	elseif(ARCHITECTURE STREQUAL "arm64")
		set(KYTY_FFMPEG_ARCH_PART "arm64.zip")
	else() # macOS universal
		set(KYTY_FFMPEG_ARCH_PART "universal.zip")
	endif()
	set(KYTY_FFMPEG_NAME "${KYTY_FFMPEG_OS_PART}-${KYTY_FFMPEG_ARCH_PART}")
	execute_process(COMMAND git rev-parse --short=12 HEAD
		WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg-core"
		OUTPUT_VARIABLE KYTY_FFMPEG_SHA OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
	if(NOT KYTY_FFMPEG_SHA MATCHES "^[0-9a-f]+$")
		set(KYTY_FFMPEG_SHA "9ac4cfd195f1")
	endif()
	# Previous known-good ffmpeg-core release, used only if the pinned one 404s.
	set(KYTY_FFMPEG_FALLBACK_SHA "666183f682cb")
	set(KYTY_FFMPEG_BASE_URL "https://github.com/KytyPS5/ext-ffmpeg-core/releases/download")
	set(KYTY_FFMPEG_ZIP "${CMAKE_BINARY_DIR}/externals/${KYTY_FFMPEG_SHA}-${KYTY_FFMPEG_NAME}")
	if(NOT FFMPEG_PREBUILT_DIR AND NOT EXISTS "${KYTY_FFMPEG_ZIP}")
		file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/externals")
		set(KYTY_FFMPEG_URLS
			"${KYTY_FFMPEG_BASE_URL}/${KYTY_FFMPEG_SHA}/${KYTY_FFMPEG_NAME}"
			"${KYTY_FFMPEG_BASE_URL}/${KYTY_FFMPEG_FALLBACK_SHA}/${KYTY_FFMPEG_NAME}")
		set(KYTY_FFMPEG_USED_SHA "")
		foreach(KYTY_FFMPEG_URL IN LISTS KYTY_FFMPEG_URLS)
			if(KYTY_FFMPEG_USED_SHA)
				break()
			endif()
			message(STATUS "Downloading FFmpeg libraries (${KYTY_FFMPEG_URL})...")
			foreach(KYTY_FFMPEG_ATTEMPT RANGE 1 5)
				file(DOWNLOAD "${KYTY_FFMPEG_URL}" "${KYTY_FFMPEG_ZIP}"
					SHOW_PROGRESS TIMEOUT 300 STATUS KYTY_FFMPEG_STATUS)
				list(GET KYTY_FFMPEG_STATUS 0 KYTY_FFMPEG_CODE)
				if(KYTY_FFMPEG_CODE EQUAL 0)
					if(KYTY_FFMPEG_URL MATCHES "/([^/]+)/[^/]+$")
						set(KYTY_FFMPEG_USED_SHA "${CMAKE_MATCH_1}")
					endif()
					break()
				endif()
				file(REMOVE "${KYTY_FFMPEG_ZIP}")
				if(KYTY_FFMPEG_ATTEMPT LESS 5)
					message(STATUS "FFmpeg download attempt ${KYTY_FFMPEG_ATTEMPT} failed (${KYTY_FFMPEG_STATUS}), retrying...")
					execute_process(COMMAND ${CMAKE_COMMAND} -E sleep 5)
				else()
					message(STATUS "FFmpeg download failed (${KYTY_FFMPEG_STATUS}), trying next URL...")
				endif()
			endforeach()
		endforeach()
		if(KYTY_FFMPEG_USED_SHA)
			set(KYTY_FFMPEG_SUMS_FILE "${CMAKE_BINARY_DIR}/externals/${KYTY_FFMPEG_USED_SHA}-SHA256SUMS")
			if(NOT EXISTS "${KYTY_FFMPEG_SUMS_FILE}")
				file(DOWNLOAD "${KYTY_FFMPEG_BASE_URL}/${KYTY_FFMPEG_USED_SHA}/SHA256SUMS"
					"${KYTY_FFMPEG_SUMS_FILE}" TIMEOUT 120 STATUS KYTY_FFMPEG_SUMS_STATUS)
				list(GET KYTY_FFMPEG_SUMS_STATUS 0 KYTY_FFMPEG_SUMS_CODE)
				if(NOT KYTY_FFMPEG_SUMS_CODE EQUAL 0)
					file(REMOVE "${KYTY_FFMPEG_SUMS_FILE}")
				endif()
			endif()
			if(EXISTS "${KYTY_FFMPEG_SUMS_FILE}")
				file(STRINGS "${KYTY_FFMPEG_SUMS_FILE}" KYTY_FFMPEG_SUM_LINE REGEX "${KYTY_FFMPEG_NAME}$")
				string(REGEX MATCH "^[0-9a-fA-F]+" KYTY_FFMPEG_EXPECTED "${KYTY_FFMPEG_SUM_LINE}")
				if(KYTY_FFMPEG_EXPECTED)
					file(SHA256 "${KYTY_FFMPEG_ZIP}" KYTY_FFMPEG_ACTUAL)
					if(NOT KYTY_FFMPEG_ACTUAL STREQUAL KYTY_FFMPEG_EXPECTED)
						file(REMOVE "${KYTY_FFMPEG_ZIP}")
						set(KYTY_FFMPEG_USED_SHA "")
						message(WARNING "FFmpeg archive hash mismatch, removed it; ffmpeg-core will retry the download")
					endif()
				endif()
			endif()
		endif()
		if(NOT KYTY_FFMPEG_USED_SHA AND NOT EXISTS "${KYTY_FFMPEG_ZIP}")
			message(WARNING "FFmpeg pre-fetch failed; falling back to the ffmpeg-core submodule download")
		endif()
	endif()
	add_subdirectory(ffmpeg-core)
	add_library(FFmpeg::ffmpeg ALIAS ffmpeg)
endif()

file(GLOB LIBATRAC9_SOURCES LibAtrac9/C/src/*.c)
add_library(LibAtrac9 STATIC ${LIBATRAC9_SOURCES})
target_include_directories(LibAtrac9 PUBLIC LibAtrac9/C/src)
if(MSVC)
	target_compile_options(LibAtrac9 PRIVATE /W0)
endif()

set(TRACY_STATIC ON CACHE BOOL "" FORCE)
set(TRACY_ENABLE ON CACHE BOOL "" FORCE)
set(TRACY_NO_CRASH_HANDLER ON CACHE BOOL "" FORCE)
set(TRACY_ON_DEMAND ON CACHE BOOL "" FORCE)
set(TRACY_DELAYED_INIT ON CACHE BOOL "" FORCE)
set(TRACY_MANUAL_LIFETIME ON CACHE BOOL "" FORCE)
add_subdirectory(tracy EXCLUDE_FROM_ALL)
if (TARGET TracyClient)
	if (MSVC)
		target_compile_options(TracyClient PRIVATE /WX-)
	endif()
	if (CLANG)
		target_compile_options(TracyClient PRIVATE
			-Wno-error
			-Wno-braced-scalar-init
			-Wno-format
			-Wno-unused-variable
		)
	endif()
endif()

if (CLANG AND MSVC)
add_subdirectory(winpthread)
endif()
