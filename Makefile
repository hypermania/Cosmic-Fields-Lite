##################################################################
##################################################################
# User settings: modify these for your use case
##################################################################
##################################################################
# Host compiler choice (needs support for C++20)
HOST_COMPILER ?= g++

##################################################################
# Include path & library path for FFTW

# Directory for fftw3.h
FFTW_INCLUDE_DIR := "/usr/local/include/"
#FFTW_INCLUDE_DIR := "/opt/homebrew/Cellar/fftw/3.3.10_1/include/"

# Directory for libfftw3.a
FFTW_LIBRARY_DIR := "/usr/local/lib/"
#FFTW_LIBRARY_DIR := "/opt/homebrew/lib/"

##################################################################
# CUDA related settings

# Set if CUDA should be disabled or not (CUDA is enabled by default)
# You can also disable cuda by calling "make disable-cuda=true"
disable-cuda := false

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda
# CUDA include path
CUDA_INCLUDE_DIR := "/usr/local/cuda/include/"
# CUDA library path
CUDA_LIBRARY_DIR := "/usr/local/cuda/lib64"

# Command for NVCC
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
NVCCFLAGS   := -m64 --threads 2

# Gencode arguments
# Use 86 for RTX 3060 Ti. Change this for other GPUs / CUDA Toolkit version.
SMS ?= 86 # 50 52 60 61 70 75 80 86

ifeq ($(SMS),)
	$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
	SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif

##################################################################
##################################################################
# Non-user-settings: You probably won't need to change these.
##################################################################
##################################################################
# File names and file paths for the program
program_NAME := main
src_DIR := src
program_C_SRCS := $(wildcard $(src_DIR)/*.c)
program_CXX_SRCS := $(wildcard $(src_DIR)/*.cpp)
program_H_SRCS := $(wildcard $(src_DIR)/*.h)
program_HPP_SRCS := $(wildcard $(src_DIR)/*.hpp)
program_C_OBJS := ${program_C_SRCS:.c=.o}
program_CXX_OBJS := ${program_CXX_SRCS:.cpp=.o}
program_CXX_ASMS := ${program_CXX_SRCS:.cpp=.s}

program_OBJS := $(program_C_OBJS) $(program_CXX_OBJS)
program_INCLUDE_DIRS := "external"
program_LIBRARY_DIRS :=
program_LIBRARIES := fftw3 m dl


# Names for CUDA C++ files
program_CU_SRCS := $(wildcard $(src_DIR)/*.cu) $(wildcard $(src_DIR)/*/*.cu)
program_CUH_SRCS := $(wildcard $(src_DIR)/*.cuh) $(wildcard $(src_DIR)/*/*.cuh)
program_CU_OBJS := ${program_CU_SRCS:.cu=.o}
device_link_OBJ := $(src_DIR)/device_link.o


# Option to disable CUDA
ifeq ($(disable-cuda),false)
	program_INCLUDE_DIRS += $(CUDA_INCLUDE_DIR)
	program_LIBRARY_DIRS += $(CUDA_LIBRARY_DIR)
	program_LIBRARIES += cudart cufft_static culibos
	program_OBJS += $(program_CU_OBJS) $(device_link_OBJ)
else
	CXXFLAGS += -D DISABLE_CUDA
endif


# Compiler flags
CXXFLAGS += $(foreach includedir,$(program_INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -std=c++20 -Wall -DEIGEN_NO_CUDA #-DEIGEN_NO_DEBUG
CXXFLAGS += -march=native -pthread
CXXFLAGS += -O3 -ffast-math

NVCC_OPTIMIZE_FLAGS := -use_fast_math # -Xptxas -O3,-v
NVCC_INCLUDE_DIR_FLAGS += $(foreach includedir,$(program_INCLUDE_DIRS),-I$(includedir))
NVCCFLAGS += -std=c++20 -DEIGEN_NO_CUDA
NVCCFLAGS += $(foreach library,$(program_LIBRARIES),-l$(library))


# Add linker flags
LDFLAGS += $(foreach librarydir,$(program_LIBRARY_DIRS),-L$(librarydir)) 
LDLIBS += $(foreach library,$(program_LIBRARIES),-l$(library))


.PHONY: all clean distclean

all: $(program_NAME)

$(program_NAME): $(program_OBJS)
	$(LINK.cc) $(program_OBJS) -o $(program_NAME) $(LDLIBS)

$(program_OBJS): $(program_H_SRCS) $(program_HPP_SRCS) $(program_CUH_SRCS)

%.o: %.cu
	$(NVCC) $(NVCC_INCLUDE_DIR_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) $(NVCC_OPTIMIZE_FLAGS) --diag-suppress 20012,20014 -o $@ -dc $<

$(device_link_OBJ): $(program_CU_OBJS)
	$(NVCC) $(NVCC_INCLUDE_DIR_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS)  -o $@ --device-link $(program_CU_OBJS)

%.s: %.cpp
	$(CXX) $(CXXFLAGS) -S -fverbose-asm $< -o $@

asm: $(program_CXX_ASMS)

clean:
	$(RM) $(program_NAME)
	$(RM) $(program_OBJS)
	$(RM) $(program_CXX_ASMS)
	$(RM) $(wildcard *~)
	$(RM) -r html latex

distclean: clean

show:
	echo $(CXX)
	echo $(GXX)
	echo $(GCC)
	echo $(LINK.cc)
	echo $(CC)
	echo $(CPP)
	echo $(RM)
	echo $(CXXFLAGS)
	echo $(NVCC)
	echo $(program_CXX_SRCS) "\n"
	echo $(program_HPP_SRCS) "\n"
	echo $(program_CXX_OBJS) "\n"
	echo $(program_OBJS) "\n"
	echo $(program_CU_SRCS) "\n"
	echo $(program_CUH_SRCS) "\n"
	echo $(program_CU_OBJS) "\n"
	echo $(device_link_OBJ) "\n"
	echo $(program_CXX_ASMS) "\n"
