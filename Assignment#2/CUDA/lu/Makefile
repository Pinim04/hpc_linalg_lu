ifndef CUDA_HOME
CUDA_HOME := /usr/local/cuda
endif

ifndef EXERCISE
EXERCISE := lu.cu
endif

UTIL_DIR  ?= ../../../utilities
BUILD_DIR ?= ./build

NVCC := $(CUDA_HOME)/bin/nvcc
CXX  := g++

OPT   := -O2 -g
NVOPT := -arch=sm_53 -Xcompiler -fopenmp -lineinfo --ptxas-options=-v

CFLAGS    := $(OPT) -I. -I$(UTIL_DIR) $(EXT_CFLAGS) $(EXT_CXXFLAGS)
CXXFLAGS  := $(CFLAGS)
LDFLAGS   := -lm -lcudart $(EXT_LDFLAGS)

NVCFLAGS  := $(CXXFLAGS) $(NVOPT)
NVLDFLAGS := $(LDFLAGS) -lgomp

SRCS := utils.c
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o) $(BUILD_DIR)/polybench.c.o $(EXERCISE:%=$(BUILD_DIR)/%.o)
EXE  := $(EXERCISE:.cu=.exe)

$(EXE): $(OBJS)
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) $(OBJS) -o $@ $(NVLDFLAGS)

$(BUILD_DIR)/polybench.c.o: $(UTIL_DIR)/polybench.c
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) -Dnew=pb_new -c $< -o $@

$(BUILD_DIR)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@
all: $(EXE)

.PHONY: run profile metrics clean
run: $(EXE)
	./$(EXE)

profile: $(EXE)
	sudo $(CUDA_HOME)/bin/nvprof --unified-memory-profiling off ./$(EXE)

metrics: $(EXE)
	sudo $(CUDA_HOME)/bin/nvprof --unified-memory-profiling off --print-gpu-trace --metrics "eligible_warps_per_cycle,achieved_occupancy,sm_efficiency,ipc" ./$(EXE)

clean:
	-rm -fr $(BUILD_DIR) *.exe *.out *~

MKDIR_P ?= mkdir -p