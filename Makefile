# 编译器定义
HIPCC      := hipcc
NVCC       := nvcc

# 通用编译选项
COMMON_FLAGS := -pthread -fopenmp -Wno-deprecated -w -ffunction-sections -fdata-sections -fmodulo-sched

# GPU库链接参数
HIP_LIBS   := -lhipblas
CUDA_LIBS  := -lcublas -lcudart

# 源文件定义
SRC_HIP    := src/class_func_hip.cpp
SRC_CUDA   := src/class_func.cpp

# 可执行文件路径
EXE_CUDA   := bin/MGFunc
EXE_HIP    := bin/MGFunc_hip

# 声明伪目标
.PHONY: all hip clean

# 默认目标：编译 CUDA 版本
all:
	mkdir -p bin
	$(NVCC) $(SRC_CUDA) -o $(EXE_CUDA) -Xcompiler "$(COMMON_FLAGS)" $(CUDA_LIBS)

# HIP编译目标
hip:
	mkdir -p bin
	$(HIPCC) $(SRC_HIP) -o $(EXE_HIP) $(COMMON_FLAGS) $(HIP_LIBS)

# 清理目标
clean:
	rm -rf $(EXE_CUDA) $(EXE_HIP)
