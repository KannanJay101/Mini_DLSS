# Makefile for Mini-DLSS CUDA kernels
# Builds each .cu file into a shared library callable from Python via ctypes.
# Usage:
#   make all          - build all three shared libraries
#   make psnr         - build psnr_kernel only
#   make warp         - build warp_kernel only
#   make blend        - build blend_kernel only
#   make clean        - remove build artifacts
#   make check        - verify nvcc and GPU are available

NVCC      := nvcc
NVCCFLAGS := -O2 -Xcompiler -fPIC --use_fast_math

# Detect OS: Windows uses .dll, Linux/macOS uses .so
ifeq ($(OS),Windows_NT)
    LIB_EXT     := dll
    SHARED_FLAG := -shared
    RM          := del /Q
    MKDIR       := mkdir
else
    LIB_EXT     := so
    SHARED_FLAG := -shared
    RM          := rm -f
    MKDIR       := mkdir -p
endif

# Output directory
BUILD_DIR := build

# Targets
PSNR_LIB  := $(BUILD_DIR)/psnr_kernel.$(LIB_EXT)
WARP_LIB  := $(BUILD_DIR)/warp_kernel.$(LIB_EXT)
BLEND_LIB := $(BUILD_DIR)/blend_kernel.$(LIB_EXT)

ALL_LIBS  := $(PSNR_LIB) $(WARP_LIB) $(BLEND_LIB)

# --------------------------------------------------------------------------- #

.PHONY: all psnr warp blend clean check

all: $(BUILD_DIR) $(ALL_LIBS)
	@echo "All kernels built successfully in $(BUILD_DIR)/"

psnr: $(BUILD_DIR) $(PSNR_LIB)
warp: $(BUILD_DIR) $(WARP_LIB)
blend: $(BUILD_DIR) $(BLEND_LIB)

$(BUILD_DIR):
ifeq ($(OS),Windows_NT)
	if not exist $(BUILD_DIR) $(MKDIR) $(BUILD_DIR)
else
	$(MKDIR) $(BUILD_DIR)
endif

# psnr_kernel.cu -> shared library
# Exports: compute_psnr_cuda()
$(PSNR_LIB): psnr_kernel.cu
	$(NVCC) $(NVCCFLAGS) $(SHARED_FLAG) -o $@ $<
	@echo "Built: $@"

# warp_kernel.cu -> shared library
# Exports: warp_kernel (device function, linked via pycuda/ctypes host wrapper)
$(WARP_LIB): warp_kernel.cu
	$(NVCC) $(NVCCFLAGS) $(SHARED_FLAG) -o $@ $<
	@echo "Built: $@"

# blend_kernel.cu -> shared library
# Exports: blend_kernel (device function)
$(BLEND_LIB): blend_kernel.cu
	$(NVCC) $(NVCCFLAGS) $(SHARED_FLAG) -o $@ $<
	@echo "Built: $@"

# --------------------------------------------------------------------------- #

clean:
ifeq ($(OS),Windows_NT)
	if exist $(BUILD_DIR) rmdir /S /Q $(BUILD_DIR)
else
	$(RM) $(ALL_LIBS)
	rmdir -p $(BUILD_DIR) 2>/dev/null || true
endif
	@echo "Cleaned build artifacts."

check:
	@echo "--- nvcc version ---"
	@$(NVCC) --version
	@echo "--- GPU info ---"
	@nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
	@echo "CUDA environment OK."
