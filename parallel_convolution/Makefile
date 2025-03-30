# Compiler settings
CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 
LDFLAGS = -lm
CUDAFLAGS = -O3 -arch=sm_61  # Adjust for your GPU architecture
OMPFLAGS = -O3 -fopenmp
OMPGPUFLAGS = -O3 -fopenmp -foffload=nvptx-none -foffload="-march=sm_53"  # Adjust for your GPU architecture
PGXX = pgc++
PGFLAGS = -O3 -acc -fast -gpu=cc61 -Minfo=accel

# Executable names
SERIAL_EXE = convolution_serial
OMP_CPU_EXE = convolution_omp_cpu
OMP_GPU_EXE = convolution_omp_gpu
CUDA_EXE = convolution_cuda
OPENACC_EXE = convolution_acc

# Source files
SERIAL_SRC = convolution_serial.cpp
OMP_CPU_SRC = convolution_omp_cpu.cpp
OMP_GPU_SRC = convolution_omp_gpu.cpp
CUDA_SRC = convolution_cuda.cu
OPENACC_SRC = convolution_acc.cpp

# All target
all: serial ompcpu ompgpu cuda acc

# Compile the SERIAL version
serial: $(SERIAL_SRC)
	$(CXX) $(CXXFLAGS) $(SERIAL_SRC) -o $(SERIAL_EXE) $(LDFLAGS)

# Compile the OpenMP CPU version
ompcpu: $(OMP_CPU_SRC)
	$(CXX) $(OMPFLAGS) $(OMP_CPU_SRC) -o $(OMP_CPU_EXE) $(LDFLAGS)

# Compile the OpenMP GPU version
ompgpu: $(OMP_GPU_SRC)
	$(CXX) $(OMPGPUFLAGS) $(OMP_GPU_SRC) -o $(OMP_GPU_EXE) $(LDFLAGS)

# Compile the CUDA version
cuda: $(CUDA_SRC)
	$(NVCC) $(CUDAFLAGS) $(CUDA_SRC) -o $(CUDA_EXE) $(LDFLAGS)

# Compile the OpenACC version
acc: $(OPENACC_SRC)
	$(PGXX) $(PGFLAGS) $(OPENACC_SRC) -o $(OPENACC_EXE) $(LDFLAGS)

# Clean up build files
clean:
	rm -f $(SERIAL_EXE) $(OMP_CPU_EXE) $(OMP_GPU_EXE) $(CUDA_EXE) $(OPENACC_EXE)
