NVFLAGS := -std=c++11 -O3 
TARGET_GPU := hw4_gpu

.PHONY: all
all: $(TARGET_GPU)

$(TARGET_GPU): sha256.o hw4_gpu.cu
	nvcc $(NVFLAGS) -o hw4_gpu hw4_gpu.cu sha256.o

sha256.o: sha256.cu
	nvcc $(NVFLAGS) -c sha256.cu

clean:
	rm -rf hw4_gpu *.o




