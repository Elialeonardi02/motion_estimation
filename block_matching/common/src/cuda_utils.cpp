#include "cuda_utils.h"
#include <iostream>
#include <iomanip>

// GPU MEMORY UTILITIES
namespace GPUMemoryUtils {

void allocateFrames(const ImageGray& curr, const ImageGray& ref,
                   unsigned char*& d_curr, unsigned char*& d_ref,
                   size_t& bytesPerFrame) {
  int currSize = curr.width * curr.height;
  int refSize = ref.width * ref.height;
  
  bytesPerFrame = currSize * sizeof(unsigned char);
  
  gpuErrorCheck(cudaMalloc(&d_curr, bytesPerFrame));
  gpuErrorCheck(cudaMalloc(&d_ref, bytesPerFrame));
}

void copyFramesToGPU(unsigned char* d_curr, unsigned char* d_ref,
                    const ImageGray& curr, const ImageGray& ref,
                    size_t bytesPerFrame) {
  gpuErrorCheck(cudaMemcpy(d_curr, curr.data.data(), bytesPerFrame, cudaMemcpyHostToDevice));
  gpuErrorCheck(cudaMemcpy(d_ref, ref.data.data(), bytesPerFrame, cudaMemcpyHostToDevice));
}

void copyMotionVectorsFromGPU(std::vector<MotionVector>& h_mv_flat,
                              MotionVector* d_mv,
                              int blocksX, int blocksY) {
  size_t mvBytes = (size_t)blocksX * blocksY * sizeof(MotionVector);
  gpuErrorCheck(cudaMemcpy(h_mv_flat.data(), d_mv, mvBytes, cudaMemcpyDeviceToHost));
}

void freeMemory(unsigned char* d_curr, unsigned char* d_ref,
               MotionVector* d_mv) {
  if (d_curr) cudaFree(d_curr);
  if (d_ref) cudaFree(d_ref);
  if (d_mv) cudaFree(d_mv);
}

void freeMemory(unsigned char* d_curr, unsigned char* d_ref,
               int* d_sad, MotionVector* d_mv) {
  if (d_curr) cudaFree(d_curr);
  if (d_ref) cudaFree(d_ref);
  if (d_sad) cudaFree(d_sad);
  if (d_mv) cudaFree(d_mv);
}

void freeMemory(unsigned char* d_curr, unsigned char* d_ref,
               void* d_extra, MotionVector* d_mv) {
  if (d_curr) cudaFree(d_curr);
  if (d_ref) cudaFree(d_ref);
  if (d_extra) cudaFree(d_extra);
  if (d_mv) cudaFree(d_mv);
}

}

// CUDA TIMER
CudaTimer::CudaTimer(const std::string& timerName) 
    : name(timerName), isRecording(false) {
  createCudaEvent(startEvent);
  createCudaEvent(stopEvent);
}

CudaTimer::~CudaTimer() {
  destroyCudaEvent(startEvent);
  destroyCudaEvent(stopEvent);
}

void CudaTimer::start() {
  recordCudaEvent(startEvent);
  isRecording = true;
}

float CudaTimer::stop() {
  if (!isRecording) {
    return 0.0f;
  }
  recordCudaEvent(stopEvent);
  isRecording = false;
  return elapsedCudaTime(startEvent, stopEvent);
}

void CudaTimer::reset() {
  isRecording = false;
}


// CUDA TIMING UTILITIES
namespace CudaTimingUtils {

void printKernelTiming(const std::string& kernelName, float milliseconds) {
  std::cout << "  " << kernelName << ": " << std::fixed << std::setprecision(4) 
      << milliseconds << " ms" << std::endl;
}

}
