#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include "FullSearchBM_cuda_naive.h"

using namespace std;

vector<vector<MotionVector>> fullSearchCUDANaiveGray(const ImageGray& curr, const ImageGray& ref, 
                                                        int blockSize, int searchRange){
    cudaSetDevice(0);
    // Calculate frame size and byte size of current and reference frames
    int frameSize = curr.width * curr.height;
    if (frameSize != ref.width * ref.height) {
        throw runtime_error("Current and reference frames must have the same dimensions.");
    }

    size_t bytesPerFrame = frameSize * sizeof(unsigned char);
    
    // Matrix of current and reference frames
    unsigned char* d_curr = nullptr;
    unsigned char* d_ref = nullptr;
    gpuErrorCheck(cudaMalloc((void**)&d_curr, bytesPerFrame));
    gpuErrorCheck(cudaMalloc((void**)&d_ref, bytesPerFrame));
    
    gpuErrorCheck(cudaMemcpy(d_curr, curr.data.data(), bytesPerFrame, cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_ref, ref.data.data(), bytesPerFrame, cudaMemcpyHostToDevice));
    
    // TODO: Implementare il processing su GPU
    
    // Libera la memoria GPU
    cudaFree(d_curr);
    cudaFree(d_ref);
    
    // Ritorna risultati (da completare)
    return vector<vector<MotionVector>>();
    }