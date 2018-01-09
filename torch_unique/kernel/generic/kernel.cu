#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/kernel.cu"
#else

#include <thrust/unique.h>

THCTensor *unique_kernel(THCState *state, THCudaLongTensor *index, THCTensor *input) {
  #if CUDA_VERSION >= 7000
    THCThrustAllocator thrustAlloc(state);
  #define THRUST_EXEC(fn, ...) fn(thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)), ##__VA_ARGS__)
  #else
  #define THRUST_EXEC(fn, ...) fn(##__VA_ARGS__)
  #endif

  return NULL;
}

#endif
