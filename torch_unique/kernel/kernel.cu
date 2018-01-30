#include <THC.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "kernel.h"

#include "THCThrustAllocator.cuh"
#include "THCNumerics.cuh"

#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

#define unique_(NAME) TH_CONCAT_4(unique, NAME, _kernel_, Real)

#if CUDA_VERSION >= 7000
#define THRUST_ALLOC(state) THCThrustAllocator thrustAlloc(state)
#define THRUST_EXEC(fn, ...) fn(thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)), ##__VA_ARGS__)
#else
#define THRUST_ALLOC(state)
#define THRUST_EXEC(fn, ...) fn(##__VA_ARGS__)
#endif

#include "generic/kernel.cu"
#include "THCGenerateAllTypes.h"
