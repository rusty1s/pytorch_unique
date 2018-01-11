#include <THC.h>

#include "kernel.h"

#include "THCThrustAllocator.cuh"

#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

#define unique_(NAME) TH_CONCAT_4(unique_, NAME, _kernel_, Real)

#if CUDA_VERSION >= 7000
#define THRUST_ALLOC(state) THCThrustAllocator thrustAlloc(state)
#define THRUST_EXEC(fn, ...) fn(thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)), ##__VA_ARGS__)
#else
#define THRUST_ALLOC(state)
#define THRUST_EXEC(fn, ...) fn(##__VA_ARGS__)
#endif

#include "generic/kernel.cu"
#include "THCGenerateFloatType.h"
#include "generic/kernel.cu"
#include "THCGenerateDoubleType.h"
#include "generic/kernel.cu"
#include "THCGenerateByteType.h"
#include "generic/kernel.cu"
#include "THCGenerateCharType.h"
#include "generic/kernel.cu"
#include "THCGenerateShortType.h"
#include "generic/kernel.cu"
#include "THCGenerateIntType.h"
#include "generic/kernel.cu"
#include "THCGenerateLongType.h"
