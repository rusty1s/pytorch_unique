#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/kernel.cu"
#else

#include <thrust/unique.h>

THCTensor *unique_kernel(THCState *state, THCudaLongTensor *index, THCTensor *input) {
  return NULL;
}

#endif
