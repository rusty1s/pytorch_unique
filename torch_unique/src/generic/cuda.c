#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/cuda.c"
#else

THCTensor *unique_cuda(THCudaLongTensor *index, THCTensor *input) {
  return unique_kernel(state, index, input);
}

#endif
