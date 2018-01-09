#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/cuda.c"
#else

THCTensor *unique(THCudaLongTensor *index, THCTensor *input) {
  return unique_kernel(state, index, input);
}

#endif
