#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/cuda.c"
#else

void unique_(single)(THCTensor *input) {
  return unique_kernel_(single)(state, input);
}

void unique_(byKey)(THCTensor *key, THCTensor *value) {
  return unique_kernel_(byKey)(state, key, value);
}

#endif
